"""Quota-aware GPT-5.6 Sol paper trader for an MT5 demo terminal.

It reads closed bars and journals proposals. There is intentionally no call to
``order_send`` in this module.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .decision import FreeDecisionAgent, observe
from .model_openai import OpenAISolDecisionModel
from .paper_portfolio import PaperPortfolio


WINDOW_SECONDS=5*60*60


def _rows(values):
    if values is None:
        raise RuntimeError("MT5 returned no rates")
    if hasattr(values,"dtype"):
        return [{key:(value.item() if hasattr(value,"item") else value)
                 for key,value in zip(values.dtype.names,row)} for row in values]
    return [dict(row) for row in values]


class LocalCallBudget:
    def __init__(self,path,max_calls=20):
        self.path=Path(path)
        self.max_calls=max_calls

    def _recent(self,now):
        if not self.path.exists():
            return []
        try:
            values=json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError,OSError):
            return []
        return sorted(float(value) for value in values if now-float(value)<WINDOW_SECONDS)

    def available(self,now=None):
        now=time.time() if now is None else float(now)
        recent=self._recent(now)
        spacing=WINDOW_SECONDS/self.max_calls
        return len(recent)<self.max_calls and (not recent or now-recent[-1]>=spacing)

    def record(self,now=None):
        now=time.time() if now is None else float(now)
        recent=self._recent(now)
        spacing=WINDOW_SECONDS/self.max_calls
        if len(recent)>=self.max_calls or (recent and now-recent[-1]<spacing):
            return False
        recent.append(now)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.path.with_suffix(self.path.suffix+".tmp")
        temporary.write_text(json.dumps(recent),encoding="utf-8")
        temporary.replace(self.path)
        return True


def _position_context(positions):
    selected=[]
    for position in positions or ():
        raw=position._asdict() if hasattr(position,"_asdict") else dict(position)
        selected.append({key:raw.get(key) for key in
                         ("type","volume","price_open","price_current","sl","tp","profit")})
    return selected


def collect_snapshot(mt5,symbol):
    mapping={
        "M5":mt5.TIMEFRAME_M5,
        "M15":mt5.TIMEFRAME_M15,
        "H1":mt5.TIMEFRAME_H1,
    }
    frames={}
    for name,timeframe in mapping.items():
        values=_rows(mt5.copy_rates_from_pos(symbol,timeframe,1,62))
        values.sort(key=lambda item:int(item["time"]))
        frames[name]=values
    tick=mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError("MT5 quote unavailable")
    bid=float(tick.bid);ask=float(tick.ask)
    if ask<bid:
        raise RuntimeError("MT5 returned a crossed quote")
    now=int(time.time())
    raw_quote_time=int(getattr(tick,"time",0))
    candidate=round((raw_quote_time-now)/3600)*3600
    server_offset=(candidate if candidate in (0,3600,7200,10800,14400)
                   and abs(raw_quote_time-now-candidate)<=120 else 0)
    for rows in frames.values():
        for row in rows:
            row["time"]=int(row["time"])-server_offset
    asof=int(frames["M5"][-1]["time"])+300
    quote_time=raw_quote_time-server_offset
    # Allow a few seconds of harmless terminal/OS clock skew after removing
    # the broker's whole-hour server offset.
    if quote_time<=0 or not -5<=now-quote_time<=120:
        raise RuntimeError("MT5 quote is stale")
    if not -300<=now-asof<=600:
        raise RuntimeError("MT5 closed bars are stale")
    market={"bid":bid,"ask":ask,"spread":ask-bid,
            "last_m5":frames["M5"][-1],"previous_m5":frames["M5"][-2],
            "last_m15":frames["M15"][-1],"quote_time":quote_time,
            "server_utc_offset_seconds":server_offset}
    return (observe(frames,ask-bid,closed_asof=asof),
            int(frames["M15"][-1]["time"]),market)


def decide_once(mt5,model,symbol,last_m15=None,*,collected=None,
                portfolio=None,symbol_info=None):
    observation,closed_m15,market=collected or collect_snapshot(mt5,symbol)
    if last_m15==closed_m15:
        return closed_m15,None
    account=mt5.account_info()
    if account is None:
        raise RuntimeError("MT5 account state unavailable")
    positions=mt5.positions_get(symbol=symbol) or () if portfolio is None else ()
    virtual_position=portfolio.state["position"] if portfolio else None
    account_context=({"balance":portfolio.state["balance"],
                      "equity":portfolio.state["equity"],
                      "margin_free":portfolio.state["equity"]}
                     if portfolio else
                     {"balance":float(account.balance),"equity":float(account.equity),
                      "margin_free":float(account.margin_free)})
    context={
        "symbol":symbol,
        "account":account_context,
        "positions":[virtual_position] if virtual_position else _position_context(positions),
        "risk_limits":{"max_risk_per_entry_pct":(
                           100*portfolio.risk_fraction if portfolio else .25),
                       "max_daily_entries":(
                           portfolio.max_daily_entries if portfolio else 3),
                       "reward_risk_range":[1.,4.],
                       "one_position_at_a_time":True},
    }
    decision=FreeDecisionAgent(model).decide(
        observation,position_open=bool(virtual_position or positions),trading_context=context)
    fill=None
    if portfolio:
        if symbol_info is None:
            raise RuntimeError("MT5 symbol metadata unavailable")
        fill=portfolio.apply_decision(decision,market["bid"],market["ask"],
                                      market["quote_time"],closed_m15,symbol_info,
                                      observation["snapshot_id"])
        portfolio.save()
    return closed_m15,{
        "status":"sol_virtual_paper" if portfolio else "sol_proposal_no_order",
        "model":"gpt-5.6-sol",
        "closed_m15":closed_m15,
        "snapshot_id":observation["snapshot_id"],
        "decision":decision.json(),
        "position_open":bool(portfolio.state["position"] if portfolio else positions),
        "virtual_fill":fill,
        "paper_metrics":portfolio.metrics() if portfolio else None,
        "server_utc_offset_seconds":market["server_utc_offset_seconds"],
    }


def main():
    parser=argparse.ArgumentParser(description="GPT-5.6 Sol MT5 demo paper trader")
    parser.add_argument("--symbol",default="XAUUSD")
    parser.add_argument("--interval",type=int,default=15)
    parser.add_argument("--reasoning-effort",default="medium",
                        choices=("none","low","medium","high","xhigh","max"))
    parser.add_argument("--max-calls-5h",type=int,default=20)
    parser.add_argument("--out",default="reports/sol_paper/decisions.jsonl")
    parser.add_argument("--budget-state",default="reports/sol_paper/call_budget.json")
    parser.add_argument("--portfolio-state",default="reports/sol_paper/portfolio.json")
    parser.add_argument("--commission-per-lot-roundtrip",type=float,default=0.)
    parser.add_argument("--max-hold-hours",type=float,default=24.)
    parser.add_argument("--risk-pct",type=float,default=.25)
    parser.add_argument("--max-daily-entries",type=int,default=3)
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    if (args.interval<5 or not 1<=args.max_calls_5h<=100
            or args.commission_per_lot_roundtrip<0 or args.max_hold_hours<=0):
        parser.error("interval must be >=5 seconds and max-calls-5h must be 1..100")
    if not 0<args.risk_pct<=2 or not 1<=args.max_daily_entries<=3:
        parser.error("risk-pct must be in (0,2] and max-daily-entries must be 1..3")
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    try:
        account=mt5.account_info()
        if account is None or account.trade_mode!=mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise RuntimeError("GPT-5.6 Sol is restricted to an MT5 demo account")
        model=OpenAISolDecisionModel(reasoning_effort=args.reasoning_effort)
        symbol_info=mt5.symbol_info(args.symbol)
        if symbol_info is None:
            raise RuntimeError("MT5 symbol metadata unavailable")
        portfolio=PaperPortfolio(args.portfolio_state,args.symbol,float(account.balance),
                                 risk_fraction=args.risk_pct/100,
                                 max_daily_entries=args.max_daily_entries,
                                 commission_per_lot_roundtrip=args.commission_per_lot_roundtrip,
                                 max_hold_hours=args.max_hold_hours)
        budget=LocalCallBudget(args.budget_state,args.max_calls_5h)
        output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
        previous=None
        while True:
            event=None
            try:
                collected=collect_snapshot(mt5,args.symbol)
                _,closed_m15,market=collected
                if closed_m15!=previous:
                    auto_close=portfolio.process_closed_bar(
                        market["last_m15"],closed_m15,market["bid"],market["ask"],
                        float(symbol_info.trade_contract_size),market["spread"])
                    if budget.record():
                        previous,event=decide_once(mt5,model,args.symbol,previous,
                                                   collected=collected,portfolio=portfolio,
                                                   symbol_info=symbol_info)
                        event["virtual_auto_close"]=auto_close
                    else:
                        previous=closed_m15
                        portfolio.save()
                        event={"status":"local_quota_guard_wait","closed_m15":closed_m15,
                               "virtual_auto_close":auto_close,
                               "paper_metrics":portfolio.metrics()}
            except Exception as exc:
                event={"status":"paper_error","error_type":type(exc).__name__}
                if args.once:
                    raise
            if event is not None:
                event["observed_utc"]=datetime.now(timezone.utc).isoformat()
                with output.open("a",encoding="utf-8") as stream:
                    stream.write(json.dumps(event,separators=(",",":"))+"\n")
                print(json.dumps(event),flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        mt5.shutdown()


if __name__=="__main__":
    main()
