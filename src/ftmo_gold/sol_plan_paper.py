"""Hourly Sol plans with deterministic M5 virtual execution on MT5 demo."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .paper_portfolio import PaperPortfolio
from .sol_paper import LocalCallBudget, collect_snapshot
from .sol_plan import SolPlanModel, trigger_plan


def _append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(value,separators=(",",":"))+"\n")


def _save_active(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2),encoding="utf-8")
    temporary.replace(path)


def _load_active(path):
    path=Path(path)
    if not path.exists():
        return {"plan":None,"consumed":False,"source_h1_closed_at":None}
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict) or not {"plan","consumed","source_h1_closed_at"}<=set(value):
        raise RuntimeError("Invalid active plan state")
    return value


def main():
    parser=argparse.ArgumentParser(description="Sol hourly plans with causal M5 virtual fills")
    parser.add_argument("--symbol",default="XAUUSD");parser.add_argument("--interval",type=int,default=15)
    parser.add_argument("--reasoning-effort",default="medium",
                        choices=("none","low","medium","high","xhigh","max"))
    parser.add_argument("--max-calls-5h",type=int,default=5)
    parser.add_argument("--risk-pct",type=float,default=2.)
    parser.add_argument("--max-daily-entries",type=int,default=2)
    parser.add_argument("--commission-per-lot-roundtrip",type=float,default=6.)
    parser.add_argument("--max-hold-hours",type=float,default=24.)
    parser.add_argument("--events",default="reports/sol_plan_paper/events.jsonl")
    parser.add_argument("--plans",default="reports/sol_plan_paper/plans.jsonl")
    parser.add_argument("--active-plan",default="reports/sol_plan_paper/active_plan.json")
    parser.add_argument("--portfolio-state",default="reports/sol_plan_paper/portfolio.json")
    parser.add_argument("--budget-state",default="reports/sol_paper/call_budget.json")
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    if (args.interval<5 or not 1<=args.max_calls_5h<=20 or not 0<args.risk_pct<=2
            or not 1<=args.max_daily_entries<=2 or args.commission_per_lot_roundtrip<0
            or args.max_hold_hours<=0):
        parser.error("Invalid safety or execution setting")

    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    try:
        account=mt5.account_info()
        if account is None or account.trade_mode!=mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise RuntimeError("Sol plan paper mode requires an MT5 demo account")
        symbol_info=mt5.symbol_info(args.symbol)
        if symbol_info is None:
            raise RuntimeError("MT5 symbol metadata unavailable")
        model=SolPlanModel(reasoning_effort=args.reasoning_effort)
        budget=LocalCallBudget(args.budget_state,args.max_calls_5h)
        portfolio=PaperPortfolio(args.portfolio_state,args.symbol,float(account.balance),
            risk_fraction=args.risk_pct/100,max_daily_entries=args.max_daily_entries,
            commission_per_lot_roundtrip=args.commission_per_lot_roundtrip,
            max_hold_hours=args.max_hold_hours,min_reward_risk=1,max_reward_risk=4)
        active=_load_active(args.active_plan)
        previous_m5=None
        while True:
            events=[]
            try:
                observation,closed_m15,market=collect_snapshot(mt5,args.symbol)
                current=market["last_m5"];prior=market["previous_m5"]
                current_m5=int(current["time"])
                if current_m5!=previous_m5:
                    previous_m5=current_m5
                    auto_close=portfolio.process_closed_bar(current,current_m5,
                        market["bid"],market["ask"],float(symbol_info.trade_contract_size),
                        market["spread"],bar_seconds=300)
                    if auto_close:
                        events.append({"status":"virtual_close","trade":auto_close})
                    plan=active.get("plan")
                    if (plan and not active.get("consumed") and not portfolio.state["position"]
                            and int(current["time"])+300<=int(plan["valid_until"])):
                        decision=trigger_plan(plan,prior,current,market["spread"])
                        if decision:
                            fill=portfolio.apply_decision(decision,market["bid"],market["ask"],
                                market["quote_time"],current_m5,symbol_info,observation["snapshot_id"])
                            active["consumed"]=True;_save_active(args.active_plan,active)
                            events.append({"status":"m5_trigger","plan_id":plan["plan_id"],
                                "decision":decision.json(),"virtual_fill":fill})
                    h1_closed=int(observation["timeframes"]["H1"]["last_closed_at_utc"])
                    plan_expired=not active.get("plan") or int(active["plan"]["valid_until"])<=int(observation["asof_utc"])
                    new_h1=active.get("source_h1_closed_at")!=h1_closed
                    if not portfolio.state["position"] and new_h1 and plan_expired:
                        if budget.record():
                            context={"paper":portfolio.context(),"risk_pct":args.risk_pct,
                                     "max_daily_entries":args.max_daily_entries,
                                     "reward_risk_range":[1,4],"symbol":args.symbol}
                            plan=model.create_plan(observation,context)
                            active={"plan":plan,"consumed":False,"source_h1_closed_at":h1_closed}
                            _save_active(args.active_plan,active)
                            _append(args.plans,{"status":"sol_hour_plan","plan":plan,
                                "observed_utc":datetime.now(timezone.utc).isoformat()})
                            events.append({"status":"sol_hour_plan","plan":plan})
                        else:
                            events.append({"status":"local_quota_guard_wait","h1_closed":h1_closed})
                    portfolio.save()
                    for event in events:
                        event["observed_utc"]=datetime.now(timezone.utc).isoformat()
                        event["paper_metrics"]=portfolio.metrics()
                        event["server_utc_offset_seconds"]=market["server_utc_offset_seconds"]
                        _append(args.events,event);print(json.dumps(event),flush=True)
            except Exception as exc:
                event={"status":"paper_error","error_type":type(exc).__name__,
                       "observed_utc":datetime.now(timezone.utc).isoformat()}
                _append(args.events,event);print(json.dumps(event),flush=True)
                if args.once:
                    raise
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        mt5.shutdown()


if __name__=="__main__":
    main()
