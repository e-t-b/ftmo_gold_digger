"""Deterministic replay of cached Sol plans on bid-based M5 OHLC bars."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .backtest import load_bars, round_down_lots
from .sol_plan import immediate_plan_decision, trigger_plan


PRAGUE=ZoneInfo("Europe/Prague")


def load_plans(path):
    plans=[]
    with Path(path).open(encoding="utf-8") as stream:
        for number,line in enumerate(stream,1):
            if not line.strip():
                continue
            raw=json.loads(line)
            plan=raw.get("plan",raw)
            required={"plan_id","created_at","valid_until","mode","setups","confidence","reason"}
            if not isinstance(plan,dict) or not required<=set(plan):
                raise ValueError(f"Invalid cached plan on line {number}")
            plans.append(plan)
    plans.sort(key=lambda item:int(item["created_at"]))
    if any(int(right["created_at"])<=int(left["created_at"])
           for left,right in zip(plans,plans[1:])):
        raise ValueError("Cached plans must have unique increasing creation times")
    return plans


def replay(bars,plans,*,initial=10000.,spread=.25,slippage=.05,
           contract_size=100.,minimum_lot=.01,lot_step=.01,
           commission_per_lot_roundtrip=6.,risk_fraction=.02,
           max_daily_entries=2,max_hold_bars=288):
    """Replay immutable plans; trigger close precedes entry on next M5 open."""
    if len(bars)<3 or any(value<0 for value in (
        spread,slippage,commission_per_lot_roundtrip)):
        raise ValueError("Insufficient bars or invalid execution costs")
    balance=float(initial);equity=balance;peak=balance;max_dd=0.
    position=None;pending=None;trades=[];equity_curve=[]
    plan_index=0;available=[];consumed=set();triggered=0;rejected=0
    active_day=None;day_start=balance;entries_today=0;daily_breaches=0
    daily_halted=False;total_breach=False

    def close_position(price,timestamp,reason):
        nonlocal balance,position
        direction=1 if position["side"]=="BUY" else -1
        gross=(price-position["entry"])*direction*position["lots"]*contract_size
        fee=position["lots"]*commission_per_lot_roundtrip
        net=gross-fee;balance+=net
        trade={**position,"closed_at":timestamp,"exit":round(price,6),
               "exit_reason":reason,"gross_pnl":round(gross,2),
               "commission":round(fee,2),"net_pnl":round(net,2),
               "balance":round(balance,2)}
        trades.append(trade);position=None

    for index,bar in enumerate(bars):
        timestamp=int(bar["time"]);closed_at=timestamp+300
        day=datetime.fromtimestamp(timestamp,timezone.utc).astimezone(PRAGUE).date()
        if day!=active_day:
            active_day=day;day_start=equity;entries_today=0;daily_halted=False

        newly_available=[]
        while plan_index<len(plans) and int(plans[plan_index]["created_at"])<=closed_at:
            available.append(plans[plan_index]);plan_index+=1
            newly_available.append(available[-1])
        available=[plan for plan in available if int(plan["valid_until"])>=closed_at]

        if pending is not None and position is None:
            side=pending["side"]
            entry=float(bar["open"])+(spread+slippage if side=="BUY" else -slippage)
            stop=float(pending["stop"]);rr=float(pending["reward_risk"])
            valid=stop<entry if side=="BUY" else stop>entry
            daily_floor=day_start-initial*.05;total_floor=initial*.90
            if (valid and entries_today<max_daily_entries and not daily_halted
                    and not total_breach and equity>max(daily_floor,total_floor)):
                distance=abs(entry-stop)
                budget=equity*risk_fraction
                per_lot=distance*contract_size+commission_per_lot_roundtrip
                lots=round_down_lots(budget/per_lot if per_lot else 0,
                                     minimum_lot,lot_step)
                if lots>0:
                    target=entry+(distance*rr if side=="BUY" else -distance*rr)
                    position={"side":side,"entry":round(entry,6),"stop":stop,
                              "target":round(target,6),"lots":lots,"opened_at":timestamp,
                              "opened_index":index,"plan_id":pending["plan_id"],
                              "setup_label":pending["setup_label"],"reward_risk":rr}
                    entries_today+=1
                else:
                    rejected+=1
            else:
                rejected+=1
            pending=None

        if position is not None:
            side=position["side"]
            low=float(bar["low"]) if side=="BUY" else float(bar["low"])+spread
            high=float(bar["high"]) if side=="BUY" else float(bar["high"])+spread
            stop_hit=low<=position["stop"] if side=="BUY" else high>=position["stop"]
            target_hit=high>=position["target"] if side=="BUY" else low<=position["target"]
            timed=index-position["opened_index"]>=max_hold_bars
            if stop_hit or target_hit or timed:
                reason="stop" if stop_hit else "target" if target_hit else "max_hold"
                raw_exit=(position["stop"] if stop_hit else position["target"] if target_hit
                          else float(bar["close"])+(spread if side=="SELL" else 0.))
                price=raw_exit+(-slippage if side=="BUY" else slippage)
                close_position(price,closed_at,reason)

        if position:
            mark=float(bar["close"]) if position["side"]=="BUY" else float(bar["close"])+spread
            direction=1 if position["side"]=="BUY" else -1
            unreal=(mark-position["entry"])*direction*position["lots"]*contract_size
        else:
            unreal=0.
        equity=balance+unreal;peak=max(peak,equity)
        max_dd=max(max_dd,(peak-equity)/peak if peak else 0.)
        if equity<=day_start-initial*.05:
            if not daily_halted:
                daily_breaches+=1
            daily_halted=True
        if equity<=initial*.90:
            total_breach=True
        equity_curve.append(equity)

        if position is None and pending is None:
            candidates=[plan for plan in available if plan["plan_id"] not in consumed]
            if candidates:
                plan=max(candidates,key=lambda item:int(item["created_at"]))
                decision=(immediate_plan_decision(plan,bar,spread)
                          if plan in newly_available else
                          trigger_plan(plan,bars[index-1],bar,spread) if index>0 else None)
                if decision:
                    setup=next(item for item in plan["setups"]
                               if item["label"]==decision.strategy_reference)
                    pending={"side":decision.action,"stop":decision.stop,
                             "reward_risk":setup["reward_risk"],"plan_id":plan["plan_id"],
                             "setup_label":setup["label"]}
                    consumed.add(plan["plan_id"]);triggered+=1

    if position:
        last=bars[-1];side=position["side"]
        price=float(last["close"])+(spread+slippage if side=="SELL" else -slippage)
        close_position(price,int(last["time"])+300,"sample_end")
    wins=sum(trade["net_pnl"]>0 for trade in trades)
    losses=sum(trade["net_pnl"]<0 for trade in trades)
    profits=sum(max(trade["net_pnl"],0) for trade in trades)
    loss_value=-sum(min(trade["net_pnl"],0) for trade in trades)
    return {"initial":initial,"final":round(balance,2),"net_pnl":round(balance-initial,2),
            "cached_plans":len(plans),"triggered_plans":triggered,"rejected_entries":rejected,
            "closed_trades":len(trades),"wins":wins,"losses":losses,
            "win_rate":round(wins/len(trades),4) if trades else None,
            "profit_factor":round(profits/loss_value,4) if loss_value else None,
            "max_drawdown_pct":round(100*max_dd,4),"daily_loss_breaches":daily_breaches,
            "total_loss_breach":total_breach,
            "assumptions":"Cached immutable Sol plans; next-M5-open entries; fixed spread/slippage/commission; bid OHLC; conservative SL priority; no swaps, margin or tick-path claim",
            "transactions":trades}


def main():
    parser=argparse.ArgumentParser(description="Replay cached Sol plans on historical M5 bars")
    parser.add_argument("--m5",required=True);parser.add_argument("--plans",required=True)
    parser.add_argument("--out",required=True);parser.add_argument("--spread",type=float,required=True)
    parser.add_argument("--slippage",type=float,default=.05)
    parser.add_argument("--contract-size",type=float,default=100.)
    parser.add_argument("--min-lot",type=float,default=.01);parser.add_argument("--lot-step",type=float,default=.01)
    parser.add_argument("--commission",type=float,default=6.);parser.add_argument("--risk-pct",type=float,default=2.)
    args=parser.parse_args()
    result=replay(load_bars(args.m5),load_plans(args.plans),spread=args.spread,
                  slippage=args.slippage,contract_size=args.contract_size,
                  minimum_lot=args.min_lot,lot_step=args.lot_step,
                  commission_per_lot_roundtrip=args.commission,risk_fraction=args.risk_pct/100)
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({key:value for key,value in result.items() if key!="transactions"},indent=2))


if __name__=="__main__":
    main()
