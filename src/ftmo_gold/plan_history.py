"""Resumable pre-holdout generation of cached hourly Sol plans."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

from .decision import observe
from .market import STEP
from .sol_paper import LocalCallBudget
from .sol_plan import SolPlanModel


def load_frames(root):
    frames={}
    for timeframe in ("M5","M15","H1"):
        rows=[]
        with (Path(root)/(timeframe.lower()+".csv")).open(newline="",encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                row={"time":int(raw["time"]),**{
                    key:float(raw[key]) for key in ("open","high","low","close")}}
                for key in ("tick_volume","real_volume","volume"):
                    if raw.get(key) not in (None,""):
                        row[key]=float(raw[key])
                rows.append(row)
        frames[timeframe]=rows
    return frames


def historical_observation(frames,asof,spread):
    selected={}
    for timeframe in ("M5","M15","H1"):
        rows=frames[timeframe]
        opens=[int(row["time"]) for row in rows]
        index=bisect.bisect_right(opens,int(asof)-STEP[timeframe])
        selected[timeframe]=rows[max(0,index-62):index]
    return observe(selected,spread,closed_asof=int(asof))


def candidate_times(frames,start,end):
    return [int(row["time"])+STEP["H1"] for row in frames["H1"]
            if int(start)<=int(row["time"])+STEP["H1"]<=int(end)]


def main():
    parser=argparse.ArgumentParser(description="Cache pre-holdout hourly Sol plans in bounded batches")
    parser.add_argument("--input",required=True);parser.add_argument("--start",type=int,required=True)
    parser.add_argument("--end",type=int,required=True);parser.add_argument("--protected-start",type=int,required=True)
    parser.add_argument("--spread",type=float,required=True);parser.add_argument("--out",required=True)
    parser.add_argument("--max-plans",type=int,default=5)
    parser.add_argument("--max-calls-5h",type=int,default=5)
    parser.add_argument("--budget-state",default="reports/sol_paper/call_budget.json")
    parser.add_argument("--reasoning-effort",default="medium",
                        choices=("none","low","medium","high","xhigh","max"))
    args=parser.parse_args()
    if not args.start<args.end<args.protected_start:
        parser.error("Historical batch must end strictly before protected-start")
    if args.spread<0 or not 1<=args.max_plans<=20:
        parser.error("Invalid spread or max-plans")
    frames=load_frames(args.input);output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
    completed=set()
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value=json.loads(line);completed.add(int(value["plan"]["created_at"]))
    model=SolPlanModel(reasoning_effort=args.reasoning_effort)
    budget=LocalCallBudget(args.budget_state,args.max_calls_5h)
    written=0
    for asof in candidate_times(frames,args.start,args.end):
        if asof in completed:
            continue
        if not budget.record():
            break
        try:
            observation=historical_observation(frames,asof,args.spread)
            plan=model.create_plan(observation,{"mode":"historical_pre_holdout",
                "risk_pct":2,"max_daily_entries":2,"reward_risk_range":[1,4]})
            value={"status":"historical_sol_hour_plan","plan":plan}
        except Exception as exc:
            value={"status":"historical_plan_error","asof":asof,
                   "error_type":type(exc).__name__}
            error_path=output.with_suffix(output.suffix+".errors")
            with error_path.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(value,separators=(",",":"))+"\n")
            continue
        with output.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(value,separators=(",",":"))+"\n")
        written+=1
        if written>=args.max_plans:
            break
    print(json.dumps({"plans_written":written,"output":str(output),
                      "protected_start":args.protected_start}))


if __name__=="__main__":
    main()
