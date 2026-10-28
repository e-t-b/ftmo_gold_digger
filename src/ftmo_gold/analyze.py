"""Offline GPT-5.6 Sol decision on historical CLOSED-bar snapshots. No MT5 orders.

Example:
  python -m ftmo_gold.analyze --input data/raw --asof 1793116800
  python -m ftmo_gold.analyze --input data/raw --asof 1793116800 --chatgpt

--chatgpt consumes the connected ChatGPT plan. --paid-api is an explicit,
separate billing path. Neither mode can execute an order.
"""
import argparse
import csv
import json
from pathlib import Path
from .decision import FreeDecisionAgent,observe
from .market import STEP

def history(input_dir,asof):
    frames={}
    for tf in ("M5","M15","H1"):
        file=Path(input_dir)/(tf.lower()+".csv")
        with file.open(newline="",encoding="utf-8") as f:
            selected=[]
            for row in csv.DictReader(f):
                timestamp=int(row["time"])
                if timestamp+STEP[tf]>asof: continue
                item={"time":timestamp,**{k:float(row[k]) for k in ("open","high","low","close")}}
                for key in ("tick_volume","real_volume","volume"):
                    if row.get(key) not in (None,""):
                        item[key]=float(row[key])
                selected.append(item)
        frames[tf]=selected[-62:]
    return frames

def analyze(input_dir,asof,spread,model=None):
    obs=observe(history(input_dir,asof),spread,closed_asof=asof)
    decision=FreeDecisionAgent(model).decide(obs)
    return {"observation":obs,"decision":decision.json(),
            "mode":"GPT_5_6_SOL_PROPOSAL_NO_ORDERS" if model else "NO_MODEL_WAIT",
            "disclaimer":"No evidence of profitable edge; no order capability"}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default="data/raw")
    parser.add_argument("--asof",type=int,required=True,help="Decision time: UNIX UTC seconds")
    parser.add_argument("--spread",type=float,required=True,help="Observed bid/ask spread, XAUUSD dollars/oz")
    billing=parser.add_mutually_exclusive_group()
    billing.add_argument("--chatgpt",action="store_true",
                         help="Use the locally connected ChatGPT plan")
    billing.add_argument("--paid-api",action="store_true",
                         help="Explicitly use separately billed OPENAI_API_KEY")
    parser.add_argument("--reasoning-effort",default="medium",
                        choices=("none","low","medium","high","xhigh","max"))
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    adapter=None
    if args.chatgpt or args.paid_api:
        from .model_openai import OpenAISolDecisionModel
        adapter=(OpenAISolDecisionModel.paid_api(reasoning_effort=args.reasoning_effort)
                 if args.paid_api else
                 OpenAISolDecisionModel(reasoning_effort=args.reasoning_effort))
    result=analyze(args.input,args.asof,args.spread,adapter)
    serialized=json.dumps(result,indent=2)
    if args.output:
        output=Path(args.output)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(serialized,encoding="utf-8")
    print(serialized)

if __name__=="__main__":
    main()
