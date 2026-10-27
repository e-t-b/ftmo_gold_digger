"""Offline decision on historical CLOSED-bar snapshots. No MT5 orders.

Example:
  python -m ftmo_gold.analyze --input data/raw --asof 1793116800
  python -m ftmo_gold.analyze --input data/raw --asof 1793116800 --model gpt-4.1-mini

--model makes an explicit paid API call, never automatically.
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
                selected.append({"time":timestamp,**{k:float(row[k]) for k in ("open","high","low","close")}})
        frames[tf]=selected[-62:]
    return frames

def analyze(input_dir,asof,spread,model=None):
    obs=observe(history(input_dir,asof),spread,closed_asof=asof)
    decision=FreeDecisionAgent(model).decide(obs)
    return {"observation":obs,"decision":decision.json(),
            "mode":"RESEARCH_NO_ORDERS","disclaimer":"No evidence of profitable edge"}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default="data/raw")
    parser.add_argument("--asof",type=int,required=True,help="Decision time: UNIX UTC seconds")
    parser.add_argument("--spread",type=float,required=True,help="Observed bid/ask spread, XAUUSD dollars/oz")
    parser.add_argument("--model",default=None,help="Enable optional OpenAI API model; requires OPENAI_API_KEY")
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    adapter=None
    if args.model:
        from .model_openai import OpenAIResearchModel
        adapter=OpenAIResearchModel(model=args.model)
    result=analyze(args.input,args.asof,args.spread,adapter)
    serialized=json.dumps(result,indent=2)
    if args.output:
        output=Path(args.output)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(serialized,encoding="utf-8")
    print(serialized)

if __name__=="__main__":
    main()
