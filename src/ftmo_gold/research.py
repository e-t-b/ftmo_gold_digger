"""Read-only data audit and signal frequency report. No fills, trades or profit claims."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from .market import STEP,audit_bars,audit_ticks,features
from .strategies import candidates,VERSION

def load(path):
    with open(path,newline="",encoding="utf-8") as file:
        return list(csv.DictReader(file))

def inspect(directory,stride=1):
    root=Path(directory)
    audits={}
    for tf in STEP:
        path=root/(tf.lower()+".csv")
        if path.exists(): audits[tf]=audit_bars(load(path),tf)
    ticks=root/"ticks.csv"
    if ticks.exists(): audits["ticks"]=audit_ticks(load(ticks))
    bars=load(root/"m5.csv")
    regimes=Counter()
    strategies=Counter()
    contexts=Counter()
    for end in range(60,len(bars),stride):
        window=bars[max(0,end-62):end+1]
        state=features(window)
        if state is None: continue
        regimes[state["regime"]]+=1
        for candidate in candidates(window):
            strategies[candidate.name]+=1
            contexts[candidate.regime]+=1
    return {"strategy_version":VERSION,"sampling_stride":stride,"audits":audits,
            "regime_counts":dict(regimes),"candidate_counts":dict(strategies),
            "candidate_prior_regimes":dict(contexts),
            "note":"Signal counts ONLY. No backtested fills, fees, performance or FTMO validation."}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default="data/raw")
    parser.add_argument("--out",default="reports/market_research.json")
    args=parser.parse_args()
    result=inspect(args.input)
    output=Path(args.out)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="audits"},indent=2))

if __name__=="__main__": main()
