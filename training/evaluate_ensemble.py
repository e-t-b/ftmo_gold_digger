"""Evaluate an ensemble of independently trained policies on validation only."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from checkpoint_io import safe_torch_load
from gold_rl_core import Market, Policy
from train_massive import evaluate


class EnsemblePolicy(nn.Module):
    def __init__(self, policies):
        super().__init__();self.policies=nn.ModuleList(policies)

    def forward(self, windows, state):
        outputs=[policy(windows,state) for policy in self.policies]
        logits=torch.stack([item[0] for item in outputs]).mean(dim=0)
        values=torch.stack([item[1] for item in outputs]).mean(dim=0)
        return logits,values


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default="data/gold_3y")
    parser.add_argument("--checkpoint",action="append",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--max-risk-fraction",type=float,default=.25)
    args=parser.parse_args()
    policies=[];members=[]
    for path in args.checkpoint:
        checkpoint=safe_torch_load(path)
        policy=Policy();policy.load_state_dict(checkpoint["policy"],strict=True)
        policy.eval();policies.append(policy)
        members.append({"path":str(path),"episode":checkpoint["episode"],
            "validation":checkpoint["validation"],
            "selection_score":checkpoint.get("selection_score")})
    market=Market(args.input)
    t1=int(pd.Timestamp("2026-01-01",tz="UTC").timestamp())
    t2=int(pd.Timestamp("2026-04-01",tz="UTC").timestamp())
    days=[d for d in market.keys if t1+86400<=d*86400<t2-86400]
    mean=safe_torch_load(args.checkpoint[0])["mean"]
    std=safe_torch_load(args.checkpoint[0])["std"]
    result,_=evaluate(market,days,mean,std,EnsemblePolicy(policies),
        max_risk_fraction=args.max_risk_fraction)
    report={"selection_data_only":True,"members":members,"ensemble":result}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
