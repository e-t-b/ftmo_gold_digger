"""Compare legacy and masked decisions on validation data only."""

import argparse
import json
from pathlib import Path

import pandas as pd

from checkpoint_io import safe_torch_load
from gold_rl_core import Market, Policy
from train_massive import evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default="data/gold_3y")
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()

    checkpoint=safe_torch_load(args.checkpoint)
    if checkpoint.get("schema") not in {
        "8action-ppo-v1","8action-ppo-v2-mask","8action-ppo-v2-mask-robust",
        "8action-ppo-v3-batched-gae","8action-ppo-v4-terminal-trade-limit",
        "8action-ppo-v5-risk-cap","8action-ppo-v6-position-context",
        "8action-ppo-v7-causal-indicators","8action-ppo-v8-feature-sets",
        "8action-ppo-v9-configurable-capacity"}:
        raise ValueError("Unsupported checkpoint schema")
    hidden=int(checkpoint["policy"]["enc.0.weight_hh_l0"].shape[1])
    state_features=int(checkpoint["policy"]["shared.0.weight"].shape[1]-hidden*5)
    input_features=int(checkpoint["policy"]["enc.0.weight_ih_l0"].shape[1])
    policy=Policy(hidden=hidden,state_features=state_features,input_features=input_features)
    policy.load_state_dict(checkpoint["policy"],strict=True)
    market=Market(args.input)
    t1=int(pd.Timestamp("2026-01-01",tz="UTC").timestamp())
    t2=int(pd.Timestamp("2026-04-01",tz="UTC").timestamp())
    validation_days=[d for d in market.keys if t1+86400<=d*86400<t2-86400]
    if len(validation_days)!=62:raise ValueError("Validation split mismatch")

    legacy,_=evaluate(market,validation_days,checkpoint["mean"],checkpoint["std"],
        policy,use_action_mask=False)
    masked,_=evaluate(market,validation_days,checkpoint["mean"],checkpoint["std"],
        policy,use_action_mask=True)
    report={"checkpoint_episode":checkpoint.get("episode"),"days":len(validation_days),
        "selection_data_only":True,"legacy":legacy,"masked_without_retraining":masked}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
