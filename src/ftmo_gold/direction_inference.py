"""Inference for the experimental BINARY 30-minute direction GRU checkpoint.

Compatible with the checkpoint produced by train_gold_direction_v2.py.
Distinct from the earlier 3-class `dl_inference.py` artifact format.
No orders. No claim of profitable decision-making.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .deep_learning import load_frame, FEATURES, TF_SECONDS, CORE_TIMEFRAMES


class DirectionGRU(nn.Module):
    def __init__(self, hidden=32, branches=5):
        super().__init__()
        self.branches=nn.ModuleList([nn.GRU(7,hidden,batch_first=True) for _ in range(branches)])
        self.head=nn.Sequential(nn.LayerNorm(hidden*branches),
            nn.Linear(hidden*branches,64),nn.GELU(),nn.Dropout(.2),nn.Linear(64,2))

    def forward(self,*frames):
        if len(frames)!=len(self.branches):
            raise ValueError("Bad timeframe count")
        return self.head(torch.cat(
            [net(x)[1][-1] for net,x in zip(self.branches,frames)],dim=1))


def infer_direction(input_dir, checkpoint_path, asof_utc):
    ckpt=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    cfg=ckpt["config"]
    if cfg.get("objective")!="binary_30m_direction":
        raise ValueError("Not a direction-v2 checkpoint")
    if tuple(cfg.get("features",[]))!=FEATURES:
        raise ValueError("Wrong feature schema")
    tfs=cfg["timeframes"]
    if not tfs or any(tf not in TF_SECONDS for tf in tfs):
        raise ValueError("Wrong timeframes")
    model=DirectionGRU(hidden=cfg["hidden"],branches=len(tfs))
    model.load_state_dict(ckpt["state_dict"],strict=True)
    model.eval()
    input_dir=Path(input_dir)
    batch=[]
    last={}
    for i,tf in enumerate(tfs):
        df=load_frame(input_dir/(tf+".csv"),tf)
        valid=df[df.close_ts<=asof_utc]
        n=cfg["lengths"][tf]
        if len(valid)<n+1:
            raise ValueError(f"{tf}: not enough closed bars")
        selected=valid.iloc[-n:]
        closes=selected.close_ts.to_numpy(dtype=np.int64)
        opens=selected.time.to_numpy(dtype=np.int64)
        if tf in CORE_TIMEFRAMES and np.any(np.diff(opens)!=TF_SECONDS[tf]):
            raise ValueError(f"{tf}: gap in intraday sequence")
        max_age=TF_SECONDS[tf] if tf in CORE_TIMEFRAMES else (
            2*86400 if tf=="h4" else 4*86400)
        if asof_utc-int(closes[-1])>max_age:
            raise ValueError(f"{tf}: stale input")
        vals=selected[list(FEATURES)].to_numpy(dtype=np.float32)
        mu=ckpt["mean"][i].numpy().reshape(1,-1)
        std=ckpt["std"][i].numpy().reshape(1,-1)
        if not np.isfinite(vals).all() or not np.isfinite(mu).all() or (
            not np.isfinite(std).all()) or np.any(std<=0):
            raise ValueError(f"{tf}: invalid scaler/input")
        normalized=np.clip((vals-mu)/std,-12,12)
        batch.append(torch.from_numpy(normalized.astype(np.float32))[None,:,:])
        last[tf]=int(closes[-1])
    with torch.no_grad():
        p=model(*batch).softmax(-1)[0].tolist()
    return {"asof_utc":int(asof_utc),"model_timeframes":list(tfs),
            "last_closed_utc":last,"probabilities":{
                "down":round(p[0],6),"up":round(p[1],6)},
            "status":"RESEARCH_ONLY_NOT_CALIBRATED",
            "warning":"No proof of predictive edge or trading profitability"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--asof",required=True,type=int)
    parser.add_argument("--out",default="reports/direction_inference.json")
    args=parser.parse_args()
    prediction=infer_direction(args.input,args.checkpoint,args.asof)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(prediction,indent=2))
    print(json.dumps(prediction,indent=2))


if __name__=="__main__":main()
