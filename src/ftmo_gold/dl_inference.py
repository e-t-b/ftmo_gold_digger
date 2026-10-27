"""Offline inference from a trained multi-timeframe PyTorch checkpoint.

Strict historical as-of timestamp, no future/partial bars, no broker orders.
Model probabilities are research context, NOT calibrated trading advice.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .deep_learning import GoldGRU, load_frame, FEATURES, TF_SECONDS, CORE_TIMEFRAMES

def infer(input_dir, checkpoint_path, asof_utc):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    tfs = tuple(config["timeframes"])
    if not tfs or any(tf not in TF_SECONDS for tf in tfs):
        raise ValueError("Invalid checkpoint timeframes")
    if tuple(config["features"]) != FEATURES:
        raise ValueError("Feature schema mismatch")
    model = GoldGRU(nfeatures=len(FEATURES), branches=len(tfs))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    input_dir = Path(input_dir)
    x = []
    newest_closes = {}
    for i, tf in enumerate(tfs):
        df = load_frame(input_dir / (tf + ".csv"), tf)
        # Only use candles completely closed at the query instant.
        eligible = df[df.close_ts <= asof_utc]
        n = int(config["lengths"][tf])
        if len(eligible) < n + 1:
            raise ValueError(f"{tf}: insufficient historical candles")
        subset = eligible.iloc[-n:]
        closes = subset.close_ts.to_numpy(dtype=np.int64)
        opens = subset.time.to_numpy(dtype=np.int64)
        if np.any(closes > asof_utc):
            raise ValueError(f"{tf}: future candle")
        if tf in CORE_TIMEFRAMES and np.any(np.diff(opens) != TF_SECONDS[tf]):
            raise ValueError(f"{tf}: noncontiguous history")
        # Broker H4/D1 rollovers may vary; never synthesize missing bars.
        if asof_utc - int(closes[-1]) > max(TF_SECONDS[tf]*2, 86400 if tf in ("h4","d1") else 0):
            raise ValueError(f"{tf}: stale input")
        vals = subset[list(FEATURES)].to_numpy(dtype=np.float32)
        mean = checkpoint["mean"][i].numpy().reshape(1, -1)
        std = checkpoint["std"][i].numpy().reshape(1, -1)
        if not np.isfinite(vals).all() or not np.isfinite(mean).all() or np.any(std<=0):
            raise ValueError(f"{tf}: invalid model features/scaler")
        vals = np.clip((vals - mean)/std, -12, 12)
        x.append(torch.from_numpy(vals.astype(np.float32)).unsqueeze(0))
        newest_closes[tf] = int(closes[-1])
    with torch.no_grad():
        probabilities = model(*x).softmax(dim=1)[0].tolist()
    return {"asof_utc":int(asof_utc),"model_timeframes":list(tfs),
            "last_closed_utc":newest_closes,
            "probabilities":{"down":round(probabilities[0],6),
                             "neutral":round(probabilities[1],6),
                             "up":round(probabilities[2],6)},
            "status":"RESEARCH_ONLY_NOT_CALIBRATED",
            "warning":"These numbers are not a probability of profitable trades."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/raw")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--asof",type=int,required=True)
    p.add_argument("--out",default="reports/inference.json")
    args=p.parse_args()
    result=infer(args.input,args.checkpoint,args.asof)
    output=Path(args.out)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
