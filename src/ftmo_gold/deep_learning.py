"""Causal multi-timeframe Gold GRU experiment (research only, no MT5 orders)."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TF_SECONDS = {"m5": 300, "m15": 900, "h1": 3600, "h4": 14400, "d1": 86400}
CORE_TIMEFRAMES = ("m5", "m15", "h1")
ALL_TIMEFRAMES = ("m5", "m15", "h1", "h4", "d1")
FEATURES = ("return", "range", "body", "upper_wick", "lower_wick", "log_volume", "spread_ratio")


def load_frame(path, timeframe):
    df = pd.read_csv(path)
    for col in ("time", "open", "high", "low", "close", "tick_volume", "spread"):
        if col not in df:
            raise ValueError(f"{path} missing {col}")
    if df.empty or not df.time.is_monotonic_increasing or df.time.duplicated().any():
        raise ValueError(f"{path}: unordered/duplicate/empty")
    if timeframe in CORE_TIMEFRAMES and (df.time.to_numpy(dtype=np.int64) % TF_SECONDS[timeframe] != 0).any():
        raise ValueError(f"{path}: misaligned timestamps")
    if not ((df.low <= df[["open", "close"]].min(axis=1)) &
            (df.high >= df[["open", "close"]].max(axis=1))).all():
        raise ValueError(f"{path}: invalid OHLC")
    if not np.isfinite(df[["open", "high", "low", "close", "tick_volume", "spread"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: non-finite")
    prev = df.close.shift(1)
    denom = prev.where(prev > 0)
    df["return"] = np.log(df.close / denom)
    df["range"] = (df.high - df.low) / denom
    df["body"] = (df.close - df.open) / denom
    df["upper_wick"] = (df.high - df[["open", "close"]].max(axis=1)) / denom
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df.low) / denom
    df["log_volume"] = np.log1p(df.tick_volume.clip(lower=0))
    # MT5 spread is points. XAUUSD metadata point=0.01 USD for this dataset.
    df["spread_ratio"] = df.spread / (denom * 100)
    df["close_ts"] = df.time + TF_SECONDS[timeframe]
    return df


def assemble(directory, sequence=32, horizon=6, cost_dollars=0.82, timeframes=CORE_TIMEFRAMES):
    if sequence < 8 or horizon < 1 or cost_dollars < 0:
        raise ValueError("Invalid sequence, horizon or cost")
    directory = Path(directory)
    frames = {tf: load_frame(directory / f"{tf}.csv", tf) for tf in timeframes}
    frames = {tf: (df.reset_index(drop=True), df[list(FEATURES)].to_numpy(dtype=np.float32),
                   df.close_ts.to_numpy(dtype=np.int64), df.time.to_numpy(dtype=np.int64))
              for tf, df in frames.items()}
    m5 = frames["m5"][0]
    xs, ys, times, moves = [], [], [], []
    lengths = {"m5": sequence, "m15": 16, "h1": 6, "h4": 12, "d1": 10}
    for idx in range(sequence, len(m5) - horizon):
        at = int(m5.close_ts.iloc[idx])
        if not (np.diff(m5.time.iloc[idx:idx + horizon + 1].to_numpy(dtype=np.int64)) == 300).all():
            continue
        chunks = []
        for tf in timeframes:
            df, vals, closes, opens = frames[tf]
            j = np.searchsorted(closes, at, side="right") - 1
            start = j - lengths[tf] + 1
            if start < 1 or j >= len(df):
                break
            if tf in CORE_TIMEFRAMES and not (np.diff(opens[start:j + 1]) == TF_SECONDS[tf]).all():
                break
            chunk = vals[start:j + 1]
            if not np.isfinite(chunk).all():
                break
            chunks.append(chunk)
        if len(chunks) != len(timeframes):
            continue
        move = float(m5.close.iloc[idx + horizon] - m5.close.iloc[idx])
        xs.append(chunks)
        ys.append(2 if move > cost_dollars else 0 if move < -cost_dollars else 1)
        times.append(at)
        moves.append(move)
    if not xs:
        raise ValueError("No aligned windows: verify historical coverage and gaps")
    return ([np.stack([row[i] for row in xs]) for i in range(len(timeframes))],
            np.array(ys, dtype=np.int64), np.array(times, dtype=np.int64),
            np.array(moves, dtype=np.float32))


def chronological_split(times, train=.65, val=.15, embargo_seconds=3600):
    if not (0 < train < 1 and 0 < val < 1 and train + val < 1):
        raise ValueError("Bad split proportions")
    lo, hi = int(times.min()), int(times.max())
    t1 = lo + int((hi - lo) * train)
    t2 = lo + int((hi - lo) * (train + val))
    masks = [times < t1 - embargo_seconds,
             (times >= t1 + embargo_seconds) & (times < t2 - embargo_seconds),
             times >= t2 + embargo_seconds]
    if min(int(m.sum()) for m in masks) < 50:
        raise ValueError("Not enough windows per split")
    return masks


class GoldGRU(nn.Module):
    def __init__(self, nfeatures=7, hidden=24, branches=3):
        super().__init__()
        self.branches = nn.ModuleList([nn.GRU(nfeatures, hidden, batch_first=True) for _ in range(branches)])
        self.head = nn.Sequential(nn.Linear(hidden*branches, 48), nn.ReLU(),
                                  nn.Dropout(.15), nn.Linear(48, 3))

    def forward(self, *frames):
        if len(frames) != len(self.branches):
            raise ValueError("GRU branches do not match input timeframes")
        vectors = [net(v)[1][-1] for net, v in zip(self.branches, frames)]
        return self.head(torch.cat(vectors, dim=1))


def stats(y_true, probs):
    preds = probs.argmax(axis=1)
    cm = np.zeros((3,3), dtype=int)
    for a,b in zip(y_true,preds):
        cm[a,b] += 1
    return {
        "n": len(y_true),
        "accuracy": round(float((preds == y_true).mean()), 5),
        "majority_baseline": round(float(np.bincount(y_true,minlength=3).max()/len(y_true)),5),
        "confusion_rows_true_columns_predicted": cm.tolist(),
        "class_counts": np.bincount(y_true,minlength=3).tolist(),
        "cross_entropy": round(float(-np.log(np.clip(probs[np.arange(len(y_true)),y_true],1e-8,1)).mean()),5),
        "brier": round(float(np.square(probs-np.eye(3)[y_true]).sum(axis=1).mean()),5),
    }


def run(input_dir, out_dir, epochs=6, batch=128, lr=.001, seed=42,
        cost_dollars=.82, sequence=32, horizon=6, use_higher_timeframes=False):
    torch.set_num_threads(2)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    timeframes = ALL_TIMEFRAMES if use_higher_timeframes else CORE_TIMEFRAMES
    x,y,times,_ = assemble(input_dir,sequence,horizon,cost_dollars,timeframes=timeframes)
    masks=chronological_split(times,embargo_seconds=max(3600,6*3600+horizon*300))
    mu=[v[masks[0]].mean(axis=(0,1),keepdims=True) for v in x]
    sigma=[v[masks[0]].std(axis=(0,1),keepdims=True).clip(min=1e-6) for v in x]
    x=[np.clip((v-m)/s,-12,12).astype(np.float32) for v,m,s in zip(x,mu,sigma)]
    model=GoldGRU(nfeatures=x[0].shape[-1],branches=len(x))
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    loss_fn=nn.CrossEntropyLoss()
    loader=DataLoader(
        TensorDataset(*(torch.from_numpy(v[masks[0]]) for v in x),
                      torch.from_numpy(y[masks[0]])),
        batch_size=batch, shuffle=True)
    def predict(mask):
        model.eval()
        with torch.no_grad():
            ids=np.flatnonzero(mask)
            return np.concatenate([
                model(*(torch.from_numpy(v[ids[start:start+512]]) for v in x)).softmax(-1).numpy()
                for start in range(0,len(ids),512)])
    best=float("inf"); saved=None; history=[]
    for epoch in range(epochs):
        model.train(); running=0.; seen=0
        for batch_data in loader:
            *xb,yb=batch_data
            optimizer.zero_grad()
            loss=loss_fn(model(*xb),yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            running+=float(loss.item())*len(yb);seen+=len(yb)
        p=predict(masks[1])
        val_ce=stats(y[masks[1]],p)["cross_entropy"]
        history.append({"epoch":epoch+1,"train_loss":round(running/seen,5),"val_ce":val_ce})
        if val_ce < best:
            best=val_ce
            saved={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(saved)
    labels=("train","validation","test")
    scores={name:stats(y[mask],predict(mask)) for name,mask in zip(labels,masks)}
    ranges={name:{"first_utc":pd.to_datetime(times[mask][0],unit="s",utc=True).isoformat(),
                  "last_utc":pd.to_datetime(times[mask][-1],unit="s",utc=True).isoformat()}
            for name,mask in zip(labels,masks)}
    output=Path(out_dir); output.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":saved,
                "mean":[torch.from_numpy(a) for a in mu],
                "std":[torch.from_numpy(a) for a in sigma],
                "config":{"features":list(FEATURES),"timeframes":list(timeframes),
                          "sequence":sequence,"lengths":{"m5":sequence,"m15":16,"h1":6,"h4":12,"d1":10},
                          "horizon":horizon,"cost_dollars":cost_dollars}},
               output/"model.pt")
    report={"task":"3-class 30-minute close movement (NOT trading profitability)",
            "architecture":f"{len(timeframes)}-branch GRU hidden24, 48-unit MLP","seed":seed,
            "training":{"epochs":epochs,"best_validation_ce":best,"history":history},
            "cost_assumption_price_usd_per_oz":cost_dollars,
            "splits":ranges,"scores":scores,"samples":len(y),
            "warning":"No simulated fills, swaps, broker fees, FTMO pass probability or profitability claim."}
    (output/"report.json").write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/raw")
    p.add_argument("--out",default="reports/gru_v1")
    p.add_argument("--epochs",type=int,default=6)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--cost-dollars",type=float,default=.82)
    p.add_argument("--sequence",type=int,default=32)
    p.add_argument("--horizon",type=int,default=6)
    p.add_argument("--with-h4-d1",action="store_true",help="Use all five timeframes; requires h4.csv and d1.csv")
    args=p.parse_args()
    print(json.dumps(run(args.input,args.out,epochs=args.epochs,seed=args.seed,
                         cost_dollars=args.cost_dollars,
                         sequence=args.sequence,horizon=args.horizon,
                         use_higher_timeframes=args.with_h4_d1),indent=2))


if __name__=="__main__":
    main()
