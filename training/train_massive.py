"""Resumable historical PPO training for Gold Digger; RESEARCH ONLY.

Usage:
  python training/train_massive.py --input data/gold_3y --out reports/rl_massive --episodes 10000
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

from gold_rl_core import Market, Policy, rollout, TFS


def evaluate(market, days, mean, std, policy):
    policy.eval()
    rows=[]
    for day in days:
        ep,*_=rollout(market,day,mean,std,policy,False)
        rows.append({"day":str(pd.to_datetime(day*86400,unit="s",utc=True).date()),
            "pnl":float(ep.equity-10000),"trades":ep.closed,"wins":ep.wins,
            "dd":float(ep.dd),"blocked":ep.blocked})
    policy.train()
    if not rows:raise ValueError("No validation days")
    a=np.array([r["pnl"] for r in rows]);ts=sum(r["trades"] for r in rows)
    result={"days":len(rows),"total_pnl":round(float(a.sum()),2),
        "avg_day":round(float(a.mean()),3),"trades":ts,
        "win_rate":round(sum(r["wins"] for r in rows)/max(1,ts),4),
        "worst_day":round(float(a.min()),2),
        "peak_day_dd":round(max(r["dd"] for r in rows),2),
        "profitable_days":int((a>0).sum()),"flat_days":int((a==0).sum())}
    return result,rows


def checkpoint(path,state):
    temp=path.with_suffix(".tmp")
    torch.save(state,temp)
    temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/gold_3y")
    p.add_argument("--out",default="reports/rl_massive")
    p.add_argument("--episodes",type=int,default=10000)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--update-epochs",type=int,default=1)
    p.add_argument("--save-every",type=int,default=25)
    p.add_argument("--validate-every",type=int,default=250)
    p.add_argument("--max-minutes",type=float,default=0,
                   help="0 means no runtime cap; resume after any clean stop")
    p.add_argument("--evaluate-test",action="store_true",
                   help="Run held-out test at end only, not during selection")
    args=p.parse_args()
    if min(args.episodes,args.update_epochs,args.save_every,args.validate_every)<1:
        p.error("Positive counts required")

    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2)
    market=Market(args.input)
    t1=int(pd.Timestamp("2026-01-01",tz="UTC").timestamp())
    t2=int(pd.Timestamp("2026-04-01",tz="UTC").timestamp())
    train_days=[d for d in market.keys if (d+1)*86400<t1-86400]
    val_days=[d for d in market.keys if t1+86400<=d*86400<t2-86400]
    test_days=[d for d in market.keys if d*86400>=t2+86400]
    if min(map(len,(train_days,val_days,test_days)))<10:
        raise RuntimeError("Insufficient split days")
    mean,std=market.fit_scaler(t1-86400)
    policy=Policy()
    optim=torch.optim.Adam(policy.parameters(),lr=3e-4)
    latest=out/"latest.pt";bestpath=out/"best_validation.pt"
    history=[];validation=[];best=-float("inf");start=0
    if latest.exists():
        ck=torch.load(latest,map_location="cpu",weights_only=False)
        if ck["seed"]!=args.seed or ck["schema"]!="8action-ppo-v1":
            raise ValueError("Checkpoint config mismatch")
        policy.load_state_dict(ck["policy"]);optim.load_state_dict(ck["optimizer"])
        random.setstate(ck["random"]);np.random.set_state(ck["numpy"])
        torch.set_rng_state(ck["torch"])
        history=ck["history"];validation=ck["validation"]
        best=ck["best"];start=ck["episode"]
        print(f"Resumed at episode {start}",flush=True)
    else:
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    started=time.monotonic()

    def save(done):
        checkpoint(latest,{"schema":"8action-ppo-v1","seed":args.seed,
            "episode":done,"policy":policy.state_dict(),
            "optimizer":optim.state_dict(),
            "random":random.getstate(),"numpy":np.random.get_state(),
            "torch":torch.get_rng_state(),"history":history,
            "validation":validation,"best":best,"mean":mean,"std":std,
            "lengths":{"m5":16,"m15":12,"h1":8,"h4":8,"d1":8},
            "timeframes":list(TFS)})
        summary={"episodes_completed":done,"episodes_target":args.episodes,
            "train_days":len(train_days),"validation_days":len(val_days),
            "test_days_held_out":len(test_days),
            "last_episodes":history[-20:],
            "validation_history":validation,
            "best_validation_mean_day_usd":best,
            "warning":"Research only; OHLC fills are approximate, no exact FTMO risk engine"}
        (out/"progress.json").write_text(json.dumps(summary,indent=2,
            default=lambda v: v.item() if hasattr(v,"item") else str(v)))

    done=start
    try:
        for n in range(start,args.episodes):
            day=random.choice(train_days)
            ep,obs,actions,oldlog,values,rewards=rollout(
                market,day,mean,std,policy,True)
            if len(rewards)>=10:
                returns=[];carry=0.
                for reward in rewards[::-1]:
                    carry=reward+.99*carry;returns.append(carry)
                ret=torch.as_tensor(returns[::-1],dtype=torch.float32)
                adv=ret-torch.as_tensor(values)
                adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-6)
                oldlog=torch.as_tensor(oldlog,dtype=torch.float32)
                actions=torch.as_tensor(actions,dtype=torch.long)
                for _ in range(args.update_epochs):
                    for ix in torch.randperm(len(actions)).split(64):
                        indexes=ix.tolist()
                        windows=[torch.from_numpy(np.stack(
                            [obs[k][0][j] for k in indexes])) for j in range(5)]
                        state=torch.from_numpy(np.stack(
                            [obs[k][1] for k in indexes]))
                        logits,val=policy(windows,state)
                        dist=Categorical(logits=logits)
                        logprob=dist.log_prob(actions[ix])
                        ratio=(logprob-oldlog[ix]).exp()
                        clipped=torch.minimum(ratio*adv[ix],
                            torch.clamp(ratio,.8,1.2)*adv[ix])
                        loss=-clipped.mean()+.3*(val-ret[ix]).square().mean()-.01*dist.entropy().mean()
                        optim.zero_grad();loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(),.5)
                        optim.step()
            history.append({"episode":n+1,"day":int(day),
                "steps":len(rewards),"pnl":round(ep.equity-10000,2),
                "trades":ep.closed})
            done=n+1
            if done%args.validate_every==0:
                score,_=evaluate(market,val_days,mean,std,policy)
                validation.append({"episode":done,**score})
                if score["avg_day"]>best:
                    best=score["avg_day"]
                    checkpoint(bestpath,{"policy":copy.deepcopy(policy.state_dict()),
                        "episode":done,"validation":score,"mean":mean,"std":std,
                        "schema":"8action-ppo-v1"})
                print("EVALUATION",done,score,flush=True)
            if done%args.save_every==0:
                save(done)
                print("SAVED",done,"last_episode_pnl",history[-1]["pnl"],flush=True)
            if args.max_minutes and (time.monotonic()-started)/60>=args.max_minutes:
                print("Runtime cap reached; saving.",flush=True)
                break
    except KeyboardInterrupt:
        print("Interrupted; checkpointing.",flush=True)
    finally:
        save(done)
    if args.evaluate_test and done>=args.episodes:
        score,rows=evaluate(market,test_days,mean,std,policy)
        (out/"final_test.json").write_text(json.dumps(
            {"episode":done,"test":score,"policy":"latest"},indent=2))
        pd.DataFrame(rows).to_csv(out/"test_days.csv",index=False)
        print("HELD_OUT_TEST",score,flush=True)
    print("DONE",done,"of",args.episodes,flush=True)


if __name__=="__main__":
    main()
