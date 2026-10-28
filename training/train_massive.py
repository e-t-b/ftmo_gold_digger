"""Resumable historical PPO training for Gold Digger; RESEARCH ONLY.

Usage:
  python training/train_massive.py --input data/gold_3y --out reports/rl_massive --episodes 10000
"""
import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

from checkpoint_io import safe_torch_load
from gold_rl_core import Market, Policy, apply_action_mask, rollout, TFS

SCHEMA="8action-ppo-v6-position-context"


def evaluate(market, days, mean, std, policy, use_action_mask=True,selection_blocks=3,
             max_risk_fraction=1.):
    policy.eval()
    rows=[];action_counts=Counter();invalid_counts=Counter()
    for day in days:
        ep,*_=rollout(market,day,mean,std,policy,False,use_action_mask,max_risk_fraction)
        action_counts.update(ep.action_counts);invalid_counts.update(ep.invalid_action_counts)
        rows.append({"day":str(pd.to_datetime(day*86400,unit="s",utc=True).date()),
            "pnl":float(ep.equity-10000),"trades":ep.closed,"wins":ep.wins,
            "dd":float(ep.dd),"blocked":ep.blocked,
            "invalid_actions":sum(ep.invalid_action_counts.values()),
            "decision_steps":ep.decision_steps})
    policy.train()
    if not rows:raise ValueError("No validation days")
    a=np.array([r["pnl"] for r in rows]);ts=sum(r["trades"] for r in rows)
    result={"days":len(rows),"total_pnl":round(float(a.sum()),2),
        "avg_day":round(float(a.mean()),3),"trades":ts,
        "win_rate":round(sum(r["wins"] for r in rows)/max(1,ts),4),
        "worst_day":round(float(a.min()),2),
        "peak_day_dd":round(max(r["dd"] for r in rows),2),
        "profitable_days":int((a>0).sum()),"flat_days":int((a==0).sum())}
    result.update({"blocked_actions":sum(r["blocked"] for r in rows),
        "invalid_actions":sum(r["invalid_actions"] for r in rows),
        "decision_steps":sum(r["decision_steps"] for r in rows),
        "action_counts":dict(action_counts),
        "invalid_action_reasons":dict(invalid_counts),
        "action_masking":bool(use_action_mask)})
    blocks=[]
    for index,indexes in enumerate(np.array_split(np.arange(len(rows)),selection_blocks),1):
        values=a[indexes]
        blocks.append({"block":index,"days":len(indexes),
            "start_day":rows[int(indexes[0])]["day"],"end_day":rows[int(indexes[-1])]["day"],
            "total_pnl":round(float(values.sum()),2),
            "avg_day":round(float(values.mean()),3),
            "profitable_days":int((values>0).sum())})
    result["validation_blocks"]=blocks
    result["worst_block_avg_day"]=min(block["avg_day"] for block in blocks)
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
    p.add_argument("--update-epochs",type=int,default=4)
    p.add_argument("--rollout-batch-days",type=int,default=16)
    p.add_argument("--minibatch-size",type=int,default=256)
    p.add_argument("--save-every",type=int,default=128)
    p.add_argument("--validate-every",type=int,default=128)
    p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--gamma",type=float,default=.99)
    p.add_argument("--gae-lambda",type=float,default=.95)
    p.add_argument("--clip-range",type=float,default=.15)
    p.add_argument("--value-coef",type=float,default=.3)
    p.add_argument("--entropy-coef",type=float,default=.025)
    p.add_argument("--grad-clip",type=float,default=.5)
    p.add_argument("--max-risk-fraction",type=float,choices=(.25,.5,1.),default=.25)
    p.add_argument("--selection-blocks",type=int,default=3)
    p.add_argument("--selection-metric",choices=("worst_block_avg_day","avg_day"),
                   default="worst_block_avg_day")
    p.add_argument("--max-minutes",type=float,default=0,
                   help="0 means no runtime cap; resume after any clean stop")
    p.add_argument("--evaluate-test",action="store_true",
                   help="Run held-out test at end only, not during selection")
    args=p.parse_args()
    if min(args.episodes,args.update_epochs,args.rollout_batch_days,
           args.minibatch_size,args.save_every,args.validate_every)<1:
        p.error("Positive counts required")
    if args.selection_blocks<1 or args.learning_rate<=0 or not 0<args.gamma<=1:
        p.error("Invalid training or selection parameter")
    if not 0<args.clip_range<1 or min(args.value_coef,args.entropy_coef,args.grad_clip)<0:
        p.error("Invalid PPO coefficient")
    if not 0<=args.gae_lambda<=1:
        p.error("GAE lambda must be between zero and one")
    if args.save_every%args.rollout_batch_days or args.validate_every%args.rollout_batch_days:
        p.error("Save and validation intervals must be multiples of rollout-batch-days")

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
    optim=torch.optim.Adam(policy.parameters(),lr=args.learning_rate)
    latest=out/"latest.pt";bestpath=out/"best_validation.pt"
    history=[];validation=[];best=-float("inf");best_validation=None;start=0
    if latest.exists():
        ck=safe_torch_load(latest)
        if ck["seed"]!=args.seed or ck["schema"]!=SCHEMA:
            raise ValueError("Checkpoint config mismatch")
        expected_hyperparameters={"learning_rate":args.learning_rate,"gamma":args.gamma,
            "gae_lambda":args.gae_lambda,"update_epochs":args.update_epochs,
            "rollout_batch_days":args.rollout_batch_days,"minibatch_size":args.minibatch_size,
            "max_risk_fraction":args.max_risk_fraction,
            "clip_range":args.clip_range,"value_coef":args.value_coef,
            "entropy_coef":args.entropy_coef,"grad_clip":args.grad_clip,
            "selection_blocks":args.selection_blocks,"selection_metric":args.selection_metric}
        if ck.get("hyperparameters")!=expected_hyperparameters:
            raise ValueError("Checkpoint hyperparameter mismatch")
        policy.load_state_dict(ck["policy"],strict=True);optim.load_state_dict(ck["optimizer"])
        random.setstate(ck["random"]);np.random.set_state(ck["numpy"])
        torch.set_rng_state(ck["torch"])
        history=ck["history"];validation=ck["validation"]
        best=ck["best"];best_validation=ck.get("best_validation");start=ck["episode"]
        print(f"Resumed at episode {start}",flush=True)
    else:
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    started=time.monotonic()

    def save(done):
        hyperparameters={"learning_rate":args.learning_rate,"gamma":args.gamma,
            "gae_lambda":args.gae_lambda,"update_epochs":args.update_epochs,
            "rollout_batch_days":args.rollout_batch_days,"minibatch_size":args.minibatch_size,
            "max_risk_fraction":args.max_risk_fraction,
            "clip_range":args.clip_range,"value_coef":args.value_coef,
            "entropy_coef":args.entropy_coef,"grad_clip":args.grad_clip,
            "selection_blocks":args.selection_blocks,"selection_metric":args.selection_metric}
        checkpoint(latest,{"schema":SCHEMA,"seed":args.seed,
            "episode":done,"policy":policy.state_dict(),
            "optimizer":optim.state_dict(),
            "random":random.getstate(),"numpy":np.random.get_state(),
            "torch":torch.get_rng_state(),"history":history,
            "validation":validation,"best":best,"mean":mean,"std":std,
            "best_validation":best_validation,
            "hyperparameters":hyperparameters,"action_masking":True,
            "lengths":{"m5":16,"m15":12,"h1":8,"h4":8,"d1":8},
            "state_features":["position","marked_equity","marked_day_pnl","trade_fraction",
                "position_r","time_remaining"],
            "timeframes":list(TFS)})
        summary={"episodes_completed":done,"episodes_target":args.episodes,
            "train_days":len(train_days),"validation_days":len(val_days),
            "test_days_held_out":len(test_days),
            "last_episodes":history[-20:],
            "validation_history":validation,
            "best_validation_mean_day_usd":(
                best_validation["avg_day"] if best_validation else None),
            "best_selection_score":best,"selection_metric":args.selection_metric,
            "schema":SCHEMA,"hyperparameters":hyperparameters,
            "warning":"Research only; OHLC fills are approximate, no exact FTMO risk engine"}
        (out/"progress.json").write_text(json.dumps(summary,indent=2,
            default=lambda v: v.item() if hasattr(v,"item") else str(v)))

    pending=[]

    def update_policy():
        nonlocal pending
        if not pending:return
        batch_obs=[];batch_actions=[];batch_oldlog=[];batch_returns=[];batch_adv=[]
        for obs,actions,oldlog,values,rewards in pending:
            values=np.asarray(values,dtype=np.float32)
            rewards=np.asarray(rewards,dtype=np.float32)
            advantages=np.empty_like(rewards);carry=0.
            for index in range(len(rewards)-1,-1,-1):
                next_value=values[index+1] if index+1<len(values) else 0.
                delta=rewards[index]+args.gamma*next_value-values[index]
                carry=delta+args.gamma*args.gae_lambda*carry
                advantages[index]=carry
            batch_obs.extend(obs);batch_actions.extend(actions.tolist())
            batch_oldlog.extend(oldlog.tolist());batch_returns.extend((advantages+values).tolist())
            batch_adv.extend(advantages.tolist())
        advantages=torch.as_tensor(batch_adv,dtype=torch.float32)
        advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-6)
        oldlog=torch.as_tensor(batch_oldlog,dtype=torch.float32)
        actions=torch.as_tensor(batch_actions,dtype=torch.long)
        returns=torch.as_tensor(batch_returns,dtype=torch.float32)
        for _ in range(args.update_epochs):
            for ix in torch.randperm(len(actions)).split(args.minibatch_size):
                indexes=ix.tolist()
                windows=[torch.from_numpy(np.stack(
                    [batch_obs[k][0][j] for k in indexes])) for j in range(5)]
                state=torch.from_numpy(np.stack([batch_obs[k][1] for k in indexes]))
                masks=torch.from_numpy(np.stack(
                    [batch_obs[k][2] for k in indexes])).to(dtype=torch.bool)
                logits,value=policy(windows,state);logits=apply_action_mask(logits,masks)
                dist=Categorical(logits=logits);logprob=dist.log_prob(actions[ix])
                ratio=(logprob-oldlog[ix]).exp()
                clipped=torch.minimum(ratio*advantages[ix],torch.clamp(ratio,
                    1-args.clip_range,1+args.clip_range)*advantages[ix])
                loss=(-clipped.mean()+args.value_coef*(value-returns[ix]).square().mean()
                    -args.entropy_coef*dist.entropy().mean())
                optim.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(),args.grad_clip)
                optim.step()
        pending=[]

    done=start
    try:
        for n in range(start,args.episodes):
            day=random.choice(train_days)
            ep,obs,actions,oldlog,values,rewards=rollout(
                market,day,mean,std,policy,True,max_risk_fraction=args.max_risk_fraction)
            if len(rewards)>=10:
                pending.append((obs,actions,oldlog,values,rewards))
                if len(pending)>=args.rollout_batch_days:update_policy()
            history.append({"episode":n+1,"day":int(day),
                "steps":len(rewards),"pnl":round(ep.equity-10000,2),
                "trades":ep.closed})
            done=n+1
            if done%args.validate_every==0:
                update_policy()
                score,_=evaluate(market,val_days,mean,std,policy,
                    selection_blocks=args.selection_blocks,
                    max_risk_fraction=args.max_risk_fraction)
                validation.append({"episode":done,**score})
                candidate_score=score[args.selection_metric]
                if candidate_score>best:
                    best=candidate_score;best_validation=score
                    checkpoint(bestpath,{"policy":copy.deepcopy(policy.state_dict()),
                        "episode":done,"validation":score,"mean":mean,"std":std,
                        "schema":SCHEMA,"hyperparameters":{
                            "learning_rate":args.learning_rate,"gamma":args.gamma,
                            "gae_lambda":args.gae_lambda,"update_epochs":args.update_epochs,
                            "rollout_batch_days":args.rollout_batch_days,
                            "minibatch_size":args.minibatch_size,
                            "max_risk_fraction":args.max_risk_fraction,
                            "clip_range":args.clip_range,"value_coef":args.value_coef,
                            "entropy_coef":args.entropy_coef,"grad_clip":args.grad_clip,
                            "selection_blocks":args.selection_blocks,
                            "selection_metric":args.selection_metric},
                        "action_masking":True,"selection_metric":args.selection_metric,
                        "selection_score":candidate_score})
                print("EVALUATION",done,score,flush=True)
            if done%args.save_every==0:
                update_policy()
                save(done)
                print("SAVED",done,"last_episode_pnl",history[-1]["pnl"],flush=True)
            if args.max_minutes and (time.monotonic()-started)/60>=args.max_minutes:
                print("Runtime cap reached; saving.",flush=True)
                break
    except KeyboardInterrupt:
        print("Interrupted; checkpointing.",flush=True)
    finally:
        update_policy()
        save(done)
    if args.evaluate_test and done>=args.episodes:
        score,rows=evaluate(market,test_days,mean,std,policy,
            max_risk_fraction=args.max_risk_fraction)
        (out/"final_test.json").write_text(json.dumps(
            {"episode":done,"test":score,"policy":"latest"},indent=2))
        pd.DataFrame(rows).to_csv(out/"test_days.csv",index=False)
        print("HELD_OUT_TEST",score,flush=True)
    print("DONE",done,"of",args.episodes,flush=True)


if __name__=="__main__":
    main()
