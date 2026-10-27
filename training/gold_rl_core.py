"""24h XAUUSD episodes and 5-timeframe PPO network. Research only; no orders."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

TFS={"m5":300,"m15":900,"h1":3600,"h4":14400,"d1":86400}
LENS={"m5":16,"m15":12,"h1":8,"h4":8,"d1":8}
ACTIONS=["WAIT","BUY_R025","BUY_R05","BUY_R1","SELL_R025","SELL_R05","SELL_R1","CLOSE"]

class Market:
    def __init__(self,folder):
        self.f={}
        for tf,seconds in TFS.items():
            df=pd.read_csv(Path(folder)/(tf+".csv"))
            for k in ("time","open","high","low","close","tick_volume","spread"):
                if k not in df: raise ValueError(f"{tf}: missing {k}")
            ts=df.time.to_numpy(dtype=np.int64)
            if not np.all(np.diff(ts)>0): raise ValueError(f"{tf}: timestamps unordered")
            p=df.close.to_numpy(dtype=float);prev=np.maximum(np.r_[p[0],p[:-1]],.01)
            o=df.open.to_numpy(dtype=float);h=df.high.to_numpy(dtype=float);l=df.low.to_numpy(dtype=float)
            if not (np.all(l<=np.minimum(o,p)) and np.all(h>=np.maximum(o,p))):
                raise ValueError(f"{tf}: invalid OHLC")
            f=np.stack([np.log(np.maximum(p,.01)/prev),(h-l)/prev,(p-o)/prev,
                (h-np.maximum(o,p))/prev,(np.minimum(o,p)-l)/prev,
                np.log1p(df.tick_volume.to_numpy(dtype=float)),
                df.spread.to_numpy(dtype=float)*.01/prev],axis=1)
            self.f[tf]={"open_ts":ts,"close_ts":ts+seconds,
                "values":f.astype("float32"),"open":o,"high":h,"low":l,
                "close":p,"spread":df.spread.to_numpy(dtype=float)*.01}
        self.days=defaultdict(list)
        for i,ts in enumerate(self.f["m5"]["open_ts"]):
            self.days[int(ts)//86400].append(i)
        self.days={k:np.array(v) for k,v in self.days.items() if len(v)>=45}
        self.keys=sorted(self.days)

    def observation(self,i,mean,std,position,equity,day_pnl,trade_count):
        stamp=int(self.f["m5"]["close_ts"][i]);x=[]
        for k in TFS:
            d=self.f[k];ix=np.searchsorted(d["close_ts"],stamp,side="right")-1;n=LENS[k]
            if ix-n+1<1:return None
            if k in ("m5","m15","h1") and np.any(np.diff(d["open_ts"][ix-n+1:ix+1])!=TFS[k]):
                return None
            if stamp-d["close_ts"][ix]>(TFS[k] if k in ("m5","m15","h1") else 4*86400):
                return None
            z=d["values"][ix-n+1:ix+1]
            x.append(np.clip((z-mean[k])/std[k],-12,12).astype("float32"))
        state=np.array([float(position),float(np.clip((equity-10000)/1000,-2,2)),
            float(np.clip(day_pnl/100,-2,2)),trade_count/3],dtype="float32")
        return x,state

    def fit_scaler(self,last_train_timestamp):
        mean={};std={}
        for k,d in self.f.items():
            z=d["values"][d["close_ts"]<=last_train_timestamp]
            if len(z)<10:raise ValueError(f"No training data for {k}")
            mean[k]=z.mean(axis=0);std[k]=np.maximum(z.std(axis=0),1e-6)
        return mean,std

class DayEpisode:
    def __init__(self,market,day,mean,std):
        self.m=market;self.mean=mean;self.std=std;self.idx=market.days[day]
        self.step_idx=0;self.equity=10000.;self.start_equity=10000.
        self.pos=0;self.entry=0.;self.sl=0.;self.tp=0.
        self.trades=0;self.closed=0;self.wins=0;self.net=0.
        self.blocked=0;self.terminated=False;self.peak=10000.;self.dd=0.
        self.entry_spread=0.;self.qty=0.;self.risk_used=0.
        self.max_equity_dd=0.;self.peak_mark=10000.

    def _settle(self,price):
        if not self.pos:return
        pnl=self.pos*(price-self.entry)*self.qty
        self.equity+=pnl;self.net+=pnl;self.wins+=int(pnl>0);self.closed+=1;self.pos=0

    def _mark(self,bid,spread):
        if not self.pos:return self.equity
        exit_price=bid if self.pos==1 else bid+spread
        return self.equity+self.pos*(exit_price-self.entry)*self.qty

    def current(self):
        while self.step_idx<len(self.idx)-1:
            obs=self.m.observation(self.idx[self.step_idx],self.mean,self.std,
                self.pos,self.equity,self.equity-10000,self.trades)
            if obs is not None:return obs
            self.step_idx+=1
        return None

    def step(self,action):
        if self.terminated:raise RuntimeError("Episode ended")
        i=int(self.idx[self.step_idx]);j=int(self.idx[self.step_idx+1]);d=self.m.f["m5"]
        if int(d["open_ts"][j]-d["open_ts"][i])!=300:
            self.terminated=True
            return -1.,True
        bid=float(d["open"][j]);spread=float(d["spread"][j]);adverse=.05
        before=self._mark(float(d["close"][i]),float(d["spread"][i]))
        if action==7 and self.pos:
            self._settle(bid-adverse if self.pos==1 else bid+spread+adverse)
        if action in (1,2,3,4,5,6) and not self.pos and self.trades<3 and before>9900:
            lo=max(1,i-14);close=d["close"][lo-1:i+1]
            high=d["high"][lo:i+1];low=d["low"][lo:i+1]
            tr=np.maximum(high-low,np.maximum(abs(high-close[:-1]),abs(low-close[:-1])))
            atr=float(np.mean(tr));fraction=(.25,.5,1.)[(action-1)%3]
            risk_budget=min(100.*fraction,max(0.,100.-max(0.,10000.-before)))
            distance=max(1.5,1.5*atr)
            qty=np.floor(min(10.,risk_budget/distance)*100)/100
            if qty>=1.:
                self.qty=qty;self.pos=1 if action<=3 else -1
                self.entry=bid+spread+adverse if self.pos==1 else bid-adverse
                self.sl=self.entry-self.pos*distance
                self.tp=self.entry+self.pos*(1.5*distance)
                self.trades+=1
            else:self.blocked+=1
        elif action in (1,2,3,4,5,6):self.blocked+=1
        if self.pos:
            hi=float(d["high"][j]);low=float(d["low"][j])
            if self.pos==1:
                if low<=self.sl:self._settle(self.sl-adverse)
                elif hi>=self.tp:self._settle(self.tp-adverse)
            else:
                if hi+spread>=self.sl:self._settle(self.sl+adverse)
                elif low+spread<=self.tp:self._settle(self.tp+adverse)
        self.step_idx+=1
        bid_close=float(d["close"][j]);spread_close=float(d["spread"][j])
        after=self._mark(bid_close,spread_close)
        self.peak=max(self.peak,after);self.dd=max(self.dd,self.peak-after)
        reward=float(np.clip((after-before)/10,-5,5))
        daily_limit=after<=9900
        finished=self.step_idx>=len(self.idx)-2 or daily_limit
        if finished:
            if self.pos:self._settle(bid_close-adverse if self.pos==1 else bid_close+spread_close+adverse)
            self.terminated=True
            if daily_limit:reward-=10.
        return reward,finished

class Policy(nn.Module):
    def __init__(self,hidden=16):
        super().__init__()
        self.enc=nn.ModuleList([nn.GRU(7,hidden,batch_first=True) for _ in TFS])
        self.shared=nn.Sequential(nn.Linear(hidden*len(TFS)+4,64),nn.Tanh())
        self.actor=nn.Linear(64,len(ACTIONS));self.critic=nn.Linear(64,1)
    def forward(self,windows,state):
        v=[r(w)[1][-1] for r,w in zip(self.enc,windows)]
        h=self.shared(torch.cat(v+[state],dim=1))
        return self.actor(h),self.critic(h).squeeze(-1)

def tensor_obs(obs):
    windows,state=obs
    return [torch.tensor(x[None,:,:],dtype=torch.float32) for x in windows],torch.tensor(state[None,:],dtype=torch.float32)

def rollout(market,day,mean,std,policy,stochastic=True):
    ep=DayEpisode(market,day,mean,std)
    observations=[];actions=[];oldlog=[];values=[];rewards=[]
    while True:
        obs=ep.current()
        if obs is None:break
        wins,st=tensor_obs(obs)
        with torch.no_grad():
            logits,val=policy(wins,st);dist=Categorical(logits=logits)
            act=dist.sample() if stochastic else logits.argmax(dim=-1)
        reward,done=ep.step(int(act.item()))
        observations.append(obs);actions.append(int(act.item()))
        oldlog.append(float(dist.log_prob(act).item()))
        values.append(float(val.item()));rewards.append(reward)
        if done:break
    return ep,observations,np.array(actions),np.array(oldlog),np.array(values),np.array(rewards)
