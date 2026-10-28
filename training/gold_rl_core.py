"""24h XAUUSD episodes and 5-timeframe PPO network. Research only; no orders."""
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

TFS={"m5":300,"m15":900,"h1":3600,"h4":14400,"d1":86400}
LENS={"m5":16,"m15":12,"h1":8,"h4":8,"d1":8}
ACTIONS=["WAIT","BUY_R025","BUY_R05","BUY_R1","SELL_R025","SELL_R05","SELL_R1","CLOSE"]
FEATURE_NAMES=("log_return","range","body","upper_wick","lower_wick","log_volume",
    "spread_relative","rsi_14","ema_9_21","macd_histogram","atr_14",
    "volume_z_20","bollinger_position_20","fibonacci_position_50")
FEATURE_DIM=len(FEATURE_NAMES)
FEATURE_SETS={"basic":FEATURE_NAMES[:7],"compact":FEATURE_NAMES[:12],
    "technical":FEATURE_NAMES}


def build_features(df):
    """Build causal, dimensionless features from current and past bars only."""
    p=pd.Series(df.close.to_numpy(dtype=float));o=pd.Series(df.open.to_numpy(dtype=float))
    h=pd.Series(df.high.to_numpy(dtype=float));l=pd.Series(df.low.to_numpy(dtype=float))
    prev=p.shift(1).fillna(p.iloc[0]).clip(lower=.01)
    volume=pd.Series(df.tick_volume.to_numpy(dtype=float));log_volume=np.log1p(volume)
    spread=pd.Series(df.spread.to_numpy(dtype=float))*.01

    delta=p.diff().fillna(0.);gain=delta.clip(lower=0.);loss=(-delta).clip(lower=0.)
    avg_gain=gain.ewm(alpha=1/14,adjust=False,min_periods=2).mean()
    avg_loss=loss.ewm(alpha=1/14,adjust=False,min_periods=2).mean()
    rs=avg_gain/avg_loss.replace(0.,np.nan)
    rsi=100.-100./(1.+rs)
    rsi=rsi.mask((avg_gain==0.)&(avg_loss==0.),50.).fillna(100.)

    ema9=p.ewm(span=9,adjust=False,min_periods=1).mean()
    ema21=p.ewm(span=21,adjust=False,min_periods=1).mean()
    ema12=p.ewm(span=12,adjust=False,min_periods=1).mean()
    ema26=p.ewm(span=26,adjust=False,min_periods=1).mean()
    macd=ema12-ema26;signal=macd.ewm(span=9,adjust=False,min_periods=1).mean()
    true_range=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    atr=true_range.ewm(alpha=1/14,adjust=False,min_periods=2).mean()

    volume_mean=log_volume.rolling(20,min_periods=2).mean()
    volume_std=log_volume.rolling(20,min_periods=2).std(ddof=0).clip(lower=1e-6)
    price_mean=p.rolling(20,min_periods=2).mean()
    price_std=p.rolling(20,min_periods=2).std(ddof=0).clip(lower=1e-6)
    swing_high=h.rolling(50,min_periods=2).max()
    swing_low=l.rolling(50,min_periods=2).min()
    swing_range=(swing_high-swing_low).clip(lower=1e-6)

    values=np.stack([np.log(p.clip(lower=.01)/prev),(h-l)/prev,(p-o)/prev,
        (h-np.maximum(o,p))/prev,(np.minimum(o,p)-l)/prev,log_volume,
        spread/prev,(rsi-50.)/50.,(ema9-ema21)/prev,(macd-signal)/prev,
        atr/prev,(log_volume-volume_mean)/volume_std,
        (p-price_mean)/(2.*price_std),2.*(p-swing_low)/swing_range-1.],axis=1)
    return np.nan_to_num(values,nan=0.,posinf=12.,neginf=-12.).astype("float32")


def apply_action_mask(logits,mask):
    """Remove impossible actions without changing the network architecture."""
    mask=torch.as_tensor(mask,dtype=torch.bool,device=logits.device)
    if mask.ndim==1:mask=mask.unsqueeze(0)
    if mask.shape!=logits.shape:raise ValueError("Action-mask shape mismatch")
    if not torch.all(mask.any(dim=1)):raise ValueError("Action mask removed every action")
    return logits.masked_fill(~mask,torch.finfo(logits.dtype).min)

class Market:
    def __init__(self,folder,feature_set="technical"):
        if feature_set not in FEATURE_SETS:raise ValueError("Unknown feature set")
        self.feature_names=FEATURE_SETS[feature_set]
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
            self.f[tf]={"open_ts":ts,"close_ts":ts+seconds,
                "values":build_features(df)[:,:len(self.feature_names)],
                "open":o,"high":h,"low":l,
                "close":p,"spread":df.spread.to_numpy(dtype=float)*.01}
        self.days=defaultdict(list)
        for i,ts in enumerate(self.f["m5"]["open_ts"]):
            self.days[int(ts)//86400].append(i)
        self.days={k:np.array(v) for k,v in self.days.items() if len(v)>=45}
        self.keys=sorted(self.days)

    def observation(self,i,mean,std,position,equity,day_pnl,trade_count,
                    position_r=0.,time_remaining=1.):
        stamp=int(self.f["m5"]["close_ts"][i]);x=[]
        for k in TFS:
            d=self.f[k];ix=np.searchsorted(d["close_ts"],stamp,side="right")-1;n=LENS[k]
            if ix-n+1<1:return None
            if k in ("m5","m15","h1") and np.any(np.diff(d["open_ts"][ix-n+1:ix+1])!=TFS[k]):
                return None
            if stamp-d["close_ts"][ix]>(TFS[k] if k in ("m5","m15","h1") else 4*86400):
                return None
            feature_count=int(np.asarray(mean[k]).shape[0])
            z=d["values"][ix-n+1:ix+1,:feature_count]
            x.append(np.clip((z-mean[k])/std[k],-12,12).astype("float32"))
        state=np.array([float(position),float(np.clip((equity-10000)/1000,-2,2)),
            float(np.clip(day_pnl/100,-2,2)),trade_count/3,
            float(np.clip(position_r,-2,2)),float(np.clip(time_remaining,0,1))],
            dtype="float32")
        return x,state

    def fit_scaler(self,last_train_timestamp):
        mean={};std={}
        for k,d in self.f.items():
            z=d["values"][d["close_ts"]<=last_train_timestamp]
            if len(z)<10:raise ValueError(f"No training data for {k}")
            mean[k]=z.mean(axis=0);std[k]=np.maximum(z.std(axis=0),1e-6)
        return mean,std

class DayEpisode:
    def __init__(self,market,day,mean,std,max_risk_fraction=1.):
        self.m=market;self.mean=mean;self.std=std;self.idx=market.days[day]
        self.step_idx=0;self.equity=10000.;self.start_equity=10000.
        self.pos=0;self.entry=0.;self.sl=0.;self.tp=0.
        self.trades=0;self.closed=0;self.wins=0;self.net=0.
        self.blocked=0;self.terminated=False;self.peak=10000.;self.dd=0.
        self.entry_spread=0.;self.qty=0.;self.risk_used=0.
        self.max_equity_dd=0.;self.peak_mark=10000.
        self.max_risk_fraction=float(max_risk_fraction)
        self.action_counts=Counter();self.invalid_action_counts=Counter()
        self.decision_steps=0;self.valid_action_slots=0

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
            i=int(self.idx[self.step_idx]);d=self.m.f["m5"]
            bid=float(d["close"][i]);spread=float(d["spread"][i])
            marked=self._mark(bid,spread);position_r=0.
            if self.pos:
                exit_price=bid if self.pos==1 else bid+spread
                risk_distance=max(abs(self.entry-self.sl),1e-6)
                position_r=self.pos*(exit_price-self.entry)/risk_distance
            time_remaining=1.-self.step_idx/max(1,len(self.idx)-1)
            obs=self.m.observation(self.idx[self.step_idx],self.mean,self.std,
                self.pos,marked,marked-10000,self.trades,position_r,time_remaining)
            if obs is not None:return obs
            self.step_idx+=1
        return None

    def _entry_quantity(self,action,before,i):
        d=self.m.f["m5"];lo=max(1,i-14);close=d["close"][lo-1:i+1]
        high=d["high"][lo:i+1];low=d["low"][lo:i+1]
        tr=np.maximum(high-low,np.maximum(abs(high-close[:-1]),abs(low-close[:-1])))
        atr=float(np.mean(tr));fraction=(.25,.5,1.)[(action-1)%3]
        risk_budget=min(100.*fraction,max(0.,100.-max(0.,10000.-before)))
        distance=max(1.5,1.5*atr)
        qty=np.floor(min(10.,risk_budget/distance)*100)/100
        return distance,qty

    def action_mask(self):
        """Return the actions that can be executed at the current decision."""
        mask=np.zeros(len(ACTIONS),dtype=bool);mask[0]=True
        i=int(self.idx[self.step_idx]);d=self.m.f["m5"]
        before=self._mark(float(d["close"][i]),float(d["spread"][i]))
        if self.pos:
            mask[7]=True
        elif self.trades<3 and before>9900:
            for action in range(1,7):
                fraction=(.25,.5,1.)[(action-1)%3]
                if fraction>self.max_risk_fraction:continue
                _,qty=self._entry_quantity(action,before,i)
                mask[action]=qty>=1.
        return mask

    def _invalid_reason(self,action):
        if action==7:return "close_without_position"
        if action in range(1,7) and (.25,.5,1.)[(action-1)%3]>self.max_risk_fraction:
            return "risk_above_cap"
        if self.pos:return "position_already_open"
        if self.trades>=3:return "trade_limit_reached"
        i=int(self.idx[self.step_idx]);d=self.m.f["m5"]
        before=self._mark(float(d["close"][i]),float(d["spread"][i]))
        if before<=9900:return "daily_risk_limit"
        return "quantity_below_minimum"

    def step(self,action):
        if self.terminated:raise RuntimeError("Episode ended")
        if not 0<=action<len(ACTIONS):raise ValueError("Unknown action")
        mask=self.action_mask();self.decision_steps+=1
        self.valid_action_slots+=int(mask.sum());self.action_counts[ACTIONS[action]]+=1
        if not mask[action]:
            self.invalid_action_counts[self._invalid_reason(action)]+=1
            if action in range(1,7):self.blocked+=1
            action=0
        i=int(self.idx[self.step_idx]);j=int(self.idx[self.step_idx+1]);d=self.m.f["m5"]
        if int(d["open_ts"][j]-d["open_ts"][i])!=300:
            self.terminated=True
            return -1.,True
        bid=float(d["open"][j]);spread=float(d["spread"][j]);adverse=.05
        before=self._mark(float(d["close"][i]),float(d["spread"][i]))
        if action==7 and self.pos:
            self._settle(bid-adverse if self.pos==1 else bid+spread+adverse)
        if action in (1,2,3,4,5,6) and not self.pos and self.trades<3 and before>9900:
            distance,qty=self._entry_quantity(action,before,i)
            if qty>=1.:
                self.qty=qty;self.pos=1 if action<=3 else -1
                self.entry=bid+spread+adverse if self.pos==1 else bid-adverse
                self.sl=self.entry-self.pos*distance
                self.tp=self.entry+self.pos*(1.5*distance)
                self.trades+=1
            else:self.blocked+=1
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
        trade_limit_finished=self.trades>=3 and not self.pos
        finished=self.step_idx>=len(self.idx)-2 or daily_limit or trade_limit_finished
        if finished:
            if self.pos:self._settle(bid_close-adverse if self.pos==1 else bid_close+spread_close+adverse)
            self.terminated=True
            if daily_limit:reward-=10.
        return reward,finished

class Policy(nn.Module):
    def __init__(self,hidden=16,state_features=6,input_features=FEATURE_DIM):
        super().__init__()
        self.state_features=state_features;self.input_features=input_features
        self.enc=nn.ModuleList([nn.GRU(input_features,hidden,batch_first=True) for _ in TFS])
        self.shared=nn.Sequential(nn.Linear(hidden*len(TFS)+state_features,64),nn.Tanh())
        self.actor=nn.Linear(64,len(ACTIONS));self.critic=nn.Linear(64,1)
    def forward(self,windows,state):
        if state.shape[1]<self.state_features:raise ValueError("Missing policy state features")
        state=state[:,:self.state_features]
        if any(window.shape[2]<self.input_features for window in windows):
            raise ValueError("Missing market features")
        windows=[window[:,:,:self.input_features] for window in windows]
        v=[r(w)[1][-1] for r,w in zip(self.enc,windows)]
        h=self.shared(torch.cat(v+[state],dim=1))
        return self.actor(h),self.critic(h).squeeze(-1)

def tensor_obs(obs):
    windows,state=obs[:2]
    return [torch.tensor(x[None,:,:],dtype=torch.float32) for x in windows],torch.tensor(state[None,:],dtype=torch.float32)

def rollout(market,day,mean,std,policy,stochastic=True,use_action_mask=True,
            max_risk_fraction=1.):
    ep=DayEpisode(market,day,mean,std,max_risk_fraction)
    observations=[];actions=[];oldlog=[];values=[];rewards=[]
    while True:
        obs=ep.current()
        if obs is None:break
        wins,st=tensor_obs(obs)
        with torch.no_grad():
            logits,val=policy(wins,st)
            mask=ep.action_mask()
            if use_action_mask:logits=apply_action_mask(logits,mask)
            dist=Categorical(logits=logits)
            act=dist.sample() if stochastic else logits.argmax(dim=-1)
        reward,done=ep.step(int(act.item()))
        observations.append((obs[0],obs[1],mask.copy()));actions.append(int(act.item()))
        oldlog.append(float(dist.log_prob(act).item()))
        values.append(float(val.item()));rewards.append(reward)
        if done:break
    return ep,observations,np.array(actions),np.array(oldlog),np.array(values),np.array(rewards)
