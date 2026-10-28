"""Simple checks for the committed PPO trainer; no broker connection."""
import importlib.util
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/"training"
sys.path.insert(0,str(ROOT))
from gold_rl_core import (Policy, ACTIONS, FEATURE_DIM, LENS, DayEpisode,
    apply_action_mask, build_features)
from checkpoint_io import safe_torch_load
from train_massive import checkpoint
import torch
import numpy as np
import pandas as pd


def test_eight_actions_and_five_branches():
    policy=Policy()
    windows=[torch.zeros(2,n,FEATURE_DIM) for n in LENS.values()]
    logits,values=policy(windows,torch.zeros(2,6))
    assert logits.shape==(2,8)
    assert values.shape==(2,)
    assert len(ACTIONS)==8


def test_indicator_features_are_finite_and_causal():
    n=120;base=3000.+np.sin(np.arange(n)/8.)*10.+np.arange(n)*.05
    frame=pd.DataFrame({"open":base-.2,"high":base+1.,"low":base-1.,
        "close":base,"tick_volume":200.+np.arange(n)%17,"spread":40.})
    original=build_features(frame)
    changed=frame.copy();changed.loc[n-1,["open","high","low","close"]]+=[50,50,50,50]
    changed.loc[n-1,"tick_volume"]*=10
    modified=build_features(changed)
    assert original.shape==(n,FEATURE_DIM)
    assert np.isfinite(original).all()
    assert np.array_equal(original[:-1],modified[:-1])
    assert not np.array_equal(original[-1],modified[-1])


def test_atomic_checkpoint(tmp_path):
    target=tmp_path/"latest.pt"
    checkpoint(target,{"episode":17})
    assert torch.load(target,weights_only=True)["episode"]==17
    assert not (tmp_path/"latest.tmp").exists()


def fake_market():
    class FakeMarket:
        pass
    market=FakeMarket();n=40
    close=np.full(n,3000.,dtype=float)
    market.f={"m5":{"open_ts":np.arange(n,dtype=np.int64)*300,
        "close_ts":np.arange(n,dtype=np.int64)*300+300,
        "open":close.copy(),"high":close+1.,"low":close-1.,
        "close":close,"spread":np.full(n,.4)}}
    market.days={0:np.arange(n,dtype=int)}
    return market


def test_action_mask_tracks_episode_constraints():
    episode=DayEpisode(fake_market(),0,{},{});episode.step_idx=20
    assert episode.action_mask().tolist()==[True,True,True,True,True,True,True,False]
    episode.pos=1
    assert episode.action_mask().tolist()==[True,False,False,False,False,False,False,True]
    episode.pos=0;episode.trades=3
    assert episode.action_mask().tolist()==[True,False,False,False,False,False,False,False]


def test_action_mask_enforces_configured_risk_cap():
    episode=DayEpisode(fake_market(),0,{}, {},max_risk_fraction=.5);episode.step_idx=20
    assert episode.action_mask().tolist()==[True,True,True,False,True,True,False,False]
    episode.step(3)
    assert episode.pos==0
    assert episode.invalid_action_counts["risk_above_cap"]==1
    episode.trades=0;episode.equity=9900.
    assert episode.action_mask().tolist()==[True,False,False,False,False,False,False,False]


def test_masked_distribution_cannot_select_invalid_action():
    logits=torch.tensor([[0.,100.,90.,80.,70.,60.,50.,40.]])
    masked=apply_action_mask(logits,[True,False,False,False,False,False,False,True])
    samples=torch.distributions.Categorical(logits=masked).sample((200,))
    assert set(samples.flatten().tolist()) <= {0,7}


def test_episode_ends_when_third_trade_is_closed():
    episode=DayEpisode(fake_market(),0,{},{});episode.step_idx=20
    episode.trades=3;episode.pos=1;episode.qty=1.;episode.entry=3000.
    episode.sl=2900.;episode.tp=3100.
    _,done=episode.step(7)
    assert done is True
    assert episode.terminated is True
    assert episode.closed==1


def test_restricted_loader_accepts_numpy_scaler(tmp_path):
    target=tmp_path/"safe.pt"
    checkpoint(target,{"weights":torch.ones(2),"mean":np.ones(3,dtype=np.float32),
        "state":np.arange(3,dtype=np.uint32)})
    loaded=safe_torch_load(target)
    assert torch.equal(loaded["weights"],torch.ones(2))
    assert np.array_equal(loaded["mean"],np.ones(3,dtype=np.float32))
