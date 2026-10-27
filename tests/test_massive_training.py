"""Simple checks for the committed PPO trainer; no broker connection."""
import importlib.util
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/"training"
sys.path.insert(0,str(ROOT))
from gold_rl_core import Policy, ACTIONS, LENS
from train_massive import checkpoint
import torch


def test_eight_actions_and_five_branches():
    policy=Policy()
    windows=[torch.zeros(2,n,7) for n in LENS.values()]
    logits,values=policy(windows,torch.zeros(2,4))
    assert logits.shape==(2,8)
    assert values.shape==(2,)
    assert len(ACTIONS)==8


def test_atomic_checkpoint(tmp_path):
    target=tmp_path/"latest.pt"
    checkpoint(target,{"episode":17})
    assert torch.load(target,weights_only=True)["episode"]==17
    assert not (tmp_path/"latest.tmp").exists()
