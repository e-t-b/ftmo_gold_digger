import numpy as np
import pandas as pd
import pytest
import torch
from ftmo_gold.deep_learning import GoldGRU, FEATURES
from ftmo_gold.dl_inference import infer


def build_fixture(tmp_path, timeframes=("m5","m15","h1")):
    seconds={"m5":300,"m15":900,"h1":3600,"h4":14400,"d1":86400}
    begin=1800000000
    end=begin+21*86400
    for tf in timeframes:
        step=seconds[tf]
        times=np.arange(begin-begin%step,end,step,dtype=np.int64)
        prices=3000+np.sin(np.arange(len(times))/30)*2
        pd.DataFrame(dict(time=times,open=prices,high=prices+1,
                          low=prices-1,close=prices,tick_volume=100,
                          spread=41)).to_csv(tmp_path/(tf+".csv"),index=False)
    lengths={"m5":32,"m15":16,"h1":6,"h4":12,"d1":10}
    model=GoldGRU(branches=len(timeframes))
    ckpt={"state_dict":model.state_dict(),
          "mean":[torch.zeros(1,1,7) for _ in timeframes],
          "std":[torch.ones(1,1,7) for _ in timeframes],
          "config":{"features":list(FEATURES),"timeframes":list(timeframes),
                    "lengths":lengths}}
    path=tmp_path/"test_model.pt"
    torch.save(ckpt,path)
    return path, end-3600


@pytest.mark.parametrize("timeframes",[("m5","m15","h1"),
                                        ("m5","m15","h1","h4","d1")])
def test_inference_timeframes(tmp_path,timeframes):
    checkpoint,asof=build_fixture(tmp_path,timeframes)
    r=infer(tmp_path,checkpoint,asof)
    assert r["status"]=="RESEARCH_ONLY_NOT_CALIBRATED"
    assert list(r["probabilities"])==["down","neutral","up"]
    assert abs(sum(r["probabilities"].values())-1)<1e-5
    assert all(t<=asof for t in r["last_closed_utc"].values())


def test_no_future_data(tmp_path):
    checkpoint,asof=build_fixture(tmp_path)
    first=infer(tmp_path,checkpoint,asof)
    file=tmp_path/"m5.csv"
    df=pd.read_csv(file)
    df.loc[df.time>asof,"close"]+=1000
    df.loc[df.time>asof,"high"]+=1000
    df.to_csv(file,index=False)
    assert infer(tmp_path,checkpoint,asof)["probabilities"]==first["probabilities"]


def test_stale_rejected(tmp_path):
    checkpoint,asof=build_fixture(tmp_path)
    with pytest.raises(ValueError,match="stale"):
        infer(tmp_path,checkpoint,asof+9*86400)
