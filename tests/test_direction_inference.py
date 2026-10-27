import numpy as np
import pandas as pd
import torch
from ftmo_gold.direction_inference import DirectionGRU, infer_direction
from ftmo_gold.deep_learning import FEATURES


def test_direction_binary_checkpoint_inference(tmp_path):
    periods={"m5":300,"m15":900,"h1":3600,"h4":14400,"d1":86400}
    t0=1800000000
    for tf,step in periods.items():
        t=np.arange(20*86400//step,dtype=np.int64)*step+t0-t0%step
        close=3000+np.arange(len(t))*.01
        pd.DataFrame({"time":t,"open":close-.2,"high":close+.8,
                      "low":close-.8,"close":close,"tick_volume":100,
                      "spread":40}).to_csv(tmp_path/(tf+".csv"),index=False)
    model=DirectionGRU(hidden=32,branches=5)
    length={"m5":32,"m15":16,"h1":6,"h4":12,"d1":10}
    checkpoint={"state_dict":model.state_dict(),
                "mean":[torch.zeros(1,1,7) for _ in periods],
                "std":[torch.ones(1,1,7) for _ in periods],
                "config":{"timeframes":list(periods),"lengths":length,
                          "features":list(FEATURES),
                          "objective":"binary_30m_direction","hidden":32}}
    file=tmp_path/"direction.pt"
    torch.save(checkpoint,file)
    asof=t0+18*86400
    result=infer_direction(tmp_path,file,asof)
    assert set(result["probabilities"])=={"down","up"}
    assert abs(sum(result["probabilities"].values())-1)<1e-5
    assert all(value<=asof for value in result["last_closed_utc"].values())
