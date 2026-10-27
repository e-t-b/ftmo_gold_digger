import numpy as np
import pandas as pd
import pytest
import torch
from ftmo_gold.deep_learning import assemble, chronological_split, GoldGRU, stats

def make_data(root):
    for tf,sec in [('m5',300),('m15',900),('h1',3600)]:
        t=np.arange(14*86400//sec)*sec+1800000000-(1800000000%sec)
        close=3000+np.sin(np.arange(len(t))/30)*10
        pd.DataFrame({'time':t,'open':close-.2,'high':close+.8,'low':close-.8,
                      'close':close,'tick_volume':200,'spread':40}).to_csv(root/(tf+'.csv'),index=False)

def test_build_and_chronology(tmp_path):
    make_data(tmp_path)
    x,y,t,moves=assemble(tmp_path,sequence=16,horizon=6)
    assert [v.shape[1:] for v in x]==[(16,7),(16,7),(6,7)]
    assert len(y)==len(t)==len(moves) and np.all(np.diff(t)>0)
    a,b,c=chronological_split(t,embargo_seconds=6*3600)
    assert max(t[a])<min(t[b])<max(t[b])<min(t[c])

def test_future_changes_cannot_affect_past_windows(tmp_path):
    make_data(tmp_path)
    x,y,t,_=assemble(tmp_path,sequence=16)
    path=tmp_path/'h1.csv';df=pd.read_csv(path)
    df.loc[df.index[-1],'close']+=1000
    df.loc[df.index[-1],'high']+=1000
    df.to_csv(path,index=False)
    xx,_,tt,_=assemble(tmp_path,sequence=16)
    n=int(np.searchsorted(t,t[-30]))
    assert np.array_equal(t[:n],tt[:n])
    for a,b in zip(x,xx): assert np.allclose(a[:n],b[:n])

def test_invalid_price_data_rejected(tmp_path):
    make_data(tmp_path)
    path=tmp_path/'m5.csv';df=pd.read_csv(path)
    df.loc[0,'high']=0;df.to_csv(path,index=False)
    with pytest.raises(ValueError,match='invalid OHLC'): assemble(tmp_path)

def test_gru_output_and_metrics():
    model=GoldGRU()
    output=model(torch.zeros(2,16,7),torch.zeros(2,16,7),torch.zeros(2,6,7))
    assert output.shape==(2,3)
    assert stats(np.array([0,1,2]),np.eye(3))['accuracy']==1
