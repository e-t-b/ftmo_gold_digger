from ftmo_gold.market import audit_bars,audit_ticks,features
from ftmo_gold.strategies import candidates,VERSION

def bars(n=90):
    return [dict(time=i*300,open=100+i*.1,high=100.8+i*.1,
                 low=99.5+i*.1,close=100.2+i*.1) for i in range(n)]

def test_quality_and_ticks():
    b=bars(4)
    assert audit_bars(b)["invalid_ohlc"]==0
    b[-1]["time"]+=300
    assert len(audit_bars(b)["gaps"])==1
    t=[dict(time_msc=1,bid=100,ask=100.41),dict(time_msc=2,bid=100,ask=100.45)]
    assert audit_ticks(t)["spread_median"] in (.41,.45)

def test_causal_features_and_gaps():
    assert features(bars(30)) is None
    b=bars()
    assert features(b)["time"]==b[-1]["time"]
    assert features(b[:-1])["time"]==b[-2]["time"]
    broken=bars()
    broken[-20]["time"]+=100
    assert features(broken) is None
    assert candidates(broken)==[]

def test_strategy_structure():
    results=candidates(bars())
    assert isinstance(results,list)
    assert all(c.side in ("buy","sell") and c.version==VERSION for c in results)
    assert all(c.stop<c.reference_close<c.target if c.side=="buy"
               else c.target<c.reference_close<c.stop for c in results)
