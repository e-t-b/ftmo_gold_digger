"""Four frozen, causal, research-only Gold strategy hypotheses."""
from dataclasses import dataclass
from .market import features

VERSION="candidate-v1-2026-10-27"

@dataclass(frozen=True)
class Candidate:
    name:str
    side:str
    signal_time:int
    reference_close:float
    stop:float
    target:float
    regime:str
    version:str=VERSION

def _make(name,side,bar,stop,rr,regime):
    price=float(bar["close"])
    risk=abs(price-stop)
    if risk<=0 or (side=="buy" and stop>=price) or (side=="sell" and stop<=price): return None
    target=price+(rr*risk if side=="buy" else -rr*risk)
    return Candidate(name,side,int(bar["time"]),price,stop,target,regime)

def candidates(closed):
    """Caller passes only closed consecutive M5 bars; no order execution."""
    f=features(closed,"M5")
    if not f: return []
    previous=features(closed[:-1],"M5")
    if not previous: return []
    b=closed[-1]
    price,op=float(b["close"]),float(b["open"])
    atr=f["prior_atr14"]
    if atr<=0: return []
    result=[]
    if previous["regime"]=="trending_up" and op<price and float(b["low"])<=float(closed[-2]["low"]):
        result.append(_make("trend_pullback","buy",b,float(b["low"])-.15*atr,2,previous["regime"]))
    if previous["regime"]=="trending_down" and op>price and float(b["high"])>=float(closed[-2]["high"]):
        result.append(_make("trend_pullback","sell",b,float(b["high"])+.15*atr,2,previous["regime"]))
    hi,lo=previous["prior_high14"],previous["prior_low14"]
    if float(b["high"])>hi and lo<price<hi and op<=hi:
        result.append(_make("liquidity_sweep","sell",b,float(b["high"])+.1*atr,1.5,previous["regime"]))
    if float(b["low"])<lo and lo<price<hi and op>=lo:
        result.append(_make("liquidity_sweep","buy",b,float(b["low"])-.1*atr,1.5,previous["regime"]))
    amplitude=float(b["high"])-float(b["low"])
    if amplitude>=1.4*atr and previous["regime"]!="volatile":
        if price>hi and op<=hi:
            result.append(_make("expansion_breakout","buy",b,float(b["low"])-.1*atr,2,previous["regime"]))
        if price<lo and op>=lo:
            result.append(_make("expansion_breakout","sell",b,float(b["high"])+.1*atr,2,previous["regime"]))
    if previous["regime"]=="ranging":
        mid=(hi+lo)/2
        if float(b["low"])<=lo+.15*(hi-lo) and price>op and price<mid:
            result.append(_make("range_reversion","buy",b,float(b["low"])-.1*atr,1,previous["regime"]))
        if float(b["high"])>=hi-.15*(hi-lo) and price<op and price>mid:
            result.append(_make("range_reversion","sell",b,float(b["high"])+.1*atr,1,previous["regime"]))
    return [c for c in result if c is not None]
