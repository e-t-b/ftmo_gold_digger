"""Causal market features and data-quality checks, using CLOSED bars only."""
from datetime import datetime, timezone
from math import isfinite
from zoneinfo import ZoneInfo

STEP={"M1":60,"M5":300,"M15":900,"H1":3600}

def audit_bars(rows,timeframe="M5"):
    step=STEP[timeframe]
    result={"count":len(rows),"invalid_ohlc":0,"unordered_or_duplicate":0,"misaligned":0,"gaps":[]}
    previous=None
    for r in rows:
        t=int(r["time"])
        o,h,l,c=(float(r[k]) for k in ("open","high","low","close"))
        if not all(isfinite(x) for x in (o,h,l,c)) or not l<=min(o,c)<=max(o,c)<=h:
            result["invalid_ohlc"]+=1
        if t%step: result["misaligned"]+=1
        if previous is not None:
            delta=t-previous
            if delta<=0: result["unordered_or_duplicate"]+=1
            elif delta>step:
                result["gaps"].append({"from":previous,"to":t,"missing":delta//step-1,"duration_hours":round(delta/3600,3)})
        previous=t
    return result

def audit_ticks(rows):
    count=invalid=crossed=unordered=0
    previous=None
    spreads=[]
    for r in rows:
        count+=1
        t=int(r["time_msc"]) if r.get("time_msc") else int(r["time"])*1000
        if previous is not None and t<previous: unordered+=1
        previous=t
        bid,ask=float(r["bid"]),float(r["ask"])
        if not all(map(isfinite,(bid,ask))) or bid<=0 or ask<=0: invalid+=1
        elif ask<bid: crossed+=1
        else: spreads.append(ask-bid)
    spreads.sort()
    def percentile(p):
        return round(spreads[round((len(spreads)-1)*p)],4) if spreads else None
    return {"count":count,"invalid":invalid,"crossed":crossed,"unordered":unordered,
            "spread_median":percentile(.5),"spread_p90":percentile(.9),"spread_p99":percentile(.99)}

def true_range(bar,previous_close):
    h,l=float(bar["high"]),float(bar["low"])
    return max(h-l,abs(h-previous_close),abs(l-previous_close))

def features(closed,timeframe="M5",period=14,context=48):
    """All observations must be closed and exactly consecutive on this timeframe."""
    step=STEP[timeframe]
    if len(closed)<context+2 or context<=period+1 or period<2: return None
    x=closed[-(context+2):]
    if any(int(b["time"])-int(a["time"])!=step for a,b in zip(x,x[1:])): return None
    if audit_bars(x,timeframe)["invalid_ohlc"]: return None
    trs=[true_range(b,float(a["close"])) for a,b in zip(x,x[1:])]
    atr=sum(trs[-period:])/period
    prior_atr=sum(trs[-period-1:-1])/period
    baseline=sum(trs[-context:-period])/(context-period)
    current,prior=x[-1],x[-period-1]
    close,old_close=float(current["close"]),float(prior["close"])
    movement=close-old_close
    path=sum(abs(float(b["close"])-float(a["close"])) for a,b in zip(x[-period-1:-1],x[-period:]))
    efficiency=abs(movement)/path if path else 0.
    signed_move=movement/(prior_atr*period) if prior_atr else 0.
    expansion=atr/baseline if baseline else 1.
    past=x[-period-1:-1]
    high=max(float(b["high"]) for b in past)
    low=min(float(b["low"]) for b in past)
    ts=int(current["time"])
    instant=datetime.fromtimestamp(ts,timezone.utc)
    if expansion>=1.6: regime="volatile"
    elif efficiency>=.55 and abs(signed_move)>=.45:
        regime="trending_up" if signed_move>0 else "trending_down"
    elif efficiency<=.25 and expansion<=1.2: regime="ranging"
    else: regime="uncertain"
    return {"time":ts,"timeframe":timeframe,"atr14":round(atr,6),
            "prior_atr14":round(prior_atr,6),"atr_expansion":round(expansion,5),
            "efficiency14":round(efficiency,5),"signed_move_atr":round(signed_move,5),
            "prior_high14":high,"prior_low14":low,
            "london_hour":instant.astimezone(ZoneInfo("Europe/London")).hour,
            "ny_hour":instant.astimezone(ZoneInfo("America/New_York")).hour,
            "regime":regime}
