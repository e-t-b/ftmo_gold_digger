"""Causal market features and data-quality checks, using CLOSED bars only."""
from datetime import datetime, timezone
from math import isfinite, sqrt
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

def _ema(values, period):
    alpha=2/(period+1)
    value=float(values[0])
    output=[value]
    for current in values[1:]:
        value=alpha*float(current)+(1-alpha)*value
        output.append(value)
    return output

def technical_features(closed):
    """Compact causal indicators for the language-model observation."""
    if len(closed)<52:
        return None
    rows=closed[-62:]
    closes=[float(row["close"]) for row in rows]
    highs=[float(row["high"]) for row in rows]
    lows=[float(row["low"]) for row in rows]
    changes=[right-left for left,right in zip(closes,closes[1:])]
    gains=[max(change,0.) for change in changes[-14:]]
    losses=[max(-change,0.) for change in changes[-14:]]
    average_gain=sum(gains)/14
    average_loss=sum(losses)/14
    if average_loss==0:
        rsi14=100. if average_gain else 50.
    else:
        rsi14=100-100/(1+average_gain/average_loss)
    ema9=_ema(closes,9)
    ema12=_ema(closes,12)
    ema21=_ema(closes,21)
    ema26=_ema(closes,26)
    macd=[fast-slow for fast,slow in zip(ema12,ema26)]
    signal=_ema(macd,9)
    window=closes[-20:]
    mean20=sum(window)/20
    std20=sqrt(sum((value-mean20)**2 for value in window)/20)
    high50=max(highs[-50:]);low50=min(lows[-50:])
    fib_position=(closes[-1]-low50)/(high50-low50) if high50>low50 else .5
    volume_key=next((key for key in ("tick_volume","real_volume","volume")
                     if any(key in row for row in rows)),None)
    relative_volume=None
    if volume_key:
        volumes=[float(row.get(volume_key,0.)) for row in rows[-20:]]
        baseline=sum(volumes[:-1])/len(volumes[:-1])
        relative_volume=volumes[-1]/baseline if baseline>0 else None
    return {
        "rsi14":round(rsi14,4),
        "ema9":round(ema9[-1],6),
        "ema21":round(ema21[-1],6),
        "ema9_minus_ema21":round(ema9[-1]-ema21[-1],6),
        "macd":round(macd[-1],6),
        "macd_signal":round(signal[-1],6),
        "macd_histogram":round(macd[-1]-signal[-1],6),
        "bollinger_z20":round((closes[-1]-mean20)/std20,5) if std20 else 0.,
        "fibonacci_range50":{
            "low":round(low50,6),"high":round(high50,6),
            "position_0_1":round(fib_position,5),
        },
        "relative_volume20":round(relative_volume,5) if relative_volume is not None else None,
    }

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
    result={"time":ts,"timeframe":timeframe,"atr14":round(atr,6),
            "prior_atr14":round(prior_atr,6),"atr_expansion":round(expansion,5),
            "efficiency14":round(efficiency,5),"signed_move_atr":round(signed_move,5),
            "prior_high14":high,"prior_low14":low,
            "london_hour":instant.astimezone(ZoneInfo("Europe/London")).hour,
            "ny_hour":instant.astimezone(ZoneInfo("America/New_York")).hour,
            "regime":regime}
    result["technical"]=technical_features(x)
    return result
