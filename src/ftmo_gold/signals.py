"""Causal candidate strategy. Research hypothesis, not a proven edge."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    side: str
    reference_price: float
    stop: float
    target: float
    timestamp: int
    strategy: str


def breakout(history: list[dict], lookback: int = 20, rr: float = 2.0):
    """M5 close breakout of preceding completed range; no future candles used.

    Entry happens at the *next* bar's executable bid/ask, not at reference_price.
    """
    if lookback < 2 or rr <= 0:
        raise ValueError("Invalid strategy parameters")
    if len(history) < lookback + 1:
        return None
    window = history[-lookback - 1:]
    if any(int(a["time"]) >= int(b["time"]) for a,b in zip(window, window[1:])):
        raise ValueError("Bars not strictly chronological")
    past, last = window[:-1], window[-1]
    high = max(float(x["high"]) for x in past)
    low = min(float(x["low"]) for x in past)
    price = float(last["close"])
    if price > high and float(last["open"]) <= high:
        side, stop, target = "buy", low, price + rr * (price - low)
    elif price < low and float(last["open"]) >= low:
        side, stop, target = "sell", high, price - rr * (high - price)
    else:
        return None
    return Signal(side, price, stop, target, int(last["time"]), "m5_breakout_20")
