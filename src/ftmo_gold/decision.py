"""Market decision protocol: discretionary scenarios are allowed in RESEARCH only.

No live order execution. Supports a deterministic WAIT fallback, a mock model,
or an optional external model adapter. Strategy names are contextual aids, not
an allowlist. The validator enforces syntax, bounded prices and provenance.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
from json import dumps
from math import isfinite
from typing import Protocol
from .knowledge import playbooks
from .market import STEP, audit_bars, features

ACTIONS = {"BUY", "SELL", "WAIT", "CLOSE", "HOLD"}


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    confidence: float | None = None
    stop: float | None = None
    target: float | None = None
    thesis: str = ""
    strategy_reference: str | None = None

    def json(self):
        return asdict(self)


class Model(Protocol):
    def decide(self, observation: dict, references: list[dict]) -> dict: ...


def wait(reason: str):
    return Decision(action="WAIT", reason=reason)


def _checked_window(rows, timeframe, count):
    if len(rows)<count:
        raise ValueError(f"{timeframe}: insufficient closed bars")
    chosen=rows[-count:]
    audit=audit_bars(chosen,timeframe)
    if audit["invalid_ohlc"] or audit["unordered_or_duplicate"] or audit["misaligned"] or audit["gaps"]:
        raise ValueError(f"{timeframe}: invalid/non-contiguous bars")
    return chosen


def observe(timeframes: dict[str,list[dict]], spread: float | None,
            *, closed_asof: int | None = None) -> dict:
    """Normalize a snapshot of *closed* bars across timeframes.

    Bars contain UTC OPEN timestamps; their final close must be <= closed_asof.
    This function prevents accidentally giving the model future/partial bars.
    The caller is responsible for providing historical spreads and quote ages.
    """
    if spread is None or not isfinite(spread) or spread<0:
        raise ValueError("Valid current bid-ask spread required")
    if closed_asof is None:
        raise ValueError("Explicit UTC as-of timestamp required")
    state={"asof_utc":int(closed_asof),"spread":round(spread,5),"timeframes":{}}
    for tf in ("M5","M15","H1"):
        rows=_checked_window(timeframes.get(tf,[]),tf,62)
        if rows[-1]["time"]+STEP[tf]>closed_asof:
            raise ValueError(f"{tf}: partial/future candle")
        # Reject snapshots too stale to support a comparable decision.
        if closed_asof-(rows[-1]["time"]+STEP[tf])>STEP[tf]:
            raise ValueError(f"{tf}: stale candle")
        derived=features(rows,timeframe=tf)
        if derived is None:
            raise ValueError(f"{tf}: missing features")
        last=rows[-1]
        state["timeframes"][tf]={
            "last_closed_at_utc":int(last["time"])+STEP[tf],
            "last_close":round(float(last["close"]),5),
            "last_high":round(float(last["high"]),5),
            "last_low":round(float(last["low"]),5),
            "features":derived,
        }
    state["snapshot_id"]=sha256(dumps(state,sort_keys=True).encode()).hexdigest()[:20]
    return state


def validate(raw: dict, observation: dict, *, position_open=False) -> Decision:
    if not isinstance(raw,dict):
        raise ValueError("Model must return JSON object")
    action=raw.get("action")
    if action not in ACTIONS:
        raise ValueError("Unknown action")
    reason=raw.get("reason")
    if not isinstance(reason,str) or not (8<=len(reason)<=1500):
        raise ValueError("Require meaningful bounded reasoning")
    if action in ("CLOSE","HOLD") and not position_open:
        raise ValueError("Position management requires an open position")
    confidence=raw.get("confidence")
    if confidence is not None and (
        isinstance(confidence,bool) or not isinstance(confidence,(int,float))
        or not isfinite(confidence) or not 0<=confidence<=1
    ):
        raise ValueError("Invalid confidence")
    strategy=raw.get("strategy_reference")
    if strategy is not None and (not isinstance(strategy,str) or len(strategy)>120):
        raise ValueError("Invalid strategy reference")
    thesis=raw.get("thesis","")
    if not isinstance(thesis,str) or len(thesis)>1000:
        raise ValueError("Invalid thesis")
    stop,target=raw.get("stop"),raw.get("target")
    if action in ("BUY","SELL"):
        if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and isfinite(x) and x>0
                   for x in (stop,target)):
            raise ValueError("Entry needs finite positive stop and target")
        close=observation["timeframes"]["M5"]["last_close"]
        if not (stop<close<target if action=="BUY" else target<close<stop):
            raise ValueError("Stops/targets inconsistent with action and observed price")
        # Only price proposal: execution adapter MUST recheck executable bid/ask.
        # Bound unreasonable stop distances; this is not a profitability filter.
        if abs(stop-close)>close*.15 or abs(target-close)>close*.30:
            raise ValueError("Price levels out of safety bounds")
    else:
        if stop is not None or target is not None:
            raise ValueError("Non-entry action cannot propose price levels")
    return Decision(action,reason,float(confidence) if confidence is not None else None,
                    float(stop) if stop is not None else None,
                    float(target) if target is not None else None,thesis,strategy)


@dataclass
class FreeDecisionAgent:
    model: Model | None = None

    def decide(self, observation: dict, *, position_open=False,
               trading_context: dict | None = None) -> Decision:
        if self.model is None:
            return wait("No AI model configured; abstain in research mode.")
        try:
            model_observation=dict(observation)
            model_observation["position"]={"open":bool(position_open)}
            if trading_context is not None:
                if not isinstance(trading_context,dict):
                    raise ValueError("Trading context must be an object")
                model_observation["trading_context"]=trading_context
            return validate(self.model.decide(model_observation,playbooks()),
                            observation,position_open=position_open)
        except Exception as exc:
            # Fail closed, never issue orders on parse/model/validation error.
            return wait("Research decision rejected: "+type(exc).__name__)
