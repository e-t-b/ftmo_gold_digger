"""Structured, cached Sol plans and deterministic closed-M5 triggers."""
from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite

from .decision import Decision
from .model_openai import MODEL, OpenAISolDecisionModel


TRIGGERS={"CLOSE_ABOVE","CLOSE_BELOW","BULLISH_REJECTION","BEARISH_REJECTION"}

PLAN_SCHEMA={
    "type":"object",
    "properties":{
        "mode":{"type":"string","enum":["CONDITIONAL","WAIT"]},
        "reason":{"type":"string","minLength":8,"maxLength":1500},
        "confidence":{"type":"number","minimum":0,"maximum":1},
        "setups":{"type":"array","minItems":0,"maxItems":2,"items":{
            "type":"object",
            "properties":{
                "label":{"type":"string","minLength":1,"maxLength":80},
                "direction":{"type":"string","enum":["BUY","SELL"]},
                "trigger":{"type":"string","enum":sorted(TRIGGERS)},
                "trigger_level":{"type":"number","exclusiveMinimum":0},
                "stop":{"type":"number","exclusiveMinimum":0},
                "reward_risk":{"type":"number","minimum":1,"maximum":4},
                "max_spread":{"type":"number","minimum":0,"exclusiveMaximum":5},
            },
            "required":["label","direction","trigger","trigger_level","stop",
                        "reward_risk","max_spread"],
            "additionalProperties":False,
        }},
    },
    "required":["mode","reason","confidence","setups"],
    "additionalProperties":False,
}

PLAN_SYSTEM="""You are GPT-5.6 Sol, producing a one-hour conditional XAUUSD plan.
Use only the supplied CLOSED M5/M15/H1 observations and account context. You do
not see future bars. Return WAIT with no setups when evidence is weak. Otherwise
return one or two precise setups that a deterministic engine can evaluate after
each future M5 close. BUY stops must be below trigger levels; SELL stops above.
Reward/risk must be 1R through 4R. Use CLOSE_ABOVE/CLOSE_BELOW for a genuine
cross from the prior close, and BULLISH_REJECTION/BEARISH_REJECTION for a level
touched intrabar and rejected by a directional candle. Keep max_spread at or
below three times the supplied spread (and always below 5). Never invent news
or fills."""


def validate_plan(raw,observation,valid_seconds=3600):
    if not isinstance(raw,dict) or raw.get("mode") not in {"CONDITIONAL","WAIT"}:
        raise ValueError("Invalid plan mode")
    reason=raw.get("reason")
    confidence=raw.get("confidence")
    setups=raw.get("setups")
    if not isinstance(reason,str) or not 8<=len(reason)<=1500:
        raise ValueError("Invalid plan reason")
    if (isinstance(confidence,bool) or not isinstance(confidence,(int,float))
            or not isfinite(confidence) or not 0<=confidence<=1):
        raise ValueError("Invalid plan confidence")
    if not isinstance(setups,list) or len(setups)>2:
        raise ValueError("Invalid plan setups")
    if raw["mode"]=="WAIT" and setups:
        raise ValueError("WAIT plan cannot contain setups")
    if raw["mode"]=="CONDITIONAL" and not setups:
        raise ValueError("Conditional plan requires a setup")
    close=float(observation["timeframes"]["M5"]["last_close"])
    spread_cap=min(5.,max(.5,float(observation["spread"])*3))
    normalized=[]
    for index,setup in enumerate(setups):
        if not isinstance(setup,dict) or set(setup)!={
            "label","direction","trigger","trigger_level","stop","reward_risk","max_spread"}:
            raise ValueError("Invalid setup fields")
        direction=setup["direction"];trigger=setup["trigger"]
        label=setup["label"]
        values=[setup[key] for key in ("trigger_level","stop","reward_risk","max_spread")]
        if (direction not in {"BUY","SELL"} or trigger not in TRIGGERS
                or not isinstance(label,str) or not 1<=len(label)<=80
                or any(isinstance(value,bool) or not isinstance(value,(int,float))
                       or not isfinite(value) for value in values)):
            raise ValueError("Invalid setup")
        level,stop,rr,max_spread=map(float,values)
        if not (level>0 and stop>0 and 1<=rr<=4 and 0<=max_spread<=spread_cap):
            raise ValueError("Invalid setup bounds")
        if abs(level-close)>close*.15 or abs(stop-close)>close*.15:
            raise ValueError("Plan levels outside safety bounds")
        if not (stop<level if direction=="BUY" else stop>level):
            raise ValueError("Stop inconsistent with setup direction")
        normalized.append({"setup_id":index,"label":label,"direction":direction,
            "trigger":trigger,"trigger_level":round(level,6),"stop":round(stop,6),
            "reward_risk":round(rr,4),"max_spread":round(max_spread,6)})
    created=int(observation["asof_utc"])
    body={"model":MODEL,"snapshot_id":observation["snapshot_id"],
          "created_at":created,"valid_until":created+int(valid_seconds),
          "mode":raw["mode"],"reason":reason,"confidence":float(confidence),
          "setups":normalized}
    body["plan_id"]=sha256(json.dumps(body,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:20]
    return body


def setup_triggered(setup,previous,current):
    level=float(setup["trigger_level"])
    prior_close=float(previous["close"]);close=float(current["close"])
    high=float(current["high"]);low=float(current["low"]);opened=float(current["open"])
    trigger=setup["trigger"]
    if trigger=="CLOSE_ABOVE":
        return prior_close<=level<close
    if trigger=="CLOSE_BELOW":
        return prior_close>=level>close
    if trigger=="BULLISH_REJECTION":
        return low<=level<close and close>opened
    return high>=level>close and close<opened


def trigger_plan(plan,previous,current,spread):
    """Return a deterministic proposal after a future M5 close, or None."""
    closed_at=int(current["time"])+300
    if (plan.get("mode")!="CONDITIONAL" or closed_at<=int(plan["created_at"])
            or closed_at>int(plan["valid_until"])):
        return None
    for setup in plan["setups"]:
        if spread>float(setup["max_spread"]) or not setup_triggered(setup,previous,current):
            continue
        direction=setup["direction"]
        reference=float(current["close"])+(spread if direction=="BUY" else 0.)
        stop=float(setup["stop"])
        if not (stop<reference if direction=="BUY" else stop>reference):
            continue
        risk=abs(reference-stop);rr=float(setup["reward_risk"])
        target=reference+(risk*rr if direction=="BUY" else -risk*rr)
        return Decision(direction,f"M5 condition triggered from plan {plan['plan_id']}.",
                        float(plan["confidence"]),stop,target,plan["reason"],setup["label"])
    return None


class SolPlanModel(OpenAISolDecisionModel):
    def create_plan(self,observation,account_context):
        self._ensure_model()
        payload={
            "model":MODEL,"instructions":PLAN_SYSTEM,
            "input":[{"role":"user","content":[{"type":"input_text","text":json.dumps({
                "observation":observation,"account_context":account_context,
            },sort_keys=True,separators=(",",":"))}]}],
            "reasoning":{"effort":self.reasoning_effort},
            "text":{"verbosity":"low","format":{"type":"json_schema",
                "name":"gold_digger_hour_plan","strict":True,"schema":PLAN_SCHEMA}},
            "max_output_tokens":1800,"store":False,"stream":True,
        }
        return validate_plan(json.loads(self._stream_response(payload)),observation)
