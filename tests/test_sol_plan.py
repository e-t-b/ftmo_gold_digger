import json

from ftmo_gold.decision import observe
from ftmo_gold.sol_plan import (PLAN_SCHEMA,SolPlanModel,immediate_plan_decision,
                                setup_triggered,trigger_plan,validate_plan)


ASOF=86400*100


def observation():
    frames={}
    for name,step in (("M5",300),("M15",900),("H1",3600)):
        frames[name]=[{"time":ASOF-(62-index)*step,"open":3000.,"high":3002.,
                       "low":2998.,"close":3000.,"tick_volume":100}
                      for index in range(62)]
    return observe(frames,.2,closed_asof=ASOF)


def raw_plan(trigger="CLOSE_ABOVE",direction="BUY",level=3001.,stop=2995.,rr=2.):
    return {"mode":"CONDITIONAL","reason":"Wait for a confirmed causal M5 condition.",
            "confidence":.7,"setups":[{"label":"breakout","direction":direction,
            "trigger":trigger,"trigger_level":level,"stop":stop,
            "reward_risk":rr,"max_spread":.5}]}


def test_plan_is_bounded_hashed_and_expires():
    plan=validate_plan(raw_plan(),observation())
    assert len(plan["plan_id"])==20
    assert plan["valid_until"]-plan["created_at"]==3600
    assert plan["snapshot_id"]==observation()["snapshot_id"]
    waiting=validate_plan({"mode":"WAIT","reason":"No reliable setup is present yet.",
                           "confidence":.2,"setups":[]},observation())
    assert waiting["valid_until"]-waiting["created_at"]==900


def test_cross_requires_prior_bar_and_future_closed_bar():
    plan=validate_plan(raw_plan(),observation())
    previous={"time":ASOF,"open":3000.,"high":3001.,"low":2999.,"close":3000.5}
    current={"time":ASOF+300,"open":3000.5,"high":3003.,"low":3000.,"close":3002.}
    assert setup_triggered(plan["setups"][0],previous,current)
    decision=trigger_plan(plan,previous,current,.2)
    assert decision.action=="BUY" and decision.stop==2995.
    assert round((decision.target-(3002.2))/(3002.2-decision.stop),5)==2
    expired=dict(plan,valid_until=ASOF+300)
    assert trigger_plan(expired,previous,current,.2) is None


def test_rejection_triggers_are_directional():
    bullish=raw_plan("BULLISH_REJECTION","BUY",3000.,2995.)["setups"][0]
    bearish=raw_plan("BEARISH_REJECTION","SELL",3000.,3005.)["setups"][0]
    previous={"close":3001.}
    up={"open":2999.,"high":3003.,"low":2998.,"close":3002.}
    down={"open":3001.,"high":3003.,"low":2997.,"close":2999.}
    assert setup_triggered(bullish,previous,up)
    assert setup_triggered(bearish,previous,down)


def test_enter_now_uses_closed_snapshot_and_executable_spread():
    plan=validate_plan(raw_plan("ENTER_NOW"),observation())
    current={"open":3000.,"high":3002.,"low":2998.,"close":3000.}
    decision=immediate_plan_decision(plan,current,.2)
    assert decision.action=="BUY" and decision.target>3000.2
    assert immediate_plan_decision(plan,current,.6) is None


def test_invalid_plan_fails_closed():
    for raw in [
        {"mode":"WAIT","reason":"No setup is currently reliable.","confidence":.2,
         "setups":raw_plan()["setups"]},
        raw_plan(direction="BUY",level=3001.,stop=3010.),
        raw_plan(rr=5.),
        {**raw_plan(),"setups":[{**raw_plan()["setups"][0],"max_spread":1.}]},
    ]:
        try: validate_plan(raw,observation())
        except ValueError: pass
        else: raise AssertionError("Invalid plan accepted")


def test_sol_plan_uses_strict_structured_stream(monkeypatch):
    captured={};raw=raw_plan()
    class Response:
        def __enter__(self): return self
        def __exit__(self,*_): return False
        def __iter__(self):
            text=json.dumps(raw)
            yield ("data: "+json.dumps({"type":"response.output_text.delta","delta":text})+"\n").encode()
            yield ("data: "+json.dumps({"type":"response.completed","response":{"output":[]}})+"\n").encode()
    def fake_urlopen(request,timeout):
        captured["payload"]=json.loads(request.data);return Response()
    monkeypatch.setattr("urllib.request.urlopen",fake_urlopen)
    model=SolPlanModel(token_provider=lambda:"plan-token",verify_model=False)
    plan=model.create_plan(observation(),{"risk_pct":2})
    assert plan["mode"]=="CONDITIONAL"
    assert captured["payload"]["text"]["format"]["schema"]==PLAN_SCHEMA
    assert isinstance(captured["payload"]["input"],list)
