import math
from datetime import datetime, timezone
from ftmo_gold.decision import observe, validate, FreeDecisionAgent
from ftmo_gold.knowledge import playbooks
from ftmo_gold.market import STEP, scheduled_market_closure

ASOF=86400*100

def source():
    data={}
    for tf,step in [("M5",300),("M15",900),("H1",3600)]:
        data[tf]=[dict(time=ASOF-(62-i)*step,
                       open=3000.0,high=3010.0,low=2990.0,close=3001.0)
                  for i in range(62)]
    return data

def observation():
    return observe(source(),.41,closed_asof=ASOF)

def test_observation_causal_multiframe():
    obs=observation()
    assert set(obs["timeframes"])=={"M5","M15","H1"}
    assert all(obs["timeframes"][tf]["features"]["technical"] is not None
               for tf in ("M5","M15","H1"))
    assert len(obs["snapshot_id"])==20
    bad=source()
    bad["M5"][-1]["time"]=ASOF
    try: observe(bad,.41,closed_asof=ASOF)
    except ValueError: pass
    else: assert False,"Future candle accepted"

def test_stale_and_gap_rejected():
    try: observe(source(),.41,closed_asof=ASOF+3700)
    except ValueError: pass
    else: assert False,"Stale snapshot accepted"
    bad=source()
    bad["M15"][-12]["time"]+=30
    try: observe(bad,.41,closed_asof=ASOF)
    except ValueError: pass
    else: assert False,"Gap accepted"

def test_scheduled_one_hour_market_pause_is_accepted():
    data=source()
    rows=data["H1"]
    boundary=next(i for i in range(1,len(rows)-1)
                  if (datetime.fromtimestamp(rows[i]["time"],timezone.utc).hour,
                      datetime.fromtimestamp(rows[i]["time"],timezone.utc).minute)==(1,0)
                  and datetime.fromtimestamp(rows[i]["time"],timezone.utc).weekday() in (1,2,3,4))
    for row in rows[:boundary]:
        row["time"]-=3600
    obs=observe(data,.41,closed_asof=ASOF)
    assert obs["timeframes"]["H1"]["features"]["technical"] is not None

def test_scheduled_weekend_close_accepts_friday_bar_closing_saturday_utc():
    friday_23_utc=1764975600
    monday_01_utc=1765155600
    assert scheduled_market_closure(friday_23_utc,monday_01_utc,STEP["H1"])

def test_aligned_but_unscheduled_gap_is_rejected():
    assert not scheduled_market_closure(1765281600,1765288800,STEP["H1"])

def test_strategy_is_optional_not_an_allowlist():
    obs=observation()
    proposal=dict(action="BUY",reason="Independent novel microstructure hypothesis",
                  confidence=.4,stop=2980.,target=3030.,thesis="custom",strategy_reference="new_idea")
    d=validate(proposal,obs)
    assert d.action=="BUY" and d.strategy_reference=="new_idea"
    assert all(not p["mandatory"] for p in playbooks())

def test_bad_decisions_fail_closed():
    obs=observation()
    for raw in [
        dict(action="BUY",reason="not safe",stop=3020,target=3100),
        dict(action="BUY",reason="direction is invalid",stop=3020,target=3100),
        dict(action="BUY",reason="invalid NaN prices",stop=float("nan"),target=3100),
        dict(action="SELL",reason="invalid target location",stop=3020,target=3100),
        dict(action="WAIT",reason="reason only",stop=3000),
        dict(action="CLOSE",reason="no position open"),
    ]:
        try: validate(raw,obs)
        except ValueError: pass
        else: assert False,repr(raw)
    class Invalid:
        def decide(self,observation,references):
            return dict(action="BUY",reason="bad stop provided",stop=3030,target=3050)
    assert FreeDecisionAgent(Invalid()).decide(obs).action=="WAIT"

def test_agent_without_model_abstains_and_model_can_wait():
    obs=observation()
    assert FreeDecisionAgent().decide(obs).action=="WAIT"
    class Wait:
        def decide(self,observation,references):
            assert len(references)==4
            return {"action":"WAIT","reason":"Not enough signal quality for a trade."}
    assert FreeDecisionAgent(Wait()).decide(obs).action=="WAIT"
