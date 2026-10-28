import hashlib
import json

from ftmo_gold.chatgpt_auth import ChatGPTPlanCredentials, make_pkce
from ftmo_gold.decision import FreeDecisionAgent, observe
from ftmo_gold.market import technical_features
from ftmo_gold.model_openai import DECISION_SCHEMA, MODEL, OpenAISolDecisionModel
from ftmo_gold.sol_paper import LocalCallBudget, decide_once


ASOF=86400*100


def bars(step, count=62):
    result=[]
    for index in range(count):
        close=3000+index*.4
        result.append({
            "time":ASOF-(count-index)*step,
            "open":close-.2,"high":close+1.,"low":close-1.,"close":close,
            "tick_volume":100+index,
        })
    return result


def snapshot():
    return observe({
        "M5":bars(300),"M15":bars(900),"H1":bars(3600),
    },.41,closed_asof=ASOF)


def test_pkce_is_s256():
    verifier,challenge=make_pkce()
    expected=__import__("base64").urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    assert challenge==expected
    assert len(verifier)>=43


def test_credentials_status_and_environment_token(tmp_path,monkeypatch):
    credentials=ChatGPTPlanCredentials(tmp_path)
    assert credentials.status()=={"connected":False,"plan_usage":False}
    monkeypatch.setenv("OPENAI_CHATGPT_ACCESS_TOKEN","test-access-token")
    assert credentials.access_token()=="test-access-token"


def test_technical_indicators_are_causal_and_present():
    source=bars(300)
    indicators=technical_features(source)
    assert 0<=indicators["rsi14"]<=100
    assert set(indicators)>=set(("rsi14","ema9","ema21","macd_histogram",
                                 "bollinger_z20","fibonacci_range50","relative_volume20"))
    changed=[dict(row) for row in source]
    changed[-1]["close"]+=10
    changed[-1]["high"]+=10
    assert technical_features(changed)!=indicators
    assert technical_features(source)==indicators


def test_sol_uses_streaming_structured_responses(monkeypatch):
    captured={}
    decision={
        "action":"WAIT","reason":"Market evidence is not strong enough.",
        "confidence":.3,"stop":None,"target":None,"thesis":"Conflicting signals",
        "strategy_reference":None,
    }
    class Response:
        def __enter__(self): return self
        def __exit__(self,*_): return False
        def __iter__(self):
            encoded=json.dumps(decision)
            yield ("data: "+json.dumps({"type":"response.output_text.delta","delta":encoded})+"\n").encode()
            yield ("data: "+json.dumps({"type":"response.completed","response":{"output":[]}})+"\n").encode()
    def fake_urlopen(request,timeout):
        captured["payload"]=json.loads(request.data)
        captured["authorization"]=request.headers["Authorization"]
        return Response()
    monkeypatch.setattr("urllib.request.urlopen",fake_urlopen)
    model=OpenAISolDecisionModel(token_provider=lambda:"plan-token",verify_model=False)
    result=FreeDecisionAgent(model).decide(snapshot())
    assert result.action=="WAIT"
    payload=captured["payload"]
    assert payload["model"]==MODEL=="gpt-5.6-sol"
    assert payload["store"] is False and payload["stream"] is True
    assert isinstance(payload["input"],list)
    supplied=json.loads(payload["input"][0]["content"][0]["text"])
    assert supplied["observation"]["snapshot_id"]==snapshot()["snapshot_id"]
    assert payload["text"]["format"]["type"]=="json_schema"
    assert payload["text"]["format"]["schema"]==DECISION_SCHEMA
    assert captured["authorization"]=="Bearer plan-token"


def test_position_state_is_given_to_sol_proposal():
    seen={}
    class Model:
        def decide(self,observation,references):
            seen.update(observation)
            return {"action":"HOLD","reason":"Existing position remains valid.",
                    "confidence":.6,"stop":None,"target":None,
                    "thesis":"Trend intact","strategy_reference":None}
    decision=FreeDecisionAgent(Model()).decide(snapshot(),position_open=True,
                                                trading_context={"equity":10000})
    assert decision.action=="HOLD"
    assert seen["position"]=={"open":True}
    assert seen["trading_context"]=={"equity":10000}


def test_local_call_budget_is_conservative(tmp_path):
    budget=LocalCallBudget(tmp_path/"budget.json",max_calls=2)
    assert budget.record(1000)
    assert budget.record(1001)
    assert not budget.record(1002)
    assert budget.record(1000+5*60*60+1)


def test_sol_paper_proposes_but_cannot_order():
    class Item:
        balance=10000;equity=10020;margin_free=9500
    class Tick:
        bid=3020.;ask=3020.4
    class MT5:
        TIMEFRAME_M5=5;TIMEFRAME_M15=15;TIMEFRAME_H1=60
        def copy_rates_from_pos(self,symbol,timeframe,start,count):
            step={5:300,15:900,60:3600}[timeframe]
            return bars(step,count)
        def symbol_info_tick(self,symbol): return Tick()
        def account_info(self): return Item()
        def positions_get(self,**kwargs): return ()
    class Model:
        def decide(self,observation,references):
            assert observation["trading_context"]["risk_limits"]["max_risk_per_entry_pct"]==.25
            return {"action":"WAIT","reason":"No sufficiently strong setup is present.",
                    "confidence":.2,"stop":None,"target":None,
                    "thesis":"Stand aside","strategy_reference":None}
    closed,event=decide_once(MT5(),Model(),"XAUUSD")
    assert closed and event["status"]=="sol_proposal_no_order"
    assert event["decision"]["action"]=="WAIT"
    assert not hasattr(MT5,"order_send")
