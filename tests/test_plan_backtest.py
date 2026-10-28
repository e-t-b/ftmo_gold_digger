import json

from ftmo_gold.plan_backtest import load_plans, replay


def plan(created=300):
    return {"plan_id":"plan-one","snapshot_id":"snapshot","model":"gpt-5.6-sol",
            "created_at":created,"valid_until":created+3600,"mode":"CONDITIONAL",
            "reason":"Cross above the verified level with follow-through.","confidence":.7,
            "setups":[{"setup_id":0,"label":"breakout","direction":"BUY",
                       "trigger":"CLOSE_ABOVE","trigger_level":101.,"stop":99.,
                       "reward_risk":2.,"max_spread":.5}]}


def bars():
    return [
        {"time":0,"open":100.,"high":101.,"low":99.,"close":100.},
        {"time":300,"open":100.,"high":103.,"low":99.5,"close":102.},
        {"time":600,"open":102.,"high":110.,"low":101.,"close":109.},
        {"time":900,"open":109.,"high":110.,"low":108.,"close":109.},
    ]


def test_cached_plan_enters_only_on_next_bar_open():
    result=replay(bars(),[plan()],spread=.2,slippage=0,commission_per_lot_roundtrip=6.)
    assert result["triggered_plans"]==1 and result["closed_trades"]==1
    trade=result["transactions"][0]
    assert trade["opened_at"]==600
    assert trade["entry"]==102.2
    assert trade["exit_reason"]=="target"
    assert result["net_pnl"]>0


def test_expired_or_too_wide_spread_plan_does_not_trade():
    expired=plan();expired["valid_until"]=500
    assert replay(bars(),[expired],spread=.2)["closed_trades"]==0
    assert replay(bars(),[plan()],spread=.6)["closed_trades"]==0


def test_load_plans_accepts_journal_wrapper(tmp_path):
    path=tmp_path/"plans.jsonl"
    path.write_text(json.dumps({"status":"plan","plan":plan()})+"\n",encoding="utf-8")
    assert load_plans(path)[0]["plan_id"]=="plan-one"


def test_enter_now_plan_still_enters_next_historical_bar():
    immediate=plan(created=300)
    immediate["setups"][0]["trigger"]="ENTER_NOW"
    result=replay(bars(),[immediate],spread=.2,slippage=0)
    assert result["transactions"][0]["opened_at"]==300
