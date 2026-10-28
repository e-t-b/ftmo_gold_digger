from types import SimpleNamespace

from ftmo_gold.decision import Decision
from ftmo_gold.paper_portfolio import PaperPortfolio


INFO=SimpleNamespace(trade_contract_size=100.,volume_min=.01,
                     volume_step=.01,volume_max=10.)


def decision(action,stop=None,target=None):
    return Decision(action=action,reason="Deterministic paper test decision.",
                    confidence=.5,stop=stop,target=target)


def test_virtual_buy_is_risk_sized_and_stopped_conservatively(tmp_path):
    portfolio=PaperPortfolio(tmp_path/"state.json","XAUUSD",10000,
                             commission_per_lot_roundtrip=4.)
    fill=portfolio.apply_decision(decision("BUY",2995.,3010.),3000.,3000.2,
                                  1000,900,INFO,"snapshot")
    assert fill["type"]=="virtual_open"
    assert fill["lots"]==.04
    portfolio.save()
    restored=PaperPortfolio(tmp_path/"state.json","XAUUSD",10000,
                            commission_per_lot_roundtrip=4.)
    trade=restored.process_closed_bar(
        {"high":3012.,"low":2994.},1800,2994.8,2995.,100.,.2)
    assert trade["exit_reason"]=="stop"
    assert trade["exit"]==2995.
    assert trade["net_pnl"]==-20.96
    metrics=restored.metrics()
    assert metrics["closed_trades"]==1 and metrics["losses"]==1
    assert metrics["net_pnl"]==-20.96


def test_virtual_sell_target_and_model_close(tmp_path):
    portfolio=PaperPortfolio(tmp_path/"state.json","XAUUSD",10000)
    fill=portfolio.apply_decision(decision("SELL",3005.,2990.),3000.,3000.2,
                                  1000,900,INFO,"snapshot")
    assert fill["lots"]==.05
    trade=portfolio.process_closed_bar(
        {"high":3001.,"low":2988.},1800,2989.,2989.2,100.,.2)
    assert trade["exit_reason"]=="target" and trade["net_pnl"]==50.
    portfolio.apply_decision(decision("BUY",2980.,3010.),2990.,2990.2,
                             3000,2700,INFO,"next")
    closed=portfolio.apply_decision(decision("CLOSE"),2992.,2992.2,
                                    3600,2700,INFO,"close")
    assert closed["exit_reason"]=="model_close"
    assert portfolio.metrics()["closed_trades"]==2


def test_two_percent_risk_and_one_to_four_r_filter(tmp_path):
    portfolio=PaperPortfolio(tmp_path/"state.json","XAUUSD",10000,
                             risk_fraction=.02,max_daily_entries=2)
    rejected=portfolio.apply_decision(decision("BUY",2995.,3003.),3000.,3000.2,
                                      1000,900,INFO,"poor-rr")
    assert rejected["reason"]=="reward_risk_out_of_range"
    assert portfolio.state["position"] is None
    fill=portfolio.apply_decision(decision("BUY",2995.,3010.),3000.,3000.2,
                                  1000,900,INFO,"valid-rr")
    assert fill["lots"]==.38
    assert fill["risk_budget"]==200.
    assert 1<=fill["reward_risk"]<=4
