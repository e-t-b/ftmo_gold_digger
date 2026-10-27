import csv
from ftmo_gold.backtest import simulate, load_bars, round_down_lots


def rows():
    bars = [dict(time=i*300,open=100.,high=101.,low=99.,close=100.) for i in range(20)]
    bars.append(dict(time=6000,open=100.,high=103.,low=99.,close=102.))
    bars.append(dict(time=6300,open=102.,high=104.,low=100.,close=102.))
    bars.append(dict(time=6600,open=102.,high=109.,low=100.,close=108.))
    return bars


def test_lot_rounding():
    assert round_down_lots(.009,.01,.01)==0
    assert round_down_lots(.037,.01,.01)==.03


def test_backtest_has_no_same_bar_entry():
    result=simulate(rows(),spread=.2,slippage=0,contract_size=100,
                    minimum_lot=.01,lot_step=.01)
    assert result["trades"]==1
    assert result["transactions"][0]["opened"]==6300
    assert result["transactions"][0]["reason"]=="target"
    assert result["final"]>10000


def test_reject_bad_ordering(tmp_path):
    p=tmp_path/"bars.csv"
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["time","open","high","low","close"])
        w.writeheader()
        w.writerows([dict(time=300,open=1,high=2,low=0,close=1),
                     dict(time=0,open=1,high=2,low=0,close=1)])
    try:
        load_bars(p)
    except ValueError:
        pass
    else:
        raise AssertionError("Unordered bars accepted")
