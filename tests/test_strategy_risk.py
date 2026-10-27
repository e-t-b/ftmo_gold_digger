from datetime import datetime, timezone
from ftmo_gold.agent import HybridAgent
from ftmo_gold.risk import RiskGate
from ftmo_gold.signals import breakout


def example():
    bars = [dict(time=i * 300, open=100, high=101, low=99, close=100) for i in range(20)]
    bars.append(dict(time=6000, open=100, high=103, low=99, close=102))
    return bars


def test_breakout_uses_closed_bars():
    assert breakout(example()[:-1]) is None
    s = breakout(example())
    assert s.side == "buy" and s.stop == 99 and s.target == 108


def test_gate_and_latching():
    g = RiskGate()
    assert g.evaluate(10000, 35)[0] is False
    g.reset_day(datetime(2026, 10, 27, tzinfo=timezone.utc), 10000)
    assert g.evaluate(10000, 35)[0]
    assert not g.evaluate(9899, 35)[0]
    assert not g.evaluate(10000, 35)[0]  # latched until next day


def test_ai_only_vetoes():
    class Reject:
        def approve(self, signal, context):
            return False
    class Fails:
        def approve(self, signal, context):
            raise RuntimeError()
    s = breakout(example())
    assert HybridAgent().decide(s) == s
    assert HybridAgent(Reject()).decide(s) is None
    assert HybridAgent(Fails()).decide(s) is None
