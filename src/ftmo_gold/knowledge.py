"""Advisory strategy playbook: context for an agent, NEVER mandatory trading rules."""
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Playbook:
    name: str
    idea: str
    favorable_context: str
    failure_modes: str
    evidence: str = "UNVALIDATED: research hypothesis; no reliable performance estimate"
    mandatory: bool = False

PLAYBOOKS = (
    Playbook("trend_pullback", "Continuation after a pullback toward a prior structure level",
             "persistent directional structure, controlled retracement",
             "late trend, news shock, shallow liquidity, range chop"),
    Playbook("liquidity_sweep", "Take a rejection after a prior high/low was briefly crossed",
             "well-observed prior extreme, rejection confirmed on closed candles",
             "breakout continuation, spread spike, level mined from hindsight"),
    Playbook("expansion_breakout", "Momentum after a compressed range is left",
             "fresh range, volatility expansion, clean session liquidity",
             "false breakout, exhaustion, adverse quote spread"),
    Playbook("range_reversion", "Mean reversion inside a verified sideways structure",
             "repeated range boundaries, low directional efficiency",
             "range breakdown, persistent trend, event risk"),
)

def playbooks():
    return [asdict(item) for item in PLAYBOOKS]
