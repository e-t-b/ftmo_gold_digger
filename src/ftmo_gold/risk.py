"""FTMO 2-Step risk gates. Amounts denominated in account currency."""
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from math import isfinite

@dataclass(frozen=True)
class RiskPolicy:
    initial_balance: float = 10000
    daily_loss_pct: float = .05
    total_loss_pct: float = .10
    internal_day_pct: float = .01
    risk_pct: float = .0035
    max_daily_entries: int = 3

class RiskGate:
    def __init__(self, policy=RiskPolicy()):
        self.p = policy
        self.day = None
        self.midnight_balance = None
        self.entries = 0
        self.halted = False

    def reset_day(self, now: datetime, midnight_balance: float):
        if now.tzinfo is None or not isfinite(midnight_balance):
            raise ValueError("Need aware timestamp and actual Prague-midnight balance")
        day = now.astimezone(ZoneInfo("Europe/Prague")).date()
        if day != self.day:
            self.day = day
            self.midnight_balance = midnight_balance
            self.entries = 0
            self.halted = False

    def floors(self):
        if self.day is None:
            raise RuntimeError("Midnight snapshot missing")
        p = self.p
        return (self.midnight_balance - p.initial_balance*p.daily_loss_pct,
                p.initial_balance*(1-p.total_loss_pct),
                self.midnight_balance - p.initial_balance*p.internal_day_pct)

    def evaluate(self, equity: float, proposed_risk: float, positions_open: int=0):
        if self.day is None or not isfinite(equity) or not isfinite(proposed_risk):
            return False, "invalid_state"
        floor = max(self.floors())
        if equity <= floor or self.halted:
            self.halted = True
            return False, "equity_limit"
        if self.entries >= self.p.max_daily_entries:
            return False, "trade_cap"
        if positions_open:
            return False, "position_open"
        budget = min(self.p.initial_balance*self.p.risk_pct, equity-floor)
        if proposed_risk <= 0 or proposed_risk > budget+1e-8:
            return False, "risk_budget"
        return True, "approved"

    def record_entry(self):
        self.entries += 1
