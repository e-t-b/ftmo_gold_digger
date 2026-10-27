"""Hybrid decision interface; AI veto-only, deterministic fallback.

No network or trading operations. A production AI adapter needs frozen
features, historical replay and an out-of-sample ablation study.
"""
from dataclasses import dataclass
from typing import Protocol
from .signals import Signal


class Supervisor(Protocol):
    def approve(self, signal: Signal, context: dict) -> bool: ...


@dataclass
class HybridAgent:
    supervisor: Supervisor | None = None

    def decide(self, signal: Signal | None, context: dict | None = None):
        if signal is None:
            return None
        if self.supervisor is None:
            return signal
        try:
            return signal if self.supervisor.approve(signal, context or {}) is True else None
        except Exception:
            return None  # fail closed on timeout / invalid AI output
