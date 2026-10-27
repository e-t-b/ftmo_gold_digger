# FTMO Gold Digger — system architecture

## Status
Research only. No broker order submission is implemented. No profitable XAUUSD
strategy, FTMO pass probability or AI uplift has been established.

## Components
1. `collector.py`: DEMO-only MT5 market data extraction, bars and ticks.
2. `signals.py`: deterministic causal M5 breakout hypothesis (unvalidated).
3. `backtest.py`: pessimistic stop-first OHLC baseline, explicitly configured
   spread/slippage/contract/commission assumptions. NOT an MT5 Strategy Tester.
4. `risk.py`: FTMO Prague-day entry veto, subject to production reconciliation.
5. `agent.py`: veto-only optional AI supervisor interface, no provider connected.

## Autonomy target
Windows MT5 + Python scheduled supervisor:
- Synchronize terminal clock and Prague-day balance snapshots
- Pull complete closed bars and live bid/ask; detect missing/stale quotes
- Generate candidate signal; optionally request AI veto with immutable features
- Size using MT5 order_calc_profit, volume step, margin and stop levels
- Independently enforce FTMO equity and exposure constraints
- Submit/reconcile DEMO orders with idempotent client IDs and persistent ledger
- Circuit-breakers: connection loss, max daily trades, risk thresholds, missing snapshot
- Record every proposal, rejection, fill and error, then alert operator

This architecture must be validated in backtests and demo BEFORE permitting
autonomous order submissions; live trading remains unimplemented and locked.

## Research gates
1. Audit true FTMO server specs and history; refuse non-demo.
2. Test >=3 distinct candidate strategy hypotheses, frozen before holdout.
3. Assess costs with real bid/ask ticks and contract size; compare to MT5
   Strategy Tester `Every tick based on real ticks`.
4. Walk-forward splits, negative controls, out-of-sample and stress scenarios.
5. Simulate FULL FTMO two-step phases, including all rule variants of account.
6. A/B evaluate quantitative baseline versus AI veto on untouched data.
7. Forward-test for weeks using demo and independent risk controls.
8. Explicit go/no-go decision. No profitability guarantees.

## Known research-model limits
- On bid OHLC with synthetic constant spread, fills are approximations.
- Same-bar SL and TP is resolved in favor of SL.
- No actual tick path, variable spreads, swaps, margin or partial fills.
- Prague midnight is enforced by flattening at the last bar close; cannot
  guarantee a real fill there.
- `RiskGate` checks entry only; live open-risk supervision is future work.
- Real FTMO rules, including any best-day consistency rule applicable to the
  chosen product/version, must be checked separately before evaluation.
