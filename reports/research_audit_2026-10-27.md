# XAUUSD FTMO-Demo — first historical audit (2026-10-27)

## Input supplied by account owner
- Exact instrument XAUUSD, server FTMO-Demo, currency USD
- M1: 87,124 candles; M5: 17,426; M15: 5,873; H1: 1,470
- Bid/ask ticks: 407,977, covering approximately 2 days
- Time range of bars: 2026-07-29 through 2026-10-27
- 0 duplicated/unordered and 0 invalid OHLC candles per collector report
- M5 gaps: 65 (primarily known market closures; audit atypical gaps separately)
- Contract: 100 ounces/lot; minimum volume 0.01; step 0.01; tick size .01; tick value $1
- Median sampled bid-ask spread: $0.41/oz; p90 $0.46/oz; p99 $0.49/oz.
  Recent 2-day spread distribution is NOT a 90-day historical spread series.

## Research run — illustrative, NOT validated
Standalone OHLC research replay of the current 20-prior-closed-M5 breakout:
- Entries at next candle open, bid OHLC with synthetic ask at +$0.41/oz
- Slippage assumption: $0.05/oz adverse each side
- SL opposite previous 20-bar range, TP at 2R from executed entry
- Up to 3 entries per Prague day; target $35 risk, 0.01-lot increments
- Exit after max 12 bars or end of Prague day; pessimistic stop-first collision
- Commission and swaps assumed zero; no margin or intrabar tick-path simulation

Measured in the standalone replay:
- 205 closed trades, 96 winners / 109 losers
- Win rate 46.83%; profit factor 0.9236
- Approx. P&L -$127.49, ending balance $9,872.51

**Decision:** reject this generic breakout as deployment candidate on current evidence.
This replay is NOT validated as identical to the repository backtest runner;
do not equate the numbers to a Strategy Tester or exact broker fills.
No out-of-sample, walk-forward, full FTMO challenge simulation, or AI uplift
has been measured. Raw user data was NOT committed.

## Priority follow-ups
1. Verify differences between standalone replay and repo runner with same inputs.
2. Inspect 65 M5 gaps, atypical feed outages and market-day calendar.
3. Reconstruct spread by time-of-day from ticks; stress costs on historical M5.
4. Collect >= 2 years of distinct unexamined GOLD history before choosing rules.
5. Compare London sweep, NY opening range, and trend pullback using frozen rules.
6. Only after robust holdout tests build DEMO order executor with independent risk monitor.
