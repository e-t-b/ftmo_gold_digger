# Steps 1–3 — XAUUSD research status (2026-10-27)

## Scope and reproducibility
This is a RESEARCH implementation, not a deployed trading agent.
The user's raw FTMO ticks/bars stay local. Only aggregate findings here.

### 1. Data audit and session-aware observations
Source: FTMO-Demo XAUUSD; 2026-07-29 through 2026-10-27.
M1=87,124; M5=17,426; M15=5,873; H1=1,470.
Ticks=407,977, approximately the latest 2 days.
Collector metadata: contract_size=100 oz/lot, minimum/step=0.01 lot,
tick_size=0.01, account USD.
The original collector reports 0 invalid OHLC, 0 unordered timestamps,
65 M5 gaps (mostly market closures, not presumed to be missing data).
The market audit explicitly enumerates gaps; no forward filling.

Tick sampled median observed spread USD/oz ~0.41; typical UTC-hour
medians in this 2-day sample: 01:00=.49, 03:00=.46,
09:00=.39, 10:00=.38, 15:00=.41, 22:00=.40.
This is a limited tick snapshot; not a 90-day spread history.

### 2. Regimes
`ftmo_gold.market.features()`: closed M5 candles only, 14-bar volatility
and efficiency, 48-bar background, exclusion of windows across gaps.
Four/five states: volatile, trending_up, trending_down, ranging, uncertain.
Initial research thresholds are explicit hypotheses, NOT tuned/validated
classifiers; DST conversion uses Europe/London and America/New_York.

Local diagnostic on the supplied historical M5 series, 1 closed bar
per sample after warm-up, with session/gap windows excluded:
- volatile 1,107
- uncertain 7,282
- ranging 5,734
- trending_up 35
- trending_down 23

These counts show the trend classifier is *overly restrictive* and
must not be treated as a useful or predictive edge yet.

### 3. Strategy candidate library
`ftmo_gold.strategies.candidates()`: independent predefined hypotheses:
trend_pullback, liquidity_sweep, expansion_breakout, range_reversion.
All use completed consecutive M5 windows, preceding regime features,
fixed version tag `candidate-v1-2026-10-27`, reference prices and
structural invalidation. Candidate generated at CLOSE; execution, if
ever authorized, belongs to the following tradable tick, not at CLOSE.

Counts from the same local M5 diagnostic:
- trend_pullback: 3
- liquidity_sweep: 1,456
- expansion_breakout: 527
- range_reversion: 823

Multiple candidates can occur at the same candle; these are NOT
position entries, filled trades, signals to buy/sell, or profits.
Frequency ≠ performance. No strategy is approved for trading.

## Exact audit command
With the previous Windows MT5 demo collection already on disk:

```powershell
git pull
pip install -e ".[dev]"
python -m pytest -q
python -m ftmo_gold.research --input data/raw --out reports/market_research.json
```

The report includes quality findings and per-regime candidate counts.
Raw account data are ignored by Git; do not commit `data/raw`.

## Gates before step 4
1. Audit irregular gaps by broker session calendar, not just bar interval.
2. Obtain independent history and deeper bid/ask ticks.
3. Validate regime stability and candidate counts across periods.
4. Build independent cost-aware tick execution and portfolio/FTMO replays.
5. Walk-forward test candidates BEFORE training an adaptive selector.
6. Add AI only if it measurably beats a frozen deterministic baseline.
No autonomous order execution is enabled.
