# Long-history XAUUSD and five-timeframe GRU — 2026-10-27

## Goal
Train a temporal Deep Learning agent on years of XAUUSD market data rather than
only ~90 days, incorporating H4 and D1 for broader trend/positioning context.

## Implemented
- MT5 demo collector exports M1, M5, M15, H1, **H4 and D1** to CSV.
- Each timeframe's quality report compares requested versus actual coverage.
- PyTorch GRU can train on M5/M15/H1 (V1) or optionally all five branches
  M5/M15/H1/H4/D1 (`--with-h4-d1`). Original model remains compatible.
- The five-branch training has NOT yet been run on years of real FTMO data;
  it has only synthetic/unit-test coverage until new exports are supplied.

## Run on user's Windows machine

In MetaTrader 5: Tools > Options > Charts > Max. bars in chart: raise enough for
requested M1 duration, restart MT5 if needed. Older data availability is
limited by the broker/server and local chart history; more days requested do
NOT guarantee more history returned.

```powershell
git pull
pip install -e ".[mt5,dev,dl]"
python -m ftmo_gold.collector --symbol XAUUSD --days 1095 --ticks-days 2 --out data/gold_3y
```

Check `data/gold_3y/quality.json`: each TF's `actual_span_days`,
`may_be_history_limited`, gaps and invalid OHLC. Prefer export of multi-year
M1/M5 history plus H4/D1 context, and ticks where broker retention allows it.
An M1 series over three years can exceed a million rows; do not expect
instant transfers or full tick retention.

Only if actual coverage is adequate:

```powershell
python -m ftmo_gold.deep_learning --input data/gold_3y --out reports/gru_5tf --with-h4-d1 --epochs 6
```

The new model is an **experiment**. Treat the first 90-day 3-branch checkpoint
as a historical baseline, not a deployable strategy.

## Data reliability gates
- MT5 rates use UTC bar OPEN timestamps; available bars are constrained by
  chart max-bars and server availability. MQL5 documentation:
  https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py
- H4/D1 bars may have broker-dependent rollover/session semantics, especially
  around DST. Daily context is historical only; exchange closures must never
  be turned into synthetic price bars. HTF branch permits natural nontrading
  gaps; this needs more session-calendar tests with actual broker history.
- Market data from another feed should not be naively joined to FTMO prices.
  Differences in timezone, rollovers, spread and instrument must be audited.
- Strictly hold out future periods; extra data can improve coverage but can
  also introduce distribution shift. Revalidate predictive performance.
- This work does not authorize autonomous order placement or claim profits.
