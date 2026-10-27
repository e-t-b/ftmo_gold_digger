# FTMO Gold — Phase 0: data audit

Independent, **read-only** XAUUSD MT5 collector. No orders are submitted. Requires Windows, a running MetaTrader 5 terminal, and a logged-in **demo** account. No passwords stored.

## Setup (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[mt5,dev]"
python -m ftmo_gold.collector --symbol XAUUSD --days 30 --ticks-days 2 --out data/raw
pytest -q
```

Use the **exact** MT5 symbol; verify it in Market Watch. Collector refuses non-demo accounts and never calls `order_send`.

Outputs: `m1.csv`, `m5.csv`, `m15.csv`, `h1.csv`, `ticks.csv`, `quality.json`. Raw files are ignored by Git. The gap report includes normal market closures; interpret by session schedule. Historical ticks may be incomplete; MT5 chart max bars and broker retention can limit depth. The report records symbol metadata, not account credentials.

## Next gates

1. Verify symbol identity, timestamps, DST, historical coverage and bid/ask availability.
2. Build realistic execution simulator (spread, slippage, commission, swap, minimum lot, FTMO equity rules).
3. Define three frozen candidate strategies and walk-forward splits.
4. Evaluate quantitative baseline versus AI filter on untouched data.
5. Forward-test in demo; only then consider FTMO challenge execution.

No profitability or challenge-pass probability is claimed.
