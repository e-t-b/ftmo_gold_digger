# FTMO Gold Digger

Research and prototype for an eventual autonomous **XAUUSD** system, aiming
to test whether a strategy could satisfy **FTMO 2-Step** rules. No evidence of
profitability exists yet. **No orders are submitted by this repository.**

## Install (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -e ".[mt5,dev]"
pytest -q
```

## 1. Collect demo-only MT5 XAUUSD data

Open the FTMO MT5 **demo** terminal and log in. Check the exact symbol:

```powershell
python -m ftmo_gold.collector --symbol XAUUSD --days 90 --ticks-days 2 --out data/raw
```

Produces `m1.csv`, `m5.csv`, `m15.csv`, `h1.csv`, `ticks.csv` and
`quality.json`. The collector refuses non-demo accounts and does not send
orders. Missing history cannot be reconstructed; quality report requires review.

## 2. Baseline research backtest

**Illustrative parameters ONLY — replace with actual broker specs and costs.**
OHLC is assumed BID. The spread is XAUUSD price units, **not pips**.

```powershell
python -m ftmo_gold.backtest --csv data/raw/m5.csv --out reports/baseline.json --spread 0.25 --slippage 0.05 --contract-size 100 --min-lot 0.01 --lot-step 0.01 --commission 0
```

See `docs/architecture.md` for major simulation limitations. Results may be
misleading without historical bid/ask, broker data, margin and swaps. The
implemented 20-bar M5 breakout is an unvalidated **engineering baseline**,
not our final Gold strategy. FTMO challenge-pass simulation and AI integration
are not yet implemented.

## 3. Testing

```powershell
pytest -q
```

GitHub Actions also runs unit tests. No credentials should be committed;
`data/`, `reports/` and `.env` are excluded in `.gitignore`.

FTMO official information: https://ftmo.com/fr/2-step-challenge/ and
https://ftmo.com/fr/faq/quels-instruments-puis-je-trader-et-quelle-strategie-je-suis-autorise-a-utiliser/
