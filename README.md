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

## 3. Autonomous demo signal scanner (NO ORDER EXECUTION)

```powershell
python -m ftmo_gold.paper --symbol XAUUSD --interval 10 --out data/paper/decisions.jsonl
```

The scanner runs continuously with the demo terminal open, consumes only
completed M5 bars, records each candle's candidate signal at most once per
process, and **never calls MT5 order_send**. It does **not** execute,
size or supervise positions. Press Ctrl+C to stop it. A future release
will add persisted state, reconciliation, paper fills and a guarded
demo execution adapter.

## Market analysis and strategy research (steps 1–3)

After collecting XAUUSD data from the exact FTMO demo server:

```powershell
python -m ftmo_gold.research --input data/raw --out reports/market_research.json
```

This generates gap/quality statistics, recent tick spread percentiles,
causal M5 regime frequencies and counts of four strategy candidates.
**Counts are not filled trades and do not establish profitability.**
See [steps 1–3 report](docs/steps_1_to_3.md). No trading is enabled.

## V2: discretionary decision research (steps 1–3)

The agent is **not forced to select a named strategy**. Strategy playbooks
are optional evidence-poor references; the model may generate its own thesis,
combine ideas, or WAIT. This does **not** mean spontaneous AI theses are
authorized for funded-account execution.

```powershell
# Choose an actual UTC epoch SECOND within your locally collected history,
# aligned to the latest completed H1/M15/M5 bar:
python -m ftmo_gold.analyze --input data/raw --asof 1793102400 --spread 0.41 --output reports/decision.json

# Optional, explicit OpenAI API use (uses your own API key and incurs costs):
$env:OPENAI_API_KEY = "YOUR_KEY"
python -m ftmo_gold.analyze --input data/raw --asof 1793102400 --spread 0.41 --model gpt-4.1-mini --output reports/decision_ai.json
```

The sample timestamps/spread are **illustrative**; choose the correct bar
boundary and observed spread for the requested historic instant.
Without a model, the agent deterministically returns WAIT.
With a model, it asks for a JSON decision and rejects malformed, stale,
noncausal or internally inconsistent proposals. Both modes are **research-only**;
the code does not submit orders. See [V2 steps 1–3](docs/v2_steps_1_to_3.md).

## Deep Learning V1 — real training on FTMO GOLD

A three-branch PyTorch GRU reads strictly closed M5 / M15 / H1 sequences.
It predicts a 30-minute 3-class price movement label; it does **not**
execute trades or establish a profitable strategy.

```powershell
pip install -e ".[dev,dl]"
python -m ftmo_gold.deep_learning --input data/raw --out reports/gru_v1 --epochs 6
```

The command writes a trained checkpoint (`model.pt`) and metrics
(`report.json`) to the ignored reports directory. The initial 90-day
FTMO-Demo training reached **40.30% held-out test accuracy**, below
the **48.71% majority-class baseline**. It is **not deployable**.
Read [the full GRU training report](docs/gru_training_2026-10-27.md).

## Long-history XAUUSD (H4/D1 enabled)

To collect up to three years **if the FTMO server actually supplies them**:

```powershell
git pull
pip install -e ".[mt5,dev,dl]"
python -m ftmo_gold.collector --symbol XAUUSD --days 1095 --ticks-days 2 --out data/gold_3y
```

The collector now exports `h4.csv` and `d1.csv` as well. Verify the
actual history span in `data/gold_3y/quality.json` before training.
When history is sufficient, train the optional five-branch network:

```powershell
python -m ftmo_gold.deep_learning --input data/gold_3y --out reports/gru_5tf --with-h4-d1 --epochs 6
```

This requires broker-provided historical bars. Results have NOT yet been
measured on the five-branch architecture. Read [long-history guidance](docs/long_history_5tf.md).

## 4. Testing

```powershell
pytest -q
```

GitHub Actions also runs unit tests. No credentials should be committed;
`data/`, `reports/` and `.env` are excluded in `.gitignore`.

FTMO official information: https://ftmo.com/fr/2-step-challenge/ and
https://ftmo.com/fr/faq/quels-instruments-puis-je-trader-et-quelle-strategie-je-suis-autorise-a-utiliser/
