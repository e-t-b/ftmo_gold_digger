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

## GPT-5.6 Sol decision engine (historical and paper modes)

The agent is **not forced to select a named strategy**. Strategy playbooks
are optional evidence-poor references; the model may generate its own thesis,
combine ideas, or WAIT. This does **not** mean spontaneous AI theses are
authorized for funded-account execution.

Gold Digger is fixed to `gpt-5.6-sol`; it does not silently route to Astra or
to PPO. The preferred connection uses the eligible ChatGPT plan through
OpenAI's public-client PKCE flow, without an API key:

```powershell
pip install -e ".[openai,mt5,dev]"
python -m ftmo_gold.chatgpt_auth login
python -m ftmo_gold.chatgpt_auth status

# Choose an actual UTC epoch SECOND aligned to completed H1/M15/M5 bars.
python -m ftmo_gold.analyze --input data/raw --asof 1793102400 --spread 0.41 --chatgpt --reasoning-effort medium --output reports/decision_sol.json

# Persistent virtual portfolio on MT5 DEMO: about one Sol analysis per hour,
# 2% virtual risk, at most two daily entries, and locally enforced 1R–4R.
python -m ftmo_gold.sol_paper --symbol XAUUSD --reasoning-effort medium --max-calls-5h 5 --risk-pct 2 --max-daily-entries 2 --commission-per-lot-roundtrip 6
```

The login stores renewable credentials outside the repository under the
platform's local application-data directory. Tokens are never printed or
written to reports. The paper module refuses non-demo accounts and contains no
order-submission function. If the local five-hour call budget is exhausted it
waits instead of changing billing paths.

The paper scanner persists virtual entries and exits, sizes from executable
bid/ask and the proposed stop, applies spread, an explicit commission
assumption, conservative same-bar SL/TP ordering, a maximum holding time and
Prague-day entry limits. Its journal reports balance, equity, win rate, profit
factor and maximum drawdown. These are simulated fills, never broker orders.

A separately billed API key is available only through an explicit opt-in:

```powershell
$env:OPENAI_API_KEY = "YOUR_KEY"
python -m ftmo_gold.analyze --input data/raw --asof 1793102400 --spread 0.41 --paid-api --output reports/decision_sol_paid.json
```

The sample timestamps/spread are **illustrative**; choose the correct bar
boundary and observed spread for the requested historic instant.
Without a model, the agent deterministically returns WAIT.
With Sol 5.6, it requests a strict structured decision and rejects malformed,
stale, noncausal or internally inconsistent proposals. Inputs include causal
RSI, EMA, MACD, ATR, Bollinger position, 50-bar range/Fibonacci position and
relative volume when supplied by the broker. Both modes remain **no broker order**;
the code does not submit trades. See [V2 steps 1–3](docs/v2_steps_1_to_3.md).

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

## Apply a trained GRU to a historical market snapshot

The checkpoint now has a separate, reproducible inference adapter
supporting both three and five timeframe models. It uses only candles
closed at `--asof` (UTC Unix seconds), verifies continuity of intraday
sequences and rejects stale price context.

```powershell
python -m ftmo_gold.dl_inference --input data/gold_3y --checkpoint reports/gru_5tf/model.pt --asof 1793102400 --out reports/inference.json
```

Replace the timestamp with a genuine fully closed-bar checkpoint in your
data. This produces DOWN/NEUTRAL/UP class scores, **not** trade instructions
or calibrated probabilities of profit. No broker order is sent.

## Second 3-year 5TF experiment: binary direction V2

A separate Deep Learning research training (PyTorch GRU M5/M15/H1/H4/D1)
estimated 30-minute price direction on the user's ~3 years of FTMO data.
On the held-out April–October 2026 set it scored **49.40% accuracy**
vs **51.20% majority-class** and ROC AUC **0.5088**: still no edge.
The trained checkpoint and exact trainer are delivered as a conversation ZIP.

The binary network has a distinct 2-output checkpoint format. For it, use:

```powershell
python -m ftmo_gold.direction_inference --input data/gold_3y --checkpoint reports/direction_v2/model.pt --asof 1793102400 --out reports/direction_inference.json
```

Use a real completed market timestamp from the CSVs. To hand the
prediction to the discretionary AI agent, use the
`DeepLearningDecisionAgent` adapter in `ftmo_gold.dl_agent` with the
corresponding timestamped market observation. It does not force any strategy
nor place orders. See [real experiment results](docs/deep_learning_direction_v2_results.md).

## Phase de vérité V1 — backtest à trades

Un replay du checkpoint PyTorch GRU V3 **gelé** a produit des trades
simulés sur les données FTMO-Demo avril–octobre 2026. La politique
directionnelle simple (et **non l'agent LLM libre**) réalise 430 trades
et **-620,55 USD** net, avec un profit factor **0,874**, sous les
hypothèses Bid OHLC + spread M5 et slippage fixes. Au spread ×2 :
**-1 235,77 USD**.

**NO-GO pour le modèle courant**. Ce replay n'est ni un vrai
backtest de l'agent discrétionnaire ni une simulation FTMO complète en
equity flottante. Le script exact, le checkpoint, les trades CSV et les
rapports sont inclus dans l'archive livrée avec l'analyse.

[Rapport technique](docs/phase_verite_v1.md).

## Train Gold Digger PPO locally (no ZIP)

The **actual training source** is committed under `training/`.
After `git pull`, with your FTMO historical CSVs already under
`data/gold_3y/`:

```powershell
pip install -e ".[dl,dev]"
python training/train_massive.py --input data/gold_3y --out reports/rl_reinforced --episodes 512 --save-every 128 --validate-every 128 --seed 17
```

Stop with Ctrl+C. Rerun the same command to resume from
`reports/rl_reinforced/latest.pt`; `progress.json` shows progress.
`best_validation.pt` is selected on the worst of three chronological
validation blocks, rather than on the aggregate peak.

The reinforced trainer uses executable-action masking, a 0.25% risk cap,
16-day PPO rollout batches, GAE, four update epochs, a 0.025 entropy
coefficient, position PnL context, and immediate termination after the third
closed trade. Checkpoints are loaded with PyTorch's restricted
`weights_only=True` loader. Old `8action-ppo-v1` checkpoints are diagnostic
inputs only and cannot resume the reinforced trainer.

Technical indicators are available as controlled research feature sets. The
default remains `basic`, because the multi-seed ablation was more robust than
the larger sets. `compact` adds RSI 14, EMA 9/21, MACD histogram, ATR 14 and
relative volume. `technical` also adds Bollinger position and a causal
50-period Fibonacci-range position:

```powershell
python training/train_massive.py --input data/gold_3y --out reports/rl_indicators --episodes 512 --feature-set compact --hidden-size 32 --seed 17
```

Every indicator is computed from the current or earlier closed bars only.
Feature-set and model-size changes require a new output directory and a fresh
training run.

For a validation-only comparison of an old checkpoint before retraining:

```powershell
python training/diagnose_policy.py --input data/gold_3y --checkpoint reports/rl_massive/best_validation.pt --output reports/policy_diagnostic.json
```

No previous chat ZIP is required to execute this.
**The earlier 100-episode binary checkpoint is NOT committed to GitHub**:
the GitHub version trains from scratch unless you already have a
compatible `latest.pt` in the chosen output directory.
The trainer is research-only, not production FTMO-compliant or profitable.
See [training report](docs/resumable_massive_training.md).

## 4. Testing

```powershell
pytest -q
```

GitHub Actions also runs unit tests. No credentials should be committed;
`data/`, `reports/` and `.env` are excluded in `.gitignore`.

FTMO official information: https://ftmo.com/fr/2-step-challenge/ and
https://ftmo.com/fr/faq/quels-instruments-puis-je-trader-et-quelle-strategie-je-suis-autorise-a-utiliser/
