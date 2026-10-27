# Gold Digger V2 — implementation checkpoint: steps 1–3

## What has actually shipped

### Step 1 — Data foundation and causal market observation
- Existing collector remains read-only and limited to MT5 demo.
- `market.py` provides multi-timeframe ATR/efficiency/regime features,
  missing-window rejection, OHLC/chronology checks and tick-spread checks.
- `decision.observe()` builds a timestamped M5/M15/H1 snapshot with a
  deterministic snapshot ID and rejects partial/future/stale candle windows.
- Input values are derived from closed historical bars, not future prices.
- Historical quotes remain stored locally by the account holder, not GitHub.

### Step 2 — Discretionary AI decision module
- `FreeDecisionAgent` takes a model implementing `decide(observation, references)`.
- An optional API-based research adapter exists in `model_openai.py`, requires
  `OPENAI_API_KEY` and an explicit CLI `--model` argument.
- Model may propose BUY/SELL/WAIT, an original thesis and a free-form strategy
  reference. An existing playbook is **never** a prerequisite for a proposal.
- Validator checks finite values, direction/stop/target consistency, action
  schema and whether CLOSE/HOLD requires an open position. On exceptions,
  agent returns WAIT. No broker order interface is wired.
- The model is NOT trained, calibrated, demonstrated profitable or actually
  trading. A single market snapshot cannot validate a new idea.

### Step 3 — Advisory strategy library
- `knowledge.py` lists trend pullback, liquidity sweep, breakout expansion
  and range reversion as *non-mandatory* ideas with failure modes and
  explicit UNVALIDATED evidence labels.
- Existing deterministic signal implementations are retained as experimental
  baselines. They do not constrain the V2 decision interface.

## Test coverage
Tests target causality, stale/partial candle rejection, missing/gapped bars,
strategy-reference freedom, invalid model output, fail-closed fallback, and
unchanged earlier collection/backtest logic.
CI runs `python -m pytest -q` on push.
See GitHub Actions for the verified final state.

## Reproducible commands (research only)
```powershell
git pull
pip install -e ".[dev]"
python -m pytest -q
python -m ftmo_gold.research --input data/raw --out reports/market_research.json
python -m ftmo_gold.analyze --input data/raw --asof <UTC_EPOCH_SECONDS> --spread <OBSERVED_PRICE_SPREAD>
```

Replace angle-bracket placeholders. Timestamp must correspond to an available
completed H1/M15/M5 checkpoint. To use actual AI model inference add an
explicit `--model` and locally configured `OPENAI_API_KEY`; data is then
sent to the API. No account numbers, passwords or keys belong in Git.

## Limitations and next gate
1. No historical decision replay yet (V2 step 4), no out-of-sample P&L or
   FTMO challenge-pass estimates from the discretionary agent.
2. Model confidence is uncalibrated; explanation is not statistical evidence.
3. Spread snapshot is supplied by caller and must be historical for replay.
4. Policy for simultaneous positions, account state, and price execution
   remains deliberately unimplemented.
5. A candidate AI trade is not a trade that can be authorized automatically
   on FTMO without validation.
6. Existing M5 baseline is negative on the preliminary supplied data and is
   not approved for deployment.

**Status: steps 1–3 implemented for research. No live/demo order execution.**
