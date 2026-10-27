# Gold Digger Deep Learning V3 — volatility barrier outcome training (2026-10-27)

**Status: actual experiment completed on FTMO-Demo XAUUSD, 2023-10-30 through 2026-10-27. Not a profitable trader.**

## What changed

This is still PyTorch Deep Learning (not a substitute ML estimator). A
five-branch GRU reads closed M5/M15/H1/H4/D1 candles. The head predicts
one of three future first-touch scenarios over **60 minutes**:

- DOWN: the price's low touches a downward threshold first;
- UP: the price's high touches an upward threshold first;
- WAIT: neither barrier is reached or both are touched inside the same bar
  (unresolvable chronological order with OHLC).

For each past decision timestamp, the barrier distance is

`max($1.00, 1.25 * ATR14_on_past_M5, 1.5 * $0.82)`.

Here $0.82 is an **assumption**, not verified historical round-trip execution
cost. Prices and barrier labels are bid-based OHLC; they are NOT real broker
fills. ATR is computed from past candles only; the future candles are used
**only to create supervised labels**, never as neural network inputs.

The five encoders see the last 32 M5, 16 M15, 6 H1, 12 H4, and 10 D1 bars.
Each bar has 7 features: log return, candle range, candle body,
upper/lower wicks, tick volume, and spread proxy.

48,161 eligible historical snapshots (every third M5 candle, excluding
windows with intraday gaps) were divided by chronological calendar time
into 65/15/20 segments, with a 7-day gap around boundaries. Feature
normalization statistics come only from training data. The model was
trained for 5 epochs with AdamW; the checkpoint was chosen via validation
cross-entropy.

## Actual measured results

| Partition | Samples | Accuracy | Majority baseline |
|---|---:|---:|---:|
| Train | 31,038 | 49.188% | 46.247% |
| Validation | 6,554 | 46.277% | 47.711% |
| Test | 9,299 | 47.016% | 47.758% |

Test predicted counts: DOWN 7,042, WAIT **0**, UP 2,257.
Test true counts: DOWN 4,441, WAIT 731, UP 4,127.

**Conclusion:** This experiment does not improve on the naive test
baseline and completely misses the WAIT class. No evidence of a
profitable trading policy or valid FTMO challenge pass rate. These tests
also reuse the broad Apr–Oct 2026 evaluation interval already examined
for V1/V2; it should not be called a fresh untouched holdout for future
model selection.

## Reproducible deliverable

The user-facing file `gold_digger_barrier_v3_3years.zip` contains:
- `train_gold_barrier_v3.py` — actual executable training/labeling script;
- `train_gold_direction_v2.py` — shared causal five-timeframe data loader;
- `gold_barrier_v3/model.pt` — actually trained 5TF GRU checkpoint;
- `gold_barrier_v3/report.json` — full metrics, confusion matrices,
  training epochs and chronological periods;
- `test_barrier_v3.py` — small independently executed unit tests
  (2 passed locally).

Run in a Python environment with PyTorch, pandas, NumPy, scikit-learn:
`python train_gold_barrier_v3.py --input path/to/gold_csv --output gold_barrier_v3 --epochs 5`.

Dataset exports/checkpoints are kept out of GitHub intentionally. The
existing repository discretionary AI architecture is retained; **no
trade orders were submitted**.

## Next useful test

Before any deployment, train a policy evaluated through executable
Bid/Ask prices, realistic spread/commission/slippage, stop/target handling,
and account-level FTMO drawdown gates. Independent forward-market data
will be needed for a truly new final evaluation.
