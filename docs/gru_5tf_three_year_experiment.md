# Five-timeframe GRU: first training on full 2023–2026 FTMO history

Date: 2026-10-27. Real user-supplied FTMO-Demo XAUUSD exports, **not synthetic market data**.

## Source coverage
- M1: 1,050,544; M5: 210,139; M15: 70,804
- H1: 17,713; H4: 4,634; D1: 773
- 2023-10-30 to 2026-10-27; no out-of-order or invalid OHLC reported
- Market-closure gaps, and a few shorter gaps, exist and must not be forward-filled as observed candles
- 448,697 historical ticks are available only over a short period, NOT three years of historical Bid/Ask quotes

## Actual experiment
- PyTorch five-branch GRU, H=24, 48-neuron shared head, seed 42, 5 epochs
- Five closed-candle sequences, lengths M5=32/M15=16/H1=6/H4=12/D1=10
- Evaluate every third M5 bar to reduce highly overlapping samples
- Require intra-session continuity in M5/M15/H1, permit market closures in H4/D1
- Time-aligned using bar CLOSE times; no unclosed higher-TF candles
- Classify 30-minute future close movement: DOWN below -$0.82, NEUTRAL within +/-$0.82, UP above +$0.82. $0.82 is an assumed price threshold, not measured historical all-in trading cost.
- Chronological 65/15/20 calendar-time split with 7-day embargo between partitions
- Normalize using training partition statistics only, early-select checkpoint on validation CE
- This is predictive classification, **not simulated trading returns**.

## Measured results

| Split | Period | Samples | Accuracy | Majority-class baseline | Cross-entropy |
|---|---|---:|---:|---:|---:|
| Train | 2023-11-14 → 2025-10-08 | 32,019 | 43.03% | 38.49% | 1.04527 |
| Validation | 2025-10-22 → 2026-03-18 | 6,763 | 47.05% | 47.32% | 0.94018 |
| Held-out test | 2026-04-02 → 2026-10-27 | 9,597 | 44.10% | 46.38% | 0.96382 |

Held-out confusion matrix (rows true Down/Neutral/Up, columns predicted):
[[440,6,4005],[74,1,872],[404,4,3791]].

**Conclusion:** The first five-TF GRU remains below the test-set
majority-label baseline and overwhelmingly predicts UP. No verified edge,
no profitability claim, not approved for trading.

Saved artifact delivered separately in chat:
`gold_digger_gru_5tf_3years.zip` containing trained `model.pt`,
`report.json`, and exact standalone reproducible training script. Large
market CSVs and checkpoints are intentionally not committed to GitHub.

## Next engineering correction
Train and evaluate multiple *causal* architectures and trading-relevant
targets on development data, then freeze the complete policy and compare
against fixed references on a genuinely untouched time period.
Do not optimize repeatedly against this held-out test and still call it
unseen. Deep Learning feeds the free decision agent; named strategies
remain optional.
