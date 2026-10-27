# Deep Learning direction V2 — FTMO-Demo XAUUSD, 2026-10-27

## What was actually executed
Trained a five-branch PyTorch GRU on 2023-10-30 to 2026-10-27
FTMO-Demo XAUUSD from the uploaded M5/M15/H1/H4/D1 data.
This is an **experimental research network**, not a profitable trader.

The V1 target was three classes with +/-$0.82 fixed thresholds;
the V2 experiment instead estimates the binary direction of the
30-minute close-price movement. This comparison is **not a direct
head-to-head accuracy comparison**, because the label space changed.

- Five separately encoded historical price scales;
  32 M5 / 16 M15 / 6 H1 / 12 H4 / 10 D1 candles.
- 49,689 valid causal, market-gap-aware sequences, stride 3 M5 bars.
- Calendar chronological split (65/15/20) plus 7-day embargo.
- 32,019 train / 6,763 validation / 9,597 test examples.
- Training-only feature scaling, hidden width 32, 64-unit head,
  AdamW, seed 42, 8 epochs.
- Best checkpoint selected on **validation log loss**, not test.
- Source code and trained checkpoint delivered as downloadable ZIP in
  this chat (the ZIP includes exact reproducible training script and report).
  GitHub contains the research report and the compatible agent bridge;
  the checkpoint and original historical data are not committed.

## Measured results

| Split | Accuracy | Majority baseline | ROC AUC | Log loss | Share predicted UP |
|---|---:|---:|---:|---:|---:|
| Train | 52.29% | 51.94% | 0.5331 | 0.6929 | 96.77% |
| Validation | 52.37% | 52.24% | 0.5226 | 0.6923 | 87.59% |
| Held-out test | **49.40%** | **51.20%** | **0.5088** | **0.6965** | **84.93%** |

Test period: 2026-04-02 to 2026-10-27.

**Conclusion:** No verified directional advantage. Model remains
strongly biased toward predicting UP. The held-out test cannot be used
repeatedly for hyperparameter search without losing its independent
status. More training iterations alone do not demonstrate improvement.

## Additional code
- `src/ftmo_gold/dl_agent.py`: attaches provenance-checked
  Deep Learning probability estimates to the existing discretionary
  agent's observation. The agent may use any strategy or none.
- `tests/test_dl_agent.py`: checks non-playbook scenarios,
  invalid evidence and timestamp mismatch.
- These modules have no MT5 order submission capabilities.

## Next experiment with a tangible trading objective
Use a *training-only* market-adaptive label/policy with
cost-aware and trade-level evaluation (BUY/SELL/WAIT, entry
Bid/Ask and spread, stops, adverse excursion), with walk-forward
segments rather than repeatedly grading against this same test.
Then freeze model selection for a new untouched forward interval.
