# Gold Digger Deep Learning — real first training, 2026-10-27

## Objective and scope
Implemented and **actually trained** PyTorch GRU on owner-supplied
FTMO-Demo XAUUSD M5/M15/H1 historical candles, approximately 90 days.
No model-generated trades were executed or backtested. This is *not* a
validated profitable strategy or autonomous MT5 execution system.

## Architecture
- Three separate GRU encoders, 24 hidden units each
- M5: 32 most recent completed bars; M15: 16; H1: 6
- Seven features per timeframe: log return, normalized candle range,
  candle body, upper/lower wick, log(1+tick_volume), spread in points
  divided by price/point. Exact FTMO point size used: 0.01 USD.
- Temporal alignment by candle **close time**, strictly historical
- No gaps inside any sequence, no forward filling
- Train-only mean/std, clipped standardized values [-12,12]
- Three-class target: XAUUSD 30-minute close change <-$0.82,
  within +/-$0.82, or >+$0.82
- $0.82 is a **hypothetical** estimated round-trip price cost and is
  NOT measured historical P&L after all costs
- Chronological 65/15/20 time-based split and ~6.5h embargo
- AdamW lr=0.001, batch=128, seed=42, 6 epochs, best by validation cross-entropy
- CPU training; no trading orders / no API AI inference

## Actual first run metrics
12,427 valid sequences.

| Split | N | Accuracy | Majority baseline | Cross entropy |
|---|---:|---:|---:|---:|
| Train | 8,246 | 47.72% | 46.24% | 0.93448 |
| Validation | 1,843 | 41.94% | 45.90% | 0.96120 |
| Held-out test | 2,211 | 40.30% | 48.71% | 0.98244 |

On test: class counts = [1077 DOWN, 271 NEUTRAL, 863 UP].
Confusion matrix true DOWN/NEUTRAL/UP rows by predicted columns:
[[398,0,679],[129,0,142],[370,0,493]].
Model makes NO neutral predictions, an important model-quality failure.

**Decision:** V1 model is **NOT approved** for the FTMO trader.
In this sample, it underperforms a majority-label baseline on
unseen data. Strong distribution drift is possible; data and objectives
are limited. Six epochs is an initial engineering training, not a
completed architecture search. Subsequent experiments must use fresh
holdout data to avoid test-set overfitting.

## Reproduction
```powershell
git pull
pip install -e ".[dl,dev]"
python -m ftmo_gold.deep_learning --input data/raw --out reports/gru_v1 --epochs 6
```

Output: `model.pt` (checkpoint) and `report.json` in ignored local
`reports/gru_v1/`. The first locally trained checkpoint is provided
separately from GitHub, which contains source/tests/report only.

## Remaining work
- Check label definition and realistic directional edge vs spread/commission
- Multi-period walk-forward and multiple seeds with untouched evaluation
- Causal decision-time inference feeding the discretionary agent
- Historical agent decision replay with SL/TP, fills, FTMO violations and costs
- No live or challenge activation before independent validation
