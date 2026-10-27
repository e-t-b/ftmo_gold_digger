# Gold Digger — First 24-hour Deep Reinforcement Learning run

Date: 2026-10-27. **Research-only, no broker order execution.**

## Implemented and run
A PyTorch five-GRU actor-critic was trained with a PPO-style clipped
objective on **random 24-hour XAUUSD historical days**, strictly sampled
from the **training period**. The agent observes historical, CLOSED
M5/M15/H1/H4/D1 price representations and current position/equity.
Each step chooses WAIT, BUY, SELL or CLOSE. Simulated execution occurs
on the next M5 bar. Rewards are based on simulated equity changes,
including changes in open-position value.

One position at a time, 0.01 lot (one ounce), 3 entries per day;
SL from ATR, TP=1.5x distance; bid OHLC and MT5 bar spread with
assumed adverse slippage. The model **does not yet control** SL,
TP, or position size, and strategy playbooks are not wired into its
policy. This PPO network is not yet the discretionary LLM agent.

## Actual smoke-test run
- Market coverage: user FTMO-Demo 2023–2026; 559 days available for
  training, 62 for validation, 148 for testing.
- **12 randomly sampled training episodes**, one PPO update epoch;
  this is a proof of execution only, NOT massive training.
- Validation: 8 sampled days, 15 closed trades, **-$104.70**
  simulated net P&L, 26.67% win rate.
- Test: 15 sampled days, 45 closed trades, **-$11.70**
  simulated net P&L, 40% win rate.
- Three focused local tests passed: policy tensor shapes;
  closed candle context; WAIT-only simulator stability.

Artifacts (exact executable trainer `gold_rl.py`, `test_gold_rl.py`,
`ppo.pt`, report JSON and per-day validation/test CSVs) were created
in the conversation as `gold_digger_deep_rl_episodes_v0.zip`.

## Known limitations
This is **not** a realistic FTMO Challenge simulation:
per-episode equity resets; no Prague-midnight balance rules, fees,
swaps, margin, or historical tick-level fills; OHLC stop-first and
bar spreads approximate outcomes. It is not an agent that can decide
arbitrary stops, targets, sizes or invent and evaluate strategies.
No statistical evidence of an edge has emerged. The test date range
was analyzed in preceding experiments and therefore is not untouched.

**Next engineering task:** strengthen the trading environment, add
separate conditional position-sizing/stop/target actions and strategy
tools, evaluate on new forward dates and scale training episodes only
once simulator invariants pass.
