# Gold Digger: 8-action Deep RL experiment — Oct 27, 2026

A reproducible actual PyTorch/GRU PPO-style training experiment on the owner-provided 2023–2026 FTMO-Demo XAUUSD five-timeframe history. Full source, checkpoint, test suite, and output are packaged in `gold_digger_rl_8actions_v1.zip` delivered in conversation. Does NOT execute MT5 orders.

## More autonomous choices
Previous Deep RL prototype had WAIT/BUY/SELL/CLOSE with a single position size. This experiment has eight options: WAIT, BUY with low/medium/high risk budgets, SELL with low/medium/high risk budgets, or CLOSE. ATR-based stops and 1.5R targets remain environmental rules; the neural policy does NOT freely author entry/exit levels nor use the conversational LLM.

## Actually executed
- 70 training episodes requested, 69 successfully optimized; each is a randomly sampled 24h historical date from the training interval.
- 5 GRU encoders: M5/M15/H1/H4/D1, with historical close timestamps only.
- 8 validation days: 19 trades, 52.63% win rate, **+$332.68 simulated P&L**.
- 15 sampled test days: 44 trades, 29.55% win rate, **-$347.57 simulated P&L**.
- Local unit tests: 3/3 pass.
- Max daily entries 3; approx 0.01–0.10 lot with ATR-based risk limit.
- Conservative intrabar stop-first tie resolution, per-bar spread proxy and assumed $0.05/oz slippage.

## Honest conclusion
**NOT profitable on the tested sample. NOT ready for FTMO.**
This is a better action space and verified training loop, but only 69 episodes were trained. The run is NOT a long-duration convergence demonstration or an effective autonomous strategy. The broad testing period was already examined by earlier models, so it is not a fresh untouched holdout. Episodes reset capital daily; no exact FTMO Prague floating-equity protection, swaps, commissions, or tick-level fills are modeled. Further work should not present additional short smoke-tests as a business outcome.

Downloadable experiment archive contains trainer, unit tests, report JSON, validation/test daily CSV files, and trained `ppo.pt` artifact. Original broker CSVs are not committed to GitHub.
