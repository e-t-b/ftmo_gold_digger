# Gold Digger — GPT-5.6 Sol decision architecture

## Decision boundary

GPT-5.6 Sol is the only language model authorized by this implementation. It
receives closed M5, M15 and H1 observations plus causal indicators, the current
position and a bounded account context. It returns a structured proposal:
`BUY`, `SELL`, `WAIT`, `HOLD` or `CLOSE`, with confidence, thesis and price
levels where applicable.

The model has no MetaTrader credentials and no order-submission tool. Its
proposal is rejected locally when its action, stop, target, prices or position
state are inconsistent. A future execution adapter must also pass the
independent FTMO risk gate; model text can never override that gate.

## Subscription path

The default authentication path is OpenAI's Sign in with ChatGPT public-client
flow. It uses PKCE, a loopback callback, ID-token verification, the explicit
`chatgpt.tokens.use.direct` permission and rotating refresh tokens. Credentials
are stored outside the repository in the operating system's local application
data directory. The paid API-key path exists only behind the explicit
`--paid-api` command-line option; there is no automatic billing fallback.

The client requests the connected account's current model catalog and fails
closed if `gpt-5.6-sol` is unavailable. It never substitutes Astra, another GPT
model or PPO.

## Quota controls

The demo paper scanner considers a new decision only on a completed M15 bar.
The rolling budget also spaces requests evenly, so a five-call/five-hour setting
produces about one Sol analysis per hour instead of consuming the allowance at
the start of the window. A failed attempt still consumes a slot because it may
already have reached the remote service. When guarded, the scanner records
`local_quota_guard_wait` and makes no request.

The persistent virtual portfolio can be configured for 2% nominal risk per
entry and at most two entries per Prague day. It sizes from executable bid/ask,
contract size, volume step, proposed stop and commission assumption. Only
targets between 1R and 4R are accepted. Closed M15 bars trigger conservative
SL/TP handling (SL first if both occur), while Sol can HOLD or CLOSE an open
virtual position. Balance, equity, P&L, win rate, profit factor and drawdown are
journaled across restarts. Server-time offsets are normalized to UTC.

## Current safety status

- Historical proposal mode: implemented.
- MT5 demo virtual-fill paper mode: implemented.
- Live or funded-account orders: not implemented.
- Virtual position sizing and executable bid/ask revalidation: implemented.
- Slippage and exact swap/commission reconciliation: still required before any
  later execution phase.
- Independent backtest and forward paper results: not yet measured.

No claim of profitability follows from selecting a more capable model. The
paper journal must be evaluated after spread, slippage and FTMO risk constraints
before the architecture can progress beyond proposal-only mode.
