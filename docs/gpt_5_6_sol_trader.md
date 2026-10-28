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

The demo paper scanner considers one decision at most per newly completed M15
bar and defaults to 20 calls per rolling five-hour window. The local budget is
conservative: a failed attempt still consumes a slot because it may already
have reached the remote service. When the local budget is exhausted the scanner
records `local_quota_guard_wait` and makes no request.

## Current safety status

- Historical proposal mode: implemented.
- MT5 demo paper-proposal mode: implemented.
- Live or funded-account orders: not implemented.
- Automated position sizing and executable bid/ask revalidation: required
  before any later execution phase.
- Independent backtest and forward paper results: not yet measured.

No claim of profitability follows from selecting a more capable model. The
paper journal must be evaluated after spread, slippage and FTMO risk constraints
before the architecture can progress beyond proposal-only mode.
