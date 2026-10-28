"""GPT-5.6 Sol decision adapter for historical and paper trading.

The model can propose a decision but has no broker credentials and cannot place
orders. Every proposal still passes through the local decision validator and,
in a future execution phase, the independent FTMO risk gate.
"""
from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable

from .chatgpt_auth import ChatGPTPlanCredentials


MODEL = "gpt-5.6-sol"
REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}

SYSTEM = """You are Gold Digger's XAUUSD decision model in historical or paper-trading mode.
Use ONLY the supplied closed-bar observations, technical indicators, account state,
position state and reference descriptions. Never invent a price, candle, news event,
fill or future observation. Decide independently; references are optional context.
Prefer WAIT when the evidence is weak, contradictory, stale, or the spread is poor.
For a new entry return BUY or SELL with a protective stop and target around the
observed M5 close. When a position is open, HOLD or CLOSE may be used; do not open
another position. This response is a proposal: deterministic local risk controls
retain final authority. Do not claim profitability from a single observation."""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["BUY", "SELL", "WAIT", "CLOSE", "HOLD"]},
        "reason": {"type": "string", "minLength": 8, "maxLength": 1500},
        "confidence": {"anyOf": [{"type": "number", "minimum": 0, "maximum": 1}, {"type": "null"}]},
        "stop": {"anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}]},
        "target": {"anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}]},
        "thesis": {"type": "string", "maxLength": 1000},
        "strategy_reference": {"anyOf": [{"type": "string", "maxLength": 120}, {"type": "null"}]},
    },
    "required": ["action", "reason", "confidence", "stop", "target", "thesis", "strategy_reference"],
    "additionalProperties": False,
}


def _extract_completed_text(response: dict) -> str:
    chunks = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "".join(chunks)


class OpenAISolDecisionModel:
    """Responses API client fixed to GPT-5.6 Sol.

    ChatGPT-plan OAuth is the default. An API key can be supplied explicitly for
    paid fallback, but the class never silently changes billing paths.
    """

    def __init__(
        self,
        *,
        reasoning_effort: str = "medium",
        timeout: int = 90,
        token_provider: Callable[[], str] | None = None,
        api_key: str | None = None,
        verify_model: bool = True,
    ):
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError("Unsupported GPT-5.6 Sol reasoning effort")
        if token_provider and api_key:
            raise ValueError("Choose ChatGPT plan usage or an API key, not both")
        self.model = MODEL
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.token_provider = token_provider or ChatGPTPlanCredentials().access_token
        self.api_key = api_key
        self.verify_model = verify_model
        self._model_verified = False

    @classmethod
    def paid_api(cls, **kwargs):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing; no paid fallback was configured")
        return cls(api_key=key, token_provider=None, **kwargs)

    def _token(self) -> str:
        return self.api_key or self.token_provider()

    def available_models(self) -> list[str]:
        request = urllib.request.Request(
            "https://api.openai.com/v1/models",
            headers={"Authorization": "Bearer " + self._token()},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = json.load(response)
        models = raw.get("models", raw.get("data", []))
        return [item.get("slug", item.get("id")) for item in models
                if item.get("visibility", "list") == "list" and item.get("slug", item.get("id"))]

    def _ensure_model(self) -> None:
        if self._model_verified or not self.verify_model:
            return
        if MODEL not in self.available_models():
            raise RuntimeError(f"{MODEL} is not available to the connected ChatGPT account")
        self._model_verified = True

    def _stream_response(self, payload: dict) -> str:
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self._token(),
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        deltas = []
        completed = None
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                kind = event.get("type")
                if kind == "response.output_text.delta":
                    deltas.append(event.get("delta", ""))
                elif kind == "response.completed":
                    completed = event.get("response", {})
                elif kind in {"response.failed", "response.incomplete"}:
                    response_data = event.get("response", {})
                    error = response_data.get("error") or {}
                    raise RuntimeError("OpenAI response failed: " + str(error.get("code", kind)))
        if completed is None:
            raise RuntimeError("OpenAI stream ended without response.completed")
        text = "".join(deltas) or _extract_completed_text(completed)
        if not text:
            raise RuntimeError("OpenAI returned no decision text")
        return text

    def decide(self, observation: dict, references: list[dict]) -> dict:
        self._ensure_model()
        payload = {
            "model": MODEL,
            "instructions": SYSTEM,
            "input": [{
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": json.dumps({
                        "observation": observation,
                        "reference_playbooks": references,
                    }, separators=(",", ":"), sort_keys=True),
                }],
            }],
            "reasoning": {"effort": self.reasoning_effort},
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "gold_digger_decision",
                    "strict": True,
                    "schema": DECISION_SCHEMA,
                },
            },
            "max_output_tokens": 1200,
            "store": False,
            "stream": True,
        }
        return json.loads(self._stream_response(payload))


# Backward-compatible name for existing callers; it now uses the agreed model.
OpenAIResearchModel = OpenAISolDecisionModel
