"""Optional local OpenAI adapter for research-only structured trading decisions.

Uses Chat Completions via stdlib HTTPS. Requires explicit API key in environment,
local outbound network access and a model supporting JSON response format.
Not called by default, and NEVER executes orders.
"""
import json
import os
import urllib.request

SYSTEM = """You are a research-only analyst of XAUUSD, not an order executor.
Use ONLY the supplied timestamped closed-bar features and reference descriptions.
The reference playbooks are optional ideas, NOT mandatory strategies.
You may propose a novel thesis or combine ideas. Prefer WAIT when evidence is weak.
Never claim statistical profitability from this one observation.
Return ONE JSON object with action BUY/SELL/WAIT, reason (>=8 characters),
confidence number 0..1 or null, thesis, strategy_reference (optional), and for
BUY/SELL only, positive numeric stop and target. Other action types are reserved
for position management and cannot be used in this stateless endpoint.
No external data, no made-up prices, no future candles."""

class OpenAIResearchModel:
    def __init__(self,model="gpt-4.1-mini",timeout=20):
        self.model=model
        self.timeout=timeout

    def decide(self,observation,references):
        key=os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing")
        payload={
            "model":self.model,
            "temperature":0,
            "response_format":{"type":"json_object"},
            "messages":[{"role":"system","content":SYSTEM},
                        {"role":"user","content":json.dumps({"observation":observation,
                                                            "reference_playbooks":references},
                                                           separators=(",",":"))}],
        }
        req=urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},
            method="POST")
        with urllib.request.urlopen(req,timeout=self.timeout) as response:
            raw=json.load(response)
        return json.loads(raw["choices"][0]["message"]["content"])
