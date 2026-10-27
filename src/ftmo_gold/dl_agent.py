"""Optional Deep Learning evidence passed to a free discretionary agent.

The agent still receives every strategy as an OPTIONAL reference.
Predictions are never converted directly into orders or forced signals.
"""
from dataclasses import dataclass
from .decision import FreeDecisionAgent, Decision


@dataclass
class DeepLearningDecisionAgent:
    """Research-only bridge between temporal model scores and an AI agent."""
    model: object | None = None

    def decide(self, observation: dict, dl_prediction: dict, *, position_open=False) -> Decision:
        if not isinstance(observation, dict) or not isinstance(dl_prediction, dict):
            raise ValueError("Market observation and DL evidence must be dictionaries")
        if dl_prediction.get("asof_utc") != observation.get("asof_utc"):
            raise ValueError("Deep-learning inference time does not match observation")
        if not dl_prediction.get("status", "").startswith("RESEARCH_ONLY"):
            raise ValueError("Unknown prediction provenance")
        scores = dl_prediction.get("probabilities")
        if not isinstance(scores, dict) or not scores:
            raise ValueError("Missing neural network scores")
        if any(not isinstance(v, (int, float)) or isinstance(v, bool)
               or not (0 <= v <= 1) for v in scores.values()):
            raise ValueError("Invalid neural network scores")
        if abs(sum(scores.values()) - 1) > 0.01:
            raise ValueError("Probabilities do not sum to one")
        enriched = dict(observation)
        enriched["deep_learning_evidence"] = {
            "probabilities": dict(scores),
            "model_timeframes": dl_prediction.get("model_timeframes", []),
            "warning": "Model is uncalibrated and may have no predictive edge; "
                       "do not treat scores as expected trade profits."
        }
        return FreeDecisionAgent(self.model).decide(enriched, position_open=position_open)
