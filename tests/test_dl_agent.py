import pytest
from ftmo_gold.dl_agent import DeepLearningDecisionAgent

OBS={"asof_utc":12345,"timeframes":{"M5":{"last_close":3000.0}}}
DL={"asof_utc":12345,"status":"RESEARCH_ONLY_NOT_CALIBRATED",
    "probabilities":{"down":0.4,"neutral":0.1,"up":0.5},
    "model_timeframes":["m5","m15","h1","h4","d1"]}


class FreeModel:
    def decide(self, observation, references):
        assert "deep_learning_evidence" in observation
        assert not any(reference["mandatory"] for reference in references)
        return {"action":"BUY","reason":"Independent evidence and freely chosen scenario",
                "stop":2990.0,"target":3020.0,
                "strategy_reference":"custom_multi_timeframe_hypothesis"}


def test_agent_can_make_non_playbook_decision():
    d=DeepLearningDecisionAgent(FreeModel()).decide(OBS,DL)
    assert d.action=="BUY"
    assert d.strategy_reference=="custom_multi_timeframe_hypothesis"


def test_agent_abstains_if_no_ai_model():
    assert DeepLearningDecisionAgent().decide(OBS,DL).action=="WAIT"


def test_never_combine_mismatched_times():
    with pytest.raises(ValueError,match="time"):
        DeepLearningDecisionAgent().decide(OBS,{**DL,"asof_utc":999})


def test_invalid_scores_rejected():
    with pytest.raises(ValueError):
        DeepLearningDecisionAgent().decide(OBS,{**DL,"probabilities":{"up":float("nan")}})
