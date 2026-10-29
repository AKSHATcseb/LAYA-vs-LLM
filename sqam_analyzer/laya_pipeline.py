"""Construct the hybrid LLM + LAYA requirement-quality pipeline."""

from sqam_analyzer.pipeline import RequirementQualityPipeline
from sqam_analyzer.layers.layer2_decision import Layer2DecisionReasoner
from sqam_analyzer.layers.layer4_decision import Layer4DecisionValidator


def build_laya_pipeline(
    llm_client,
    backend,
    threshold: float = 0.5,
):
    """
    Build a pipeline where:
    - Layer 1 uses the existing LLM.
    - Layer 2 uses LAYA.
    - Layer 3 uses the existing LLM.
    - Layer 4 uses LAYA.
    - Layer 5 retains the existing report-generation implementation.
    """

    return RequirementQualityPipeline(
        default_llm=llm_client,
        layer2_reasoner=Layer2DecisionReasoner(
            backend,
            threshold=threshold,
        ),
        layer4_validator=Layer4DecisionValidator(
            backend,
            threshold=threshold,
        ),
    )