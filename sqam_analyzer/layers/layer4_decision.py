from __future__ import annotations
import json
from typing import List
from sqam_analyzer.layers.layer4_validator import Layer4SolutionValidator
from sqam_analyzer.models import (
    Layer3Output, Layer4Output, Requirement, Shortcoming, SRSContext,
    ValidationCheck, ValidationVerdict,
)

# (criterion, question, "fail_if_true")
CHECKS = [
    ("Defect Resolution", "Is at least one of the listed defects still present in the refined requirement?", True),
    ("Intent Preservation", "Does the refined requirement change the functional intent of the original?", True),
    ("New Ambiguity", "Does the refined requirement introduce new ambiguity?", True),
    ("Unwarranted Assumptions", "Does the refined requirement add assumptions not supported by the original or the context?", True),
]


class Layer4DecisionValidator(Layer4SolutionValidator):
    """Layer 4 judged by LAYA instead of an LLM."""

    def __init__(self, backend, threshold: float = 0.5, max_neighbours: int = 5):
        super().__init__(llm_client=None)
        self.backend = backend
        self.threshold = threshold
        self.max_neighbours = max_neighbours

    def execute(self, requirement: Requirement, srs_context: SRSContext,
                retained_defects: List[Shortcoming], layer3_output: Layer3Output,
                temperature: float = 0.0) -> Layer4Output:
        # Same early exit as the LLM layer
        if not retained_defects and layer3_output.refined_text.strip() == requirement.text.strip():
            return super().execute(requirement, srs_context, retained_defects,
                                   layer3_output, temperature)

        defects = "; ".join(f"{d.attribute.value}: '{d.problematic_excerpt}'" for d in retained_defects)
        state = (
            f"Domain: {srs_context.domain_summary}\n"
            f"Nearby requirements: {' | '.join(srs_context.surrounding_requirements[: self.max_neighbours])}\n"
            f"Original: {requirement.text}\n"
            f"Defects: {defects or 'none'}\n"
            f"Refined: {layer3_output.refined_text}"
        )
        checks, assumptions, ambiguities = [], [], []
        for name, question, fail_if_true in CHECKS:
            p = self.backend.prob_true(state, question)
            passed = (p < self.threshold) if fail_if_true else (p >= self.threshold)
            checks.append(ValidationCheck(
                criterion=name, passed=passed,
                observations=f"LAYA P(problem) = {p:.3f} (threshold {self.threshold}).",
            ))
            if not passed and name == "New Ambiguity":
                ambiguities.append("LAYA flagged new ambiguity")
            if not passed and name == "Unwarranted Assumptions":
                assumptions.append("LAYA flagged unwarranted assumption")

        valid = all(c.passed for c in checks)
        return Layer4Output(
            req_id=requirement.req_id, is_valid=valid,
            verdict=ValidationVerdict.VALIDATED if valid else ValidationVerdict.REJECTED,
            checks=checks,
            unwarranted_assumptions_detected=assumptions,
            new_ambiguities_detected=ambiguities,
            critique_and_feedback="Decided by LAYA; see per-check probabilities.",
        )