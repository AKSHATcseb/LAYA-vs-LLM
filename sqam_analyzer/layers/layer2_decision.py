from __future__ import annotations
from typing import List
from sqam_analyzer.layers.layer2_reasoner import Layer2DefectReasoner
from sqam_analyzer.models import (
    DefectEvaluation, DefectStatus, Layer2Output, Requirement, Shortcoming, SRSContext,
)


class Layer2DecisionReasoner(Layer2DefectReasoner):
    """Layer 2 judged by LAYA instead of an LLM."""

    def __init__(self, backend, threshold: float = 0.5, max_neighbours: int = 5):
        super().__init__(llm_client=None)
        self.backend = backend
        self.threshold = threshold
        self.max_neighbours = max_neighbours

    def _state(self, req: Requirement, ctx: SRSContext, d: Shortcoming) -> str:
        neigh = " | ".join(ctx.surrounding_requirements[: self.max_neighbours])
        return (
            f"Domain: {ctx.domain_summary}\n"
            f"Nearby requirements: {neigh}\n"
            f"Requirement: {req.text}\n"
            f"Flagged attribute: {d.attribute.value}\n"
            f"Flagged excerpt: {d.problematic_excerpt}\n"
            f"Claimed problem: {d.explanation}"
        )

    def execute(self, requirement: Requirement, srs_context: SRSContext,
                candidate_shortcomings: List[Shortcoming],
                temperature: float = 0.0) -> Layer2Output:
        if not candidate_shortcomings:
            return Layer2Output(
                req_id=requirement.req_id, evaluations=[], retained_defects=[],
                discarded_defects=[],
                reasoning_summary="No candidate shortcomings were flagged by Layer 1.",
            )
        evals, kept, dropped = [], [], []
        for d in candidate_shortcomings:
            p = self.backend.prob_true(
                self._state(requirement, srs_context, d),
                "Is the flagged excerpt a genuine defect in this context?",
            )
            justified = p >= self.threshold
            evals.append(DefectEvaluation(
                defect_id=d.defect_id, attribute=d.attribute,
                original_problematic_excerpt=d.problematic_excerpt,
                status=DefectStatus.JUSTIFIED if justified else DefectStatus.DISCARDED,
                context_reasoning=f"LAYA P(genuine defect) = {p:.3f} (threshold {self.threshold}).",
                confidence=max(p, 1 - p),
            ))
            (kept if justified else dropped).append(d)
        return Layer2Output(
            req_id=requirement.req_id, evaluations=evals,
            retained_defects=kept, discarded_defects=dropped,
            reasoning_summary=f"LAYA retained {len(kept)} and discarded {len(dropped)} candidate defects.",
        )