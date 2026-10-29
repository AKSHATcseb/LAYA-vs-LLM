import unittest
from sqam_analyzer.layers.layer2_decision import Layer2DecisionReasoner
from sqam_analyzer.layers.layer4_decision import Layer4DecisionValidator
from sqam_analyzer.models import (
    Layer3Output, QualityAttribute, Requirement, Shortcoming, SRSContext, ValidationVerdict,
)


class FakeBackend:
    def __init__(self, p): self.p = p
    def prob_true(self, state, question): return self.p


def _fixtures():
    req = Requirement(req_id="R1", text="The system shall be fast.")
    ctx = SRSContext(domain_summary="demo", surrounding_requirements=[])
    d = Shortcoming(defect_id="D1", attribute=list(QualityAttribute)[0],
                    problematic_excerpt="fast", explanation="vague")
    l3 = Layer3Output(req_id="R1", original_text=req.text,
                      refined_text="The system shall respond within 2 seconds.",
                      changes_made=["quantified"], addressed_defect_ids=["D1"],
                      functional_intent_preservation_rationale="same intent")
    return req, ctx, d, l3


class TestLayaLayers(unittest.TestCase):
    def test_l2_keeps_when_high(self):
        req, ctx, d, _ = _fixtures()
        out = Layer2DecisionReasoner(FakeBackend(0.9)).execute(req, ctx, [d])
        self.assertEqual(len(out.retained_defects), 1)

    def test_l2_discards_when_low(self):
        req, ctx, d, _ = _fixtures()
        out = Layer2DecisionReasoner(FakeBackend(0.1)).execute(req, ctx, [d])
        self.assertEqual(len(out.discarded_defects), 1)

    def test_l2_no_candidates(self):
        req, ctx, _, _ = _fixtures()
        out = Layer2DecisionReasoner(FakeBackend(0.9)).execute(req, ctx, [])
        self.assertEqual(out.evaluations, [])

    def test_l4_validates_when_no_problems(self):
        req, ctx, d, l3 = _fixtures()
        out = Layer4DecisionValidator(FakeBackend(0.1)).execute(req, ctx, [d], l3)
        self.assertEqual(out.verdict, ValidationVerdict.VALIDATED)

    def test_l4_rejects_when_any_problem(self):
        req, ctx, d, l3 = _fixtures()
        out = Layer4DecisionValidator(FakeBackend(0.9)).execute(req, ctx, [d], l3)
        self.assertFalse(out.is_valid)


if __name__ == "__main__":
    unittest.main()