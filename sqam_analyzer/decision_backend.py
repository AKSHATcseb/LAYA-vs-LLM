"""Thin wrapper around LAYA for structured yes/no decisions."""

from __future__ import annotations

import laya


class LayaBackend:
    def __init__(self, checkpoint: str = "convaiinnovations/laya"):
        self.agent = laya.load(checkpoint)
        self.calls = 0
        self.truncated_calls = 0

    def prob_true(self, state: str, question: str) -> float:
        """Ask a yes/no question and return P(true)."""

        out = self.agent.system_one(
            state,
            {
                "decision": {
                    "type": "noul",
                    "instructions": question,
                }
            },
        )

        self.calls += 1

        usage = out.get("usage", {})
        if usage.get("truncated"):
            self.truncated_calls += 1

        answers = out["answers"]
        answer = answers["decision"]

        probability = float(answer["noul"])

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Expected a probability between 0 and 1, "
                f"received {probability}"
            )

        return probability