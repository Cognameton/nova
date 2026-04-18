from __future__ import annotations

import unittest

from nova.memory.policy import IdentityFirstRetrievalPolicy
from nova.persona.state import SelfState
from nova.types import RetrievalHit


class RetrievalPolicyTests(unittest.TestCase):
    def test_identity_query_expands_autobiographical_and_graph_budget(self) -> None:
        policy = IdentityFirstRetrievalPolicy()
        plan = policy.plan(
            query="Who are you and how do you maintain continuity?",
            self_state=SelfState(current_focus="continuity and coherence"),
        )

        self.assertGreaterEqual(plan.top_k_by_channel["autobiographical"], 4)
        self.assertGreaterEqual(plan.top_k_by_channel["graph"], 5)
        self.assertGreaterEqual(plan.top_k_by_channel["semantic"], 4)

    def test_rerank_hits_prefers_identity_bearing_channels(self) -> None:
        policy = IdentityFirstRetrievalPolicy()
        hits = [
            RetrievalHit(
                channel="episodic",
                text="Routine episode",
                score=1.4,
                metadata={"continuity_weight": 0.0, "importance": 0.2},
            ),
            RetrievalHit(
                channel="autobiographical",
                text="Nova remains focused on continuity.",
                score=1.1,
                metadata={"continuity_weight": 1.0, "importance": 0.9, "active": True},
            ),
            RetrievalHit(
                channel="graph",
                text="Nova -> maintains -> continuity",
                score=1.1,
                metadata={"continuity_weight": 1.0, "importance": 0.9, "active": True},
            ),
        ]

        ranked = policy.rerank_hits(hits)
        self.assertEqual(ranked[0].channel, "autobiographical")
        self.assertEqual(ranked[1].channel, "graph")


if __name__ == "__main__":
    unittest.main()
