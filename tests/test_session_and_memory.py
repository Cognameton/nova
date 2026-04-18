from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.memory.graph import SqliteGraphMemoryStore
from nova.memory.episodic import JsonlEpisodicMemoryStore
from nova.memory.retrieval import BasicMemoryEventFactory
from nova.memory.semantic import JsonlSemanticMemoryStore
from nova.persona.state import PersonaState, SelfState
from nova.session import JsonlSessionStore
from nova.types import MemoryEvent, TurnRecord, ValidationResult


class SessionAndMemoryTests(unittest.TestCase):
    def test_session_store_round_trips_recent_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlSessionStore(tmpdir)
            session_id = store.start_session(session_id="abc123")
            turn = TurnRecord(
                session_id=session_id,
                turn_id="turn-1",
                timestamp="2026-04-18T00:00:00Z",
                user_text="Hello",
                final_answer="Hi",
                raw_answer="Hi",
                validation=ValidationResult(valid=True),
                model_id="fake-model",
            )
            store.append_turn(turn)

            turns = store.recent_turns(session_id=session_id, limit=5)
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0].user_text, "Hello")
            self.assertEqual(turns[0].final_answer, "Hi")

    def test_episodic_memory_returns_matching_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlEpisodicMemoryStore(Path(tmpdir) / "episodic.jsonl")
            store.add(
                MemoryEvent(
                    event_id="e1",
                    timestamp="2026-04-18T00:00:00Z",
                    session_id="s1",
                    turn_id="t1",
                    channel="episodic",
                    kind="note",
                    text="Nova values continuity and clarity.",
                    source="nova",
                )
            )
            hits = store.search("continuity", top_k=3)
            self.assertEqual(len(hits), 1)
            self.assertIn("continuity", hits[0].text.lower())

    def test_memory_event_factory_emits_policy_driven_graph_and_autobiographical_events(self) -> None:
        factory = BasicMemoryEventFactory()
        events = factory.from_turn(
            session_id="s1",
            turn_id="t1",
            user_text="I prefer local inference for Nova.",
            final_answer="My name is Nova. I remain focused on continuity.",
            persona=PersonaState(name="Nova"),
            self_state=SelfState(identity_summary="Nova is continuity-focused."),
        )

        channels = [event.channel for event in events]
        self.assertIn("episodic", channels)
        self.assertIn("engram", channels)
        self.assertIn("graph", channels)
        self.assertIn("autobiographical", channels)

        graph_events = [event for event in events if event.channel == "graph"]
        self.assertTrue(any(event.kind == "preference_fact" for event in graph_events))
        self.assertTrue(any(event.kind == "identity_fact" for event in graph_events))
        self.assertTrue(
            any(event.continuity_weight > 0.0 and event.retention == "active" for event in events)
        )

    def test_semantic_memory_returns_summary_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlSemanticMemoryStore(Path(tmpdir) / "semantic.jsonl")
            store.add(
                MemoryEvent(
                    event_id="s1",
                    timestamp="2026-04-18T00:00:00Z",
                    session_id="session",
                    turn_id="turn",
                    channel="semantic",
                    kind="theme_summary",
                    text="The user consistently prioritizes continuity over feature sprawl.",
                    summary="User prioritizes continuity over feature sprawl.",
                    importance=0.8,
                    confidence=0.9,
                    continuity_weight=0.8,
                    source="reflection",
                    metadata={"theme": "continuity-priority"},
                )
            )
            hits = store.search("continuity", top_k=3)
            self.assertEqual(len(hits), 1)
            self.assertIn("continuity", hits[0].text.lower())
            self.assertEqual(hits[0].channel, "semantic")

    def test_graph_memory_prefers_active_continuity_weighted_fact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SqliteGraphMemoryStore(Path(tmpdir) / "graph.db")
            store.add(
                MemoryEvent(
                    event_id="f1",
                    timestamp="2026-04-18T00:00:00Z",
                    session_id="s1",
                    turn_id="t1",
                    channel="graph",
                    kind="identity_fact",
                    text="Nova maintains continuity.",
                    importance=0.8,
                    confidence=0.9,
                    continuity_weight=1.0,
                    source="nova",
                    metadata={
                        "fact_id": "nova-maintains-continuity",
                        "fact_domain": "identity",
                        "subject_type": "self",
                        "subject_key": "nova",
                        "subject_name": "Nova",
                        "relation": "maintains",
                        "object_type": "concept",
                        "object_key": "continuity",
                        "object_name": "continuity",
                        "weight": 0.8,
                        "confidence": 0.9,
                        "continuity_weight": 1.0,
                        "active": True,
                        "evidence_text": "Nova maintains continuity.",
                    },
                )
            )
            store.add(
                MemoryEvent(
                    event_id="f2",
                    timestamp="2026-04-17T00:00:00Z",
                    session_id="s1",
                    turn_id="t0",
                    channel="graph",
                    kind="identity_fact",
                    text="Nova used to maintain continuity.",
                    importance=0.5,
                    confidence=0.6,
                    continuity_weight=0.4,
                    source="nova",
                    metadata={
                        "fact_id": "nova-old-continuity",
                        "fact_domain": "identity",
                        "subject_type": "self",
                        "subject_key": "nova",
                        "subject_name": "Nova",
                        "relation": "maintains",
                        "object_type": "concept",
                        "object_key": "continuity",
                        "object_name": "continuity",
                        "weight": 0.5,
                        "confidence": 0.6,
                        "continuity_weight": 0.4,
                        "active": False,
                        "evidence_text": "Nova used to maintain continuity.",
                    },
                )
            )

            hits = store.search("continuity", top_k=5)
            self.assertGreaterEqual(len(hits), 2)
            self.assertEqual(hits[0].metadata.get("active"), True)
            self.assertIn("active", hits[0].tags)
            self.assertIn("historical", hits[1].tags)


if __name__ == "__main__":
    unittest.main()
