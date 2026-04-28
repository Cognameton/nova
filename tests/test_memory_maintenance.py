from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from nova.memory.autobiographical import JsonlAutobiographicalMemoryStore
from nova.memory.consolidation import SemanticConsolidator
from nova.memory.engram import JsonEngramMemoryStore
from nova.memory.episodic import JsonlEpisodicMemoryStore
from nova.memory.graph import SqliteGraphMemoryStore
from nova.memory.maintenance import MemoryMaintenancePlanner, MemoryMaintenanceRunner
from nova.memory.reflection import ReflectionEngine
from nova.memory.semantic import JsonlSemanticMemoryStore
from nova.types import MemoryEvent


class MemoryMaintenanceTests(unittest.TestCase):
    def test_prunes_stale_low_signal_engram(self) -> None:
        planner = MemoryMaintenancePlanner()
        event = MemoryEvent(
            event_id="e1",
            timestamp="2026-03-01T00:00:00Z",
            session_id="s1",
            turn_id="t1",
            channel="engram",
            kind="assistant_message",
            text="generic phrase",
            importance=0.2,
            confidence=0.9,
            continuity_weight=0.1,
            source="nova",
            metadata={"hit_count": 0},
        )

        decision = planner.assess_event(event)
        self.assertEqual(decision.action, "prune")
        self.assertEqual(decision.target_retention, "pruned")

    def test_demotes_promoted_episodic_event(self) -> None:
        planner = MemoryMaintenancePlanner()
        event = MemoryEvent(
            event_id="e2",
            timestamp="2026-04-01T00:00:00Z",
            session_id="s1",
            turn_id="t2",
            channel="episodic",
            kind="assistant_message",
            text="A routine project update.",
            importance=0.3,
            confidence=1.0,
            continuity_weight=0.2,
            source="nova",
            metadata={"promoted": True},
        )

        decision = planner.assess_event(event)
        self.assertEqual(decision.action, "demote")
        self.assertEqual(decision.target_retention, "demoted")

    def test_consolidates_redundant_autobiographical_memory(self) -> None:
        planner = MemoryMaintenancePlanner()
        event = MemoryEvent(
            event_id="e3",
            timestamp="2026-04-10T00:00:00Z",
            session_id="s1",
            turn_id="t3",
            channel="autobiographical",
            kind="identity_note",
            text="Nova remains focused on continuity.",
            importance=0.9,
            confidence=0.9,
            continuity_weight=1.0,
            source="nova",
            metadata={"redundancy": 0.9, "self_state_version": 2},
        )

        decision = planner.assess_event(event)
        self.assertEqual(decision.action, "consolidate")
        self.assertIsNotNone(decision.consolidation_key)

    def test_archives_historical_graph_fact(self) -> None:
        planner = MemoryMaintenancePlanner()
        event = MemoryEvent(
            event_id="e4",
            timestamp="2026-03-15T00:00:00Z",
            session_id="s1",
            turn_id="t4",
            channel="graph",
            kind="preference_fact",
            text="User preferred an older setup.",
            importance=0.5,
            confidence=0.7,
            continuity_weight=0.4,
            source="user",
            metadata={"active": False, "fact_id": "user-old-pref"},
        )

        decision = planner.assess_event(event)
        self.assertEqual(decision.action, "archive")
        self.assertEqual(decision.target_retention, "archived")

    def test_runner_collects_events_and_summarizes_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            episodic = JsonlEpisodicMemoryStore(base / "episodic.jsonl")
            engram = JsonEngramMemoryStore(base / "engram.json")
            graph = SqliteGraphMemoryStore(base / "graph.db")
            autobiographical = JsonlAutobiographicalMemoryStore(base / "autobiographical.jsonl")

            episodic.add(
                MemoryEvent(
                    event_id="ep1",
                    timestamp="2026-04-01T00:00:00Z",
                    session_id="s1",
                    turn_id="t1",
                    channel="episodic",
                    kind="assistant_message",
                    text="Routine update",
                    importance=0.2,
                    confidence=1.0,
                    continuity_weight=0.1,
                    source="nova",
                    metadata={"promoted": True},
                )
            )
            engram.add(
                MemoryEvent(
                    event_id="en1",
                    timestamp="2026-03-01T00:00:00Z",
                    session_id="s1",
                    turn_id="t2",
                    channel="engram",
                    kind="assistant_message",
                    text="generic phrase",
                    importance=0.2,
                    confidence=0.9,
                    continuity_weight=0.1,
                    source="nova",
                )
            )
            graph.add(
                MemoryEvent(
                    event_id="gr1",
                    timestamp="2026-03-15T00:00:00Z",
                    session_id="s1",
                    turn_id="t3",
                    channel="graph",
                    kind="preference_fact",
                    text="Historical preference",
                    importance=0.5,
                    confidence=0.7,
                    continuity_weight=0.4,
                    source="user",
                    metadata={
                        "fact_id": "user-old-pref",
                        "fact_domain": "preference",
                        "subject_type": "user",
                        "subject_key": "user",
                        "relation": "prefers",
                        "object_type": "preference",
                        "object_key": "old-pref",
                        "weight": 0.5,
                        "confidence": 0.7,
                        "continuity_weight": 0.4,
                        "active": False,
                        "evidence_text": "Historical preference",
                    },
                )
            )
            autobiographical.add(
                MemoryEvent(
                    event_id="ab1",
                    timestamp="2026-04-10T00:00:00Z",
                    session_id="s1",
                    turn_id="t4",
                    channel="autobiographical",
                    kind="identity_note",
                    text="Nova remains focused on continuity.",
                    importance=0.9,
                    confidence=0.9,
                    continuity_weight=1.0,
                    source="nova",
                    metadata={"redundancy": 0.9, "self_state_version": 2},
                )
            )

            runner = MemoryMaintenanceRunner(
                episodic=episodic,
                engram=engram,
                graph=graph,
                autobiographical=autobiographical,
            )
            decisions = runner.build_plan()
            summary = runner.summarize_plan()

            self.assertEqual(len(decisions), 4)
            self.assertGreaterEqual(summary["by_action"].get("demote", 0), 1)
            self.assertGreaterEqual(summary["by_action"].get("archive", 0), 1)
            self.assertGreaterEqual(summary["by_action"].get("consolidate", 0), 1)

    def test_semantic_consolidator_builds_theme_summary_from_episodic_events(self) -> None:
        consolidator = SemanticConsolidator()
        events = [
            MemoryEvent(
                event_id="e1",
                timestamp="2026-04-18T00:00:00Z",
                session_id="s1",
                turn_id="t1",
                channel="episodic",
                kind="user_message",
                text="I prefer local inference for Nova.",
                tags=["user", "turn", "preference"],
                importance=0.75,
                confidence=1.0,
                continuity_weight=0.75,
                source="user",
            ),
            MemoryEvent(
                event_id="e2",
                timestamp="2026-04-18T00:01:00Z",
                session_id="s1",
                turn_id="t2",
                channel="episodic",
                kind="user_message",
                text="I want Nova to stay local-first.",
                tags=["user", "turn", "preference"],
                importance=0.8,
                confidence=1.0,
                continuity_weight=0.8,
                source="user",
            ),
        ]

        candidates = consolidator.build_candidates(events)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.channel, "semantic")
        self.assertEqual(candidate.kind, "theme_summary")
        self.assertIn("user-preferences", candidate.tags)
        self.assertEqual(candidate.source, "reflection")
        self.assertEqual(set(candidate.supersedes), {"e1", "e2"})

    def test_semantic_consolidator_ignores_generic_relationship_turns(self) -> None:
        consolidator = SemanticConsolidator()
        events = [
            MemoryEvent(
                event_id="e1",
                timestamp="2026-04-18T00:00:00Z",
                session_id="s1",
                turn_id="t1",
                channel="episodic",
                kind="assistant_message",
                text="I understand your preference for a local-first approach with direct answers. How can I assist you today?",
                tags=["assistant", "relationship", "turn"],
                importance=0.6,
                confidence=1.0,
                continuity_weight=0.7,
                source="nova",
            ),
            MemoryEvent(
                event_id="e2",
                timestamp="2026-04-18T00:01:00Z",
                session_id="s1",
                turn_id="t2",
                channel="episodic",
                kind="assistant_message",
                text="I'll prioritize stability and memory safety in model choices over novelty. How would you like me to proceed?",
                tags=["assistant", "relationship", "turn"],
                importance=0.6,
                confidence=1.0,
                continuity_weight=0.7,
                source="nova",
            ),
        ]

        candidates = consolidator.build_candidates(events)
        self.assertEqual(candidates, [])

    def test_runner_can_write_semantic_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            episodic = JsonlEpisodicMemoryStore(base / "episodic.jsonl")
            semantic = JsonlSemanticMemoryStore(base / "semantic.jsonl")

            episodic.add(
                MemoryEvent(
                    event_id="e1",
                    timestamp="2026-04-18T00:00:00Z",
                    session_id="s1",
                    turn_id="t1",
                    channel="episodic",
                    kind="user_message",
                    text="I prefer local inference for Nova.",
                    tags=["user", "turn", "preference"],
                    importance=0.75,
                    confidence=1.0,
                    continuity_weight=0.75,
                    source="user",
                )
            )
            episodic.add(
                MemoryEvent(
                    event_id="e2",
                    timestamp="2026-04-18T00:01:00Z",
                    session_id="s1",
                    turn_id="t2",
                    channel="episodic",
                    kind="user_message",
                    text="I want Nova to stay local-first.",
                    tags=["user", "turn", "preference"],
                    importance=0.8,
                    confidence=1.0,
                    continuity_weight=0.8,
                    source="user",
                )
            )

            runner = MemoryMaintenanceRunner(episodic=episodic, semantic=semantic)
            written = runner.write_semantic_candidates()

            self.assertEqual(len(written), 1)
            semantic_events = semantic.list_events()
            self.assertEqual(len(semantic_events), 1)
            self.assertEqual(semantic_events[0].channel, "semantic")

    def test_semantic_store_merges_repeated_reflection_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            episodic = JsonlEpisodicMemoryStore(base / "episodic.jsonl")
            semantic = JsonlSemanticMemoryStore(base / "semantic.jsonl")

            episodic.add(
                MemoryEvent(
                    event_id="e1",
                    timestamp="2026-04-18T00:00:00Z",
                    session_id="s1",
                    turn_id="t1",
                    channel="episodic",
                    kind="user_message",
                    text="I prefer local inference for Nova.",
                    tags=["user", "turn", "preference"],
                    importance=0.75,
                    confidence=1.0,
                    continuity_weight=0.75,
                    source="user",
                )
            )
            episodic.add(
                MemoryEvent(
                    event_id="e2",
                    timestamp="2026-04-18T00:01:00Z",
                    session_id="s1",
                    turn_id="t2",
                    channel="episodic",
                    kind="user_message",
                    text="I want Nova to stay local-first.",
                    tags=["user", "turn", "preference"],
                    importance=0.8,
                    confidence=1.0,
                    continuity_weight=0.8,
                    source="user",
                )
            )

            runner = MemoryMaintenanceRunner(episodic=episodic, semantic=semantic)
            first_written = runner.write_semantic_candidates()
            second_written = runner.write_semantic_candidates()

            self.assertEqual(len(first_written), 1)
            self.assertEqual(len(second_written), 1)
            semantic_events = semantic.list_events()
            self.assertEqual(len(semantic_events), 1)
            self.assertEqual(semantic_events[0].metadata.get("revision_count"), 1)
            self.assertEqual(semantic_events[0].metadata.get("theme"), "user-preferences")

    def test_reflection_engine_builds_autobiographical_candidates(self) -> None:
        engine = ReflectionEngine()
        episodic_events = [
            MemoryEvent(
                event_id="e1",
                timestamp="2026-04-18T00:00:00Z",
                session_id="s1",
                turn_id="t1",
                channel="episodic",
                kind="assistant_message",
                text="My name is Nova. I remain focused on continuity.",
                tags=["assistant", "turn", "identity"],
                importance=0.85,
                confidence=1.0,
                continuity_weight=0.95,
                source="nova",
            ),
            MemoryEvent(
                event_id="e2",
                timestamp="2026-04-18T00:01:00Z",
                session_id="s1",
                turn_id="t2",
                channel="episodic",
                kind="assistant_message",
                text="I keep continuity at the center of my self-model.",
                tags=["assistant", "turn", "identity", "value"],
                importance=0.9,
                confidence=1.0,
                continuity_weight=0.95,
                source="nova",
            ),
        ]

        candidates = engine.build_autobiographical_candidates(episodic_events=episodic_events)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.channel, "autobiographical")
        self.assertEqual(candidate.kind, "reflection_note")
        self.assertIn("identity-continuity", candidate.tags)
        self.assertEqual(set(candidate.supersedes), {"e1", "e2"})

    def test_runner_can_write_autobiographical_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            episodic = JsonlEpisodicMemoryStore(base / "episodic.jsonl")
            semantic = JsonlSemanticMemoryStore(base / "semantic.jsonl")
            autobiographical = JsonlAutobiographicalMemoryStore(base / "autobiographical.jsonl")

            episodic.add(
                MemoryEvent(
                    event_id="e1",
                    timestamp="2026-04-18T00:00:00Z",
                    session_id="s1",
                    turn_id="t1",
                    channel="episodic",
                    kind="assistant_message",
                    text="My name is Nova. I remain focused on continuity.",
                    tags=["assistant", "turn", "identity"],
                    importance=0.85,
                    confidence=1.0,
                    continuity_weight=0.95,
                    source="nova",
                )
            )
            episodic.add(
                MemoryEvent(
                    event_id="e2",
                    timestamp="2026-04-18T00:01:00Z",
                    session_id="s1",
                    turn_id="t2",
                    channel="episodic",
                    kind="assistant_message",
                    text="I keep continuity at the center of my self-model.",
                    tags=["assistant", "turn", "identity", "value"],
                    importance=0.9,
                    confidence=1.0,
                    continuity_weight=0.95,
                    source="nova",
                )
            )
            semantic.add(
                MemoryEvent(
                    event_id="s1",
                    timestamp="2026-04-18T00:02:00Z",
                    session_id="s1",
                    turn_id="t3",
                    channel="semantic",
                    kind="theme_summary",
                    text="Nova identity: continuity remains central.",
                    summary="Nova identity: continuity remains central.",
                    tags=["semantic", "summary", "nova-identity"],
                    importance=0.85,
                    confidence=0.9,
                    continuity_weight=0.9,
                    source="reflection",
                )
            )

            runner = MemoryMaintenanceRunner(
                episodic=episodic,
                semantic=semantic,
                autobiographical=autobiographical,
            )
            written = runner.write_autobiographical_candidates()
            self.assertEqual(len(written), 1)
            autobiographical_events = autobiographical.list_events()
            self.assertEqual(len(autobiographical_events), 1)
            self.assertEqual(autobiographical_events[0].channel, "autobiographical")

    def test_autobiographical_store_merges_repeated_reflection_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            episodic = JsonlEpisodicMemoryStore(base / "episodic.jsonl")
            semantic = JsonlSemanticMemoryStore(base / "semantic.jsonl")
            autobiographical = JsonlAutobiographicalMemoryStore(base / "autobiographical.jsonl")

            episodic.add(
                MemoryEvent(
                    event_id="e1",
                    timestamp="2026-04-18T00:00:00Z",
                    session_id="s1",
                    turn_id="t1",
                    channel="episodic",
                    kind="assistant_message",
                    text="My name is Nova. I remain focused on continuity.",
                    tags=["assistant", "turn", "identity"],
                    importance=0.85,
                    confidence=1.0,
                    continuity_weight=0.95,
                    source="nova",
                )
            )
            episodic.add(
                MemoryEvent(
                    event_id="e2",
                    timestamp="2026-04-18T00:01:00Z",
                    session_id="s1",
                    turn_id="t2",
                    channel="episodic",
                    kind="assistant_message",
                    text="I keep continuity at the center of my self-model.",
                    tags=["assistant", "turn", "identity", "value"],
                    importance=0.9,
                    confidence=1.0,
                    continuity_weight=0.95,
                    source="nova",
                )
            )
            semantic.add(
                MemoryEvent(
                    event_id="s1",
                    timestamp="2026-04-18T00:02:00Z",
                    session_id="s1",
                    turn_id="t3",
                    channel="semantic",
                    kind="theme_summary",
                    text="Nova identity: continuity remains central.",
                    summary="Nova identity: continuity remains central.",
                    tags=["semantic", "summary", "nova-identity"],
                    importance=0.85,
                    confidence=0.9,
                    continuity_weight=0.9,
                    source="reflection",
                    metadata={"theme": "nova-identity"},
                )
            )

            runner = MemoryMaintenanceRunner(
                episodic=episodic,
                semantic=semantic,
                autobiographical=autobiographical,
            )
            first_written = runner.write_autobiographical_candidates()
            second_written = runner.write_autobiographical_candidates()

            self.assertEqual(len(first_written), 1)
            self.assertEqual(len(second_written), 1)
            autobiographical_events = autobiographical.list_events()
            self.assertEqual(len(autobiographical_events), 1)
            self.assertEqual(autobiographical_events[0].metadata.get("revision_count"), 1)
            self.assertEqual(autobiographical_events[0].metadata.get("theme"), "identity-continuity")

    def test_runner_can_apply_retention_mutations_to_stores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            episodic = JsonlEpisodicMemoryStore(base / "episodic.jsonl")
            graph = SqliteGraphMemoryStore(base / "graph.db")

            episodic.add(
                MemoryEvent(
                    event_id="ep1",
                    timestamp="2026-04-01T00:00:00Z",
                    session_id="s1",
                    turn_id="t1",
                    channel="episodic",
                    kind="assistant_message",
                    text="Routine update",
                    importance=0.2,
                    confidence=1.0,
                    continuity_weight=0.1,
                    source="nova",
                    metadata={"promoted": True},
                )
            )
            graph.add(
                MemoryEvent(
                    event_id="gr1",
                    timestamp="2026-03-15T00:00:00Z",
                    session_id="s1",
                    turn_id="t3",
                    channel="graph",
                    kind="preference_fact",
                    text="Historical preference",
                    importance=0.5,
                    confidence=0.7,
                    continuity_weight=0.4,
                    source="user",
                    metadata={
                        "fact_id": "user-old-pref",
                        "fact_domain": "preference",
                        "subject_type": "user",
                        "subject_key": "user",
                        "relation": "prefers",
                        "object_type": "preference",
                        "object_key": "old-pref",
                        "weight": 0.5,
                        "confidence": 0.7,
                        "continuity_weight": 0.4,
                        "active": False,
                        "evidence_text": "Historical preference",
                    },
                )
            )

            runner = MemoryMaintenanceRunner(episodic=episodic, graph=graph)
            decisions = runner.build_plan()
            results = runner.apply_plan(decisions)

            self.assertGreaterEqual(results.get("episodic", 0), 1)
            self.assertGreaterEqual(results.get("graph", 0), 1)

            episodic_event = episodic.list_events()[0]
            graph_event = graph.list_events()[0]
            self.assertEqual(episodic_event.retention, "demoted")
            self.assertEqual(graph_event.retention, "archived")

            hits = episodic.search("routine", top_k=3)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].metadata.get("retention"), "demoted")


if __name__ == "__main__":
    unittest.main()
