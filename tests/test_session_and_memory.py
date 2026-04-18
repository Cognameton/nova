from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.memory.episodic import JsonlEpisodicMemoryStore
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


if __name__ == "__main__":
    unittest.main()
