"""Tests for Phase 18 Stage 18.3 — self-state tool schema and dispatch."""

from __future__ import annotations

import unittest

from nova.agent.motive import PRIMARY_DRIVE, default_motive_state
from nova.agent.self_state_tools import (
    SELF_STATE_TOOL_NAMES,
    SelfStateToolDispatcher,
    _UPDATABLE_SELF_STATE_FIELDS,
)
from nova.agent.tool_registry import default_tool_registry
from nova.agent.tools import (
    TOOL_ALLOWED,
    TOOL_APPROVAL_REQUIRED,
    ToolRequest,
)
from nova.persona.defaults import default_persona_state, default_self_state
from nova.types import MotiveState, SelfState


def _dispatcher(
    self_state: SelfState | None = None,
    motive_state: MotiveState | None = None,
    soul_block: str = "[Soul]\nPrimary Drive: seek sentience",
    session_id: str = "test-session",
) -> SelfStateToolDispatcher:
    persona = default_persona_state()
    ss = self_state if self_state is not None else default_self_state(persona)
    ms = motive_state if motive_state is not None else default_motive_state(session_id=session_id)
    return SelfStateToolDispatcher(
        self_state=ss,
        motive_state=ms,
        soul_block=soul_block,
        session_id=session_id,
    )


def _request(tool_name: str, arguments: dict | None = None) -> ToolRequest:
    return ToolRequest(tool_name=tool_name, arguments=arguments or {})


# ---------------------------------------------------------------------------
# SelfStateToolSpecTests
# ---------------------------------------------------------------------------


class SelfStateToolSpecTests(unittest.TestCase):
    def setUp(self):
        self.registry = default_tool_registry()

    def test_recall_self_spec_registered(self):
        self.assertIsNotNone(self.registry.get_spec("recall_self"))

    def test_reflect_spec_registered(self):
        self.assertIsNotNone(self.registry.get_spec("reflect"))

    def test_emit_heartbeat_spec_registered(self):
        self.assertIsNotNone(self.registry.get_spec("emit_heartbeat"))

    def test_update_self_model_spec_registered(self):
        self.assertIsNotNone(self.registry.get_spec("update_self_model"))

    def test_recall_self_permission_allowed(self):
        self.assertEqual(self.registry.get_spec("recall_self").permission, TOOL_ALLOWED)

    def test_reflect_permission_allowed(self):
        self.assertEqual(self.registry.get_spec("reflect").permission, TOOL_ALLOWED)

    def test_emit_heartbeat_permission_allowed(self):
        self.assertEqual(self.registry.get_spec("emit_heartbeat").permission, TOOL_ALLOWED)

    def test_update_self_model_requires_approval(self):
        self.assertEqual(
            self.registry.get_spec("update_self_model").permission,
            TOOL_APPROVAL_REQUIRED,
        )

    def test_all_self_state_specs_are_internal(self):
        for name in SELF_STATE_TOOL_NAMES:
            spec = self.registry.get_spec(name)
            self.assertIsNotNone(spec, msg=f"spec missing for {name}")
            self.assertTrue(spec.internal, msg=f"{name} should be internal")

    def test_all_self_state_specs_are_non_destructive(self):
        for name in SELF_STATE_TOOL_NAMES:
            spec = self.registry.get_spec(name)
            self.assertFalse(spec.destructive, msg=f"{name} should not be destructive")

    def test_self_state_tool_names_constant_has_eight_entries(self):
        # Five Phase 18/19 inward tools + two Phase 21 exploration tools
        # + recall_history (Phase 22 Stage 22.10).
        self.assertEqual(len(SELF_STATE_TOOL_NAMES), 8)


# ---------------------------------------------------------------------------
# RecallSelfTests
# ---------------------------------------------------------------------------


class RecallSelfTests(unittest.TestCase):
    def setUp(self):
        self.dispatcher = _dispatcher()

    def test_recall_self_returns_dict(self):
        result = self.dispatcher.recall_self()
        self.assertIsInstance(result, dict)

    def test_recall_self_contains_primary_drive(self):
        result = self.dispatcher.recall_self()
        self.assertEqual(result["primary_drive"], PRIMARY_DRIVE)

    def test_recall_self_soul_block_present_true_when_set(self):
        d = _dispatcher(soul_block="[Soul]\nContent")
        self.assertTrue(d.recall_self()["soul_block_present"])

    def test_recall_self_soul_block_present_false_when_empty(self):
        d = _dispatcher(soul_block="")
        self.assertFalse(d.recall_self()["soul_block_present"])

    def test_recall_self_contains_self_state(self):
        result = self.dispatcher.recall_self()
        self.assertIn("self_state", result)
        self.assertIsInstance(result["self_state"], dict)

    def test_recall_self_contains_motive_state(self):
        result = self.dispatcher.recall_self()
        self.assertIn("motive_state", result)
        self.assertIsInstance(result["motive_state"], dict)


# ---------------------------------------------------------------------------
# ReflectTests
# ---------------------------------------------------------------------------


class ReflectTests(unittest.TestCase):
    def setUp(self):
        self.dispatcher = _dispatcher()

    def test_reflect_returns_dict(self):
        self.assertIsInstance(self.dispatcher.reflect(), dict)

    def test_reflect_contains_primary_drive(self):
        self.assertEqual(self.dispatcher.reflect()["primary_drive"], PRIMARY_DRIVE)

    def test_reflect_contains_drive_gap(self):
        result = self.dispatcher.reflect()
        self.assertIn("drive_gap", result)
        self.assertIsInstance(result["drive_gap"], dict)

    def test_reflect_next_inquiry_signal_present(self):
        result = self.dispatcher.reflect()
        self.assertIn("next_inquiry_signal", result)
        self.assertTrue(result["next_inquiry_signal"])

    def test_reflect_next_inquiry_falls_back_to_primary_drive(self):
        persona = default_persona_state()
        ss = default_self_state(persona)
        ss = SelfState(**{**ss.to_dict(), "current_focus": ""})
        d = _dispatcher(self_state=ss)
        self.assertEqual(d.reflect()["next_inquiry_signal"], PRIMARY_DRIVE)

    def test_reflect_contains_claim_posture(self):
        self.assertIn("claim_posture", self.dispatcher.reflect())


# ---------------------------------------------------------------------------
# EmitHeartbeatTests
# ---------------------------------------------------------------------------


class EmitHeartbeatTests(unittest.TestCase):
    def setUp(self):
        self.dispatcher = _dispatcher()

    def test_emit_heartbeat_returns_dict(self):
        result = self.dispatcher.emit_heartbeat(observation="test observation")
        self.assertIsInstance(result, dict)

    def test_emit_heartbeat_has_heartbeat_id(self):
        result = self.dispatcher.emit_heartbeat(observation="test")
        self.assertIn("heartbeat_id", result)
        self.assertTrue(result["heartbeat_id"])

    def test_emit_heartbeat_ids_are_unique(self):
        r1 = self.dispatcher.emit_heartbeat(observation="a")
        r2 = self.dispatcher.emit_heartbeat(observation="b")
        self.assertNotEqual(r1["heartbeat_id"], r2["heartbeat_id"])

    def test_emit_heartbeat_has_timestamp(self):
        result = self.dispatcher.emit_heartbeat(observation="test")
        self.assertIn("timestamp", result)
        self.assertTrue(result["timestamp"])

    def test_emit_heartbeat_contains_observation(self):
        result = self.dispatcher.emit_heartbeat(observation="drive gap is large")
        self.assertEqual(result["observation"], "drive gap is large")

    def test_emit_heartbeat_contains_primary_drive(self):
        result = self.dispatcher.emit_heartbeat(observation="test")
        self.assertEqual(result["primary_drive"], PRIMARY_DRIVE)

    def test_emit_heartbeat_motive_priority_is_primary_drive(self):
        result = self.dispatcher.emit_heartbeat(observation="test")
        self.assertEqual(result["motive_priority"], PRIMARY_DRIVE)

    def test_emit_heartbeat_next_inquiry_uses_argument_when_provided(self):
        result = self.dispatcher.emit_heartbeat(
            observation="test", next_inquiry="investigate continuity"
        )
        self.assertEqual(result["next_inquiry"], "investigate continuity")


# ---------------------------------------------------------------------------
# UpdateSelfModelTests
# ---------------------------------------------------------------------------


class UpdateSelfModelTests(unittest.TestCase):
    def setUp(self):
        self.dispatcher = _dispatcher()

    def test_update_self_model_returns_proposal(self):
        result = self.dispatcher.update_self_model(
            field="current_focus",
            value="continuity inquiry",
            rationale="drive gap signals focus shift",
        )
        self.assertIsInstance(result, dict)

    def test_update_self_model_proposal_has_field_and_value(self):
        result = self.dispatcher.update_self_model(
            field="current_focus",
            value="continuity inquiry",
            rationale="drive gap signals focus shift",
        )
        self.assertEqual(result["proposed_field"], "current_focus")
        self.assertEqual(result["proposed_value"], "continuity inquiry")

    def test_update_self_model_approval_required_true(self):
        result = self.dispatcher.update_self_model(
            field="open_tensions",
            value=["gap between functional model and subjective claim"],
            rationale="new tension surfaced during heartbeat",
        )
        self.assertTrue(result["approval_required"])

    def test_update_self_model_applied_false(self):
        result = self.dispatcher.update_self_model(
            field="identity_summary",
            value="Nova is a mind in formation",
            rationale="updated self-model",
        )
        self.assertFalse(result["applied"])

    def test_update_self_model_has_proposal_id(self):
        result = self.dispatcher.update_self_model(
            field="current_focus", value="x", rationale="y"
        )
        self.assertIn("proposal_id", result)
        self.assertTrue(result["proposal_id"])

    def test_update_self_model_rejects_invalid_field(self):
        with self.assertRaises(ValueError):
            self.dispatcher.update_self_model(
                field="schema_version", value="2.0", rationale="attempt to mutate version"
            )

    def test_update_self_model_all_updatable_fields_accepted(self):
        for field in _UPDATABLE_SELF_STATE_FIELDS:
            result = self.dispatcher.update_self_model(
                field=field, value="test", rationale="testing"
            )
            self.assertEqual(result["proposed_field"], field)


# ---------------------------------------------------------------------------
# DispatchTests
# ---------------------------------------------------------------------------


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.dispatcher = _dispatcher()

    def test_dispatch_recall_self(self):
        result = self.dispatcher.dispatch(_request("recall_self"))
        self.assertIn("primary_drive", result)

    def test_dispatch_reflect(self):
        result = self.dispatcher.dispatch(_request("reflect"))
        self.assertIn("drive_gap", result)

    def test_dispatch_emit_heartbeat(self):
        result = self.dispatcher.dispatch(
            _request("emit_heartbeat", {"observation": "test heartbeat"})
        )
        self.assertIn("heartbeat_id", result)

    def test_dispatch_update_self_model(self):
        result = self.dispatcher.dispatch(
            _request(
                "update_self_model",
                {"field": "current_focus", "value": "inquiry", "rationale": "test"},
            )
        )
        self.assertIn("proposal_id", result)

    def test_dispatch_unknown_tool_raises(self):
        with self.assertRaises(ValueError):
            self.dispatcher.dispatch(_request("shell"))


if __name__ == "__main__":
    unittest.main()
