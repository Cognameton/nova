"""Tests for Phase 19 Stage 19.2/19.3 — Nova Daemon Harness."""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from nova.daemon import (
    DaemonStatus,
    NovaDaemon,
    NovaAttachClient,
    _extract_tool_name,
    _send,
    default_socket_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_runtime(session_id: str = "test") -> MagicMock:
    rt = MagicMock()
    rt.config.app.data_dir = "./data"
    turn = MagicMock()
    turn.final_answer = "Hello from Nova."
    turn.turn_id = "t1"
    rt.respond.return_value = turn
    tick = MagicMock()
    tick.tick_id = "tick1"
    tick.adapter_audit = {"tool_requested": "recall_self"}
    rt.model_self_state_tick.return_value = tick
    rt.close.return_value = None
    rt.start.return_value = None
    return rt


def _make_daemon(tmpdir: str, tick_interval: int = 3600) -> NovaDaemon:
    return NovaDaemon(
        runtime=_mock_runtime(),
        socket_path=Path(tmpdir) / "nova.sock",
        tick_interval_seconds=tick_interval,
        session_id="test-daemon",
    )


def _start_daemon_background(daemon: NovaDaemon) -> threading.Thread:
    """Start daemon.start() in a background thread; return thread."""
    t = threading.Thread(target=daemon.start, daemon=True)
    t.start()
    # Give the socket server a moment to bind.
    time.sleep(0.1)
    return t


# ---------------------------------------------------------------------------
# DaemonStatus
# ---------------------------------------------------------------------------

class DaemonStatusTests(unittest.TestCase):
    def test_defaults(self):
        s = DaemonStatus()
        self.assertFalse(s.running)
        self.assertEqual(s.tick_count, 0)
        self.assertEqual(s.tick_interval_seconds, 300)
        self.assertEqual(s.session_id, "")

    def test_to_dict_contains_all_fields(self):
        s = DaemonStatus(running=True, session_id="x", tick_count=3)
        d = s.to_dict()
        self.assertTrue(d["running"])
        self.assertEqual(d["session_id"], "x")
        self.assertEqual(d["tick_count"], 3)
        self.assertIn("uptime_seconds", d)
        self.assertIn("last_tick_at", d)
        self.assertIn("attached_clients", d)
        self.assertIn("socket_path", d)

    def test_to_dict_serializable(self):
        s = DaemonStatus(running=True, session_id="s", started_at="2026-06-10T00:00:00+00:00")
        json.dumps(s.to_dict())  # should not raise


# ---------------------------------------------------------------------------
# default_socket_path
# ---------------------------------------------------------------------------

class DefaultSocketPathTests(unittest.TestCase):
    def test_returns_path_object(self):
        p = default_socket_path("./data")
        self.assertIsInstance(p, Path)

    def test_filename_is_nova_sock(self):
        p = default_socket_path("/tmp/nova_data")
        self.assertEqual(p.name, "nova.sock")

    def test_parent_matches_data_dir(self):
        p = default_socket_path("/tmp/custom_data")
        self.assertEqual(str(p.parent), "/tmp/custom_data")


# ---------------------------------------------------------------------------
# _extract_tool_name
# ---------------------------------------------------------------------------

class ExtractToolNameTests(unittest.TestCase):
    def test_returns_tool_name_from_adapter_audit(self):
        tick = MagicMock()
        tick.adapter_audit = {"tool_requested": "reflect"}
        self.assertEqual(_extract_tool_name(tick), "reflect")

    def test_returns_empty_when_no_adapter_audit(self):
        tick = MagicMock()
        del tick.adapter_audit
        self.assertEqual(_extract_tool_name(tick), "")

    def test_returns_empty_when_audit_is_none(self):
        tick = MagicMock()
        tick.adapter_audit = None
        self.assertEqual(_extract_tool_name(tick), "")

    def test_returns_empty_when_key_missing(self):
        tick = MagicMock()
        tick.adapter_audit = {}
        self.assertEqual(_extract_tool_name(tick), "")


# ---------------------------------------------------------------------------
# NovaDaemon — unit tests (no live socket)
# ---------------------------------------------------------------------------

class NovaDaemonInitTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.daemon = _make_daemon(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_session_id_stored(self):
        self.assertEqual(self.daemon.session_id, "test-daemon")

    def test_tick_interval_stored(self):
        self.assertEqual(self.daemon.tick_interval_seconds, 3600)

    def test_socket_path_is_path(self):
        self.assertIsInstance(self.daemon.socket_path, Path)

    def test_stop_event_not_set_initially(self):
        self.assertFalse(self.daemon._stop_event.is_set())

    def test_tick_count_starts_at_zero(self):
        self.assertEqual(self.daemon._tick_count, 0)

    def test_client_count_starts_at_zero(self):
        self.assertEqual(self.daemon._client_count, 0)


class NovaDaemonStatusTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.daemon = _make_daemon(self._tmpdir.name)
        self.daemon._started_at = "2026-06-10T00:00:00+00:00"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_status_running_true_when_not_stopped(self):
        s = self.daemon.status()
        self.assertTrue(s.running)

    def test_status_running_false_after_stop(self):
        self.daemon._stop_event.set()
        s = self.daemon.status()
        self.assertFalse(s.running)

    def test_status_session_id(self):
        s = self.daemon.status()
        self.assertEqual(s.session_id, "test-daemon")

    def test_status_started_at(self):
        s = self.daemon.status()
        self.assertEqual(s.started_at, "2026-06-10T00:00:00+00:00")

    def test_status_uptime_is_positive(self):
        s = self.daemon.status()
        self.assertGreater(s.uptime_seconds, 0)

    def test_status_socket_path(self):
        s = self.daemon.status()
        self.assertIn("nova.sock", s.socket_path)

    def test_status_tick_count(self):
        self.daemon._tick_count = 7
        s = self.daemon.status()
        self.assertEqual(s.tick_count, 7)


class NovaDaemonDispatchTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.daemon = _make_daemon(self._tmpdir.name)
        self.daemon._started_at = "2026-06-10T00:00:00+00:00"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_dispatch_status(self):
        resp = self.daemon._dispatch({"type": "status"})
        self.assertEqual(resp["type"], "status")
        self.assertIn("running", resp)
        self.assertIn("tick_count", resp)

    def test_dispatch_shutdown(self):
        resp = self.daemon._dispatch({"type": "shutdown"})
        self.assertEqual(resp["type"], "shutdown")
        self.assertTrue(resp["ok"])
        self.assertTrue(self.daemon._stop_event.is_set())

    def test_dispatch_tick(self):
        resp = self.daemon._dispatch({"type": "tick"})
        self.assertEqual(resp["type"], "tick")
        self.assertEqual(resp["tool_name"], "recall_self")
        self.assertEqual(self.daemon._tick_count, 1)

    def test_dispatch_tick_increments_count(self):
        self.daemon._dispatch({"type": "tick"})
        self.daemon._dispatch({"type": "tick"})
        self.assertEqual(self.daemon._tick_count, 2)

    def test_dispatch_chat(self):
        resp = self.daemon._dispatch({"type": "chat", "prompt": "Hello?"})
        self.assertEqual(resp["type"], "chat")
        self.assertEqual(resp["answer"], "Hello from Nova.")
        self.assertEqual(resp["session_id"], "test-daemon")

    def test_dispatch_chat_empty_prompt(self):
        resp = self.daemon._dispatch({"type": "chat", "prompt": ""})
        self.assertEqual(resp["type"], "error")
        self.assertIn("prompt required", resp["message"])

    def test_dispatch_unknown_type(self):
        resp = self.daemon._dispatch({"type": "unknown_xyz"})
        self.assertEqual(resp["type"], "error")
        self.assertIn("unknown_xyz", resp["message"])

    def test_dispatch_tick_error_returns_error_response(self):
        self.daemon.runtime.model_self_state_tick.side_effect = RuntimeError("model error")
        resp = self.daemon._dispatch({"type": "tick"})
        self.assertEqual(resp["type"], "error")
        self.assertIn("model error", resp["message"])

    def test_dispatch_chat_error_returns_error_response(self):
        self.daemon.runtime.respond.side_effect = RuntimeError("inference error")
        resp = self.daemon._dispatch({"type": "chat", "prompt": "Hello"})
        self.assertEqual(resp["type"], "error")
        self.assertIn("inference error", resp["message"])


# ---------------------------------------------------------------------------
# Live socket integration tests
# ---------------------------------------------------------------------------

class NovaDaemonSocketTests(unittest.TestCase):
    """Full socket roundtrip tests using a real Unix domain socket."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.daemon = _make_daemon(self._tmpdir.name, tick_interval=3600)
        self.daemon._started_at = "2026-06-10T00:00:00+00:00"
        self._thread = _start_daemon_background(self.daemon)

    def tearDown(self):
        self.daemon._stop_event.set()
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(self.daemon.socket_path))
            sock.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)
        self._tmpdir.cleanup()

    def _connect(self) -> socket.socket:
        for _ in range(20):
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(str(self.daemon.socket_path))
                return sock
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.05)
        self.fail("Could not connect to daemon socket")

    def _roundtrip(self, msg: dict) -> dict:
        sock = self._connect()
        try:
            response = NovaAttachClient.send(sock, msg)
        finally:
            sock.close()
        return response

    def test_status_response(self):
        resp = self._roundtrip({"type": "status"})
        self.assertEqual(resp["type"], "status")
        self.assertTrue(resp["running"])
        self.assertEqual(resp["session_id"], "test-daemon")

    def test_chat_response(self):
        resp = self._roundtrip({"type": "chat", "prompt": "Hello?"})
        self.assertEqual(resp["type"], "chat")
        self.assertEqual(resp["answer"], "Hello from Nova.")

    def test_tick_response(self):
        resp = self._roundtrip({"type": "tick"})
        self.assertEqual(resp["type"], "tick")
        self.assertEqual(resp["tool_name"], "recall_self")

    def test_invalid_json_returns_error(self):
        sock = self._connect()
        try:
            sock.sendall(b"not json\n")
            buf = b""
            sock.settimeout(3.0)
            while b"\n" not in buf:
                buf += sock.recv(256)
            line = buf.split(b"\n")[0]
            resp = json.loads(line.decode())
        finally:
            sock.close()
        self.assertEqual(resp["type"], "error")
        self.assertIn("invalid json", resp["message"])

    def test_shutdown_stops_daemon(self):
        resp = self._roundtrip({"type": "shutdown"})
        self.assertEqual(resp["type"], "shutdown")
        self.assertTrue(resp["ok"])
        time.sleep(0.2)
        self.assertTrue(self.daemon._stop_event.is_set())

    def test_socket_file_created(self):
        self.assertTrue(self.daemon.socket_path.exists())

    def test_multiple_sequential_connections(self):
        for _ in range(3):
            resp = self._roundtrip({"type": "status"})
            self.assertEqual(resp["type"], "status")


# ---------------------------------------------------------------------------
# NovaAttachClient — unit tests (mock socket)
# ---------------------------------------------------------------------------

class NovaAttachClientInitTests(unittest.TestCase):
    def test_stores_socket_path(self):
        client = NovaAttachClient(Path("/tmp/nova.sock"))
        self.assertEqual(client.socket_path, Path("/tmp/nova.sock"))

    def test_str_path_converted_to_path(self):
        client = NovaAttachClient("/tmp/nova.sock")
        self.assertIsInstance(client.socket_path, Path)


class NovaAttachClientSendStatusTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.daemon = _make_daemon(self._tmpdir.name, tick_interval=3600)
        self.daemon._started_at = "2026-06-10T00:00:00+00:00"
        self._thread = _start_daemon_background(self.daemon)

    def tearDown(self):
        self.daemon._stop_event.set()
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(self.daemon.socket_path))
            sock.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)
        self._tmpdir.cleanup()

    def _wait_for_socket(self):
        for _ in range(20):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(str(self.daemon.socket_path))
                s.close()
                return
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.05)

    def test_send_status_returns_daemon_status(self):
        self._wait_for_socket()
        client = NovaAttachClient(self.daemon.socket_path)
        status = client.send_status()
        self.assertIsNotNone(status)
        self.assertIsInstance(status, DaemonStatus)
        self.assertTrue(status.running)

    def test_send_status_when_not_running(self):
        client = NovaAttachClient(Path("/tmp/nonexistent_nova_xyz.sock"))
        status = client.send_status()
        self.assertIsNone(status)

    def test_send_shutdown_acknowledged(self):
        self._wait_for_socket()
        client = NovaAttachClient(self.daemon.socket_path)
        ok = client.send_shutdown()
        self.assertTrue(ok)
        time.sleep(0.2)
        self.assertTrue(self.daemon._stop_event.is_set())

    def test_send_shutdown_when_not_running(self):
        client = NovaAttachClient(Path("/tmp/nonexistent_nova_xyz.sock"))
        ok = client.send_shutdown()
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
