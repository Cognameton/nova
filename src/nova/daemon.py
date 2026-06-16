"""Nova daemon — Phase 19 Stage 19.2/19.3.

Runs the NovaRuntime as an always-on background process. Exposes a Unix
domain socket for attach/detach CLI connections. Fires model_self_state_tick()
autonomously on a configurable interval. Handles SIGTERM for clean GPU release.

Protocol: newline-delimited JSON over a Unix socket.
  client → {"type": "chat"|"tick"|"status"|"shutdown", ...}
  server → {"type": ..., ...}
"""

from __future__ import annotations

import json
import signal
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Status record
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DaemonStatus:
    running: bool = False
    session_id: str = ""
    started_at: str = ""
    uptime_seconds: float = 0.0
    tick_count: int = 0
    tick_interval_seconds: int = 300
    last_tick_at: str = ""
    attached_clients: int = 0
    socket_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# NovaDaemon
# ---------------------------------------------------------------------------

class NovaDaemon:
    """Always-on Nova runtime daemon with Unix socket attach interface.

    One model lock serializes all model access (inference is not thread-safe).
    The tick loop runs in a background thread; each client connection runs in
    its own thread. All model calls go through _model_lock.
    """

    def __init__(
        self,
        *,
        runtime,
        socket_path: Path,
        tick_interval_seconds: int = 300,
        session_id: str = "nova-daemon",
    ) -> None:
        self.runtime = runtime
        self.socket_path = Path(socket_path)
        self.tick_interval_seconds = tick_interval_seconds
        self.session_id = session_id

        self._model_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._tick_thread: threading.Thread | None = None
        self._server_socket: socket.socket | None = None
        self._started_at: str = ""
        self._tick_count: int = 0
        self._last_tick_at: str = ""
        self._client_count: int = 0
        self._client_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the daemon: runtime, tick loop, socket server (blocking)."""
        self._started_at = datetime.now(timezone.utc).isoformat()
        self.runtime.start(session_id=self.session_id)

        # Signal handlers can only be registered from the main thread.
        try:
            signal.signal(signal.SIGTERM, self._on_sigterm)
            signal.signal(signal.SIGINT, self._on_sigterm)
        except ValueError:
            pass

        self._tick_thread = threading.Thread(
            target=self._tick_loop, name="nova-tick", daemon=True
        )
        self._tick_thread.start()

        self._serve()

    def shutdown(self) -> None:
        """Signal the daemon to stop and close the socket."""
        self._stop_event.set()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass
        try:
            self.runtime.close()
        except Exception:
            pass

    def status(self) -> DaemonStatus:
        uptime = 0.0
        if self._started_at:
            try:
                started = datetime.fromisoformat(self._started_at)
                uptime = (datetime.now(timezone.utc) - started).total_seconds()
            except ValueError:
                pass
        with self._client_lock:
            clients = self._client_count
        return DaemonStatus(
            running=not self._stop_event.is_set(),
            session_id=self.session_id,
            started_at=self._started_at,
            uptime_seconds=round(uptime, 1),
            tick_count=self._tick_count,
            tick_interval_seconds=self.tick_interval_seconds,
            last_tick_at=self._last_tick_at,
            attached_clients=clients,
            socket_path=str(self.socket_path),
        )

    # ------------------------------------------------------------------
    # Tick loop (background thread)
    # ------------------------------------------------------------------

    def _tick_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.tick_interval_seconds)
            if self._stop_event.is_set():
                break
            with self._model_lock:
                try:
                    self.runtime.model_self_state_tick(trigger="daemon_tick")
                    self._tick_count += 1
                    self._last_tick_at = datetime.now(timezone.utc).isoformat()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Socket server (main thread, blocking)
    # ------------------------------------------------------------------

    def _serve(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(str(self.socket_path))
        self._server_socket.listen(8)
        self._server_socket.settimeout(1.0)

        while not self._stop_event.is_set():
            try:
                conn, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                daemon=True,
            )
            thread.start()

        self.shutdown()

    def _handle_connection(self, conn: socket.socket) -> None:
        with self._client_lock:
            self._client_count += 1
        try:
            buf = b""
            with conn:
                conn.settimeout(120.0)
                while not self._stop_event.is_set():
                    try:
                        chunk = conn.recv(4096)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            _send(conn, {"type": "error", "message": "invalid json"})
                            continue
                        response = self._dispatch(msg)
                        _send(conn, response)
                        if msg.get("type") == "shutdown":
                            return
        finally:
            with self._client_lock:
                self._client_count -= 1

    def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any]:
        msg_type = str(msg.get("type", ""))

        if msg_type == "status":
            return {"type": "status", **self.status().to_dict()}

        if msg_type == "shutdown":
            self._stop_event.set()
            return {"type": "shutdown", "ok": True}

        if msg_type == "tick":
            with self._model_lock:
                try:
                    tick = self.runtime.model_self_state_tick(trigger="manual_tick")
                    self._tick_count += 1
                    self._last_tick_at = datetime.now(timezone.utc).isoformat()
                    return {
                        "type": "tick",
                        "tick_id": getattr(tick, "tick_id", ""),
                        "tool_name": _extract_tool_name(tick),
                    }
                except Exception as exc:
                    return {"type": "error", "message": str(exc)}

        if msg_type == "chat":
            prompt = str(msg.get("prompt", "")).strip()
            if not prompt:
                return {"type": "error", "message": "prompt required"}
            with self._model_lock:
                try:
                    turn = self.runtime.respond(prompt)
                    return {
                        "type": "chat",
                        "answer": turn.final_answer,
                        "turn_id": getattr(turn, "turn_id", ""),
                        "session_id": self.session_id,
                    }
                except Exception as exc:
                    return {"type": "error", "message": str(exc)}

        return {"type": "error", "message": f"unknown message type: {msg_type!r}"}

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_sigterm(self, signum, frame) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Attach client
# ---------------------------------------------------------------------------

class NovaAttachClient:
    """Connect to a running NovaDaemon and provide an interactive REPL."""

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = Path(socket_path)

    def run_repl(self, *, session_label: str = "nova") -> None:
        """Connect and run interactive prompt loop until EOF or !detach."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(self.socket_path))
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            print(f"[nova] cannot connect to daemon at {self.socket_path}: {exc}")
            return

        # Print status on connect
        status = self.send(sock, {"type": "status"})
        if status:
            print(
                f"[nova] attached — session: {status.get('session_id', '?')} "
                f"ticks: {status.get('tick_count', 0)} "
                f"uptime: {status.get('uptime_seconds', 0):.0f}s"
            )

        print(f"[nova] type your prompt. Ctrl+D or '!detach' to detach.\n")

        try:
            import readline  # noqa: F401 — enables line editing on Linux
        except ImportError:
            pass

        buf = b""
        while True:
            try:
                prompt = input(f"{session_label}> ")
            except EOFError:
                print()
                break

            if prompt.strip() == "!detach":
                break
            if prompt.strip() == "!status":
                s = self.send(sock, {"type": "status"})
                if s:
                    print(json.dumps(s, indent=2))
                continue
            if prompt.strip() == "!tick":
                t = self.send(sock, {"type": "tick"})
                if t:
                    print(f"[tick] {t}")
                continue
            if not prompt.strip():
                continue

            response = self.send(sock, {"type": "chat", "prompt": prompt})
            if response is None:
                print("[nova] connection lost")
                break
            if response.get("type") == "error":
                print(f"[error] {response.get('message', '?')}")
            else:
                print(f"\n{response.get('answer', '')}\n")

        try:
            sock.close()
        except OSError:
            pass
        print("[nova] detached")

    def send_status(self) -> DaemonStatus | None:
        """Return daemon status without entering the REPL."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(self.socket_path))
            data = self.send(sock, {"type": "status"})
            sock.close()
            if data is None:
                return None
            return DaemonStatus(
                running=bool(data.get("running", False)),
                session_id=str(data.get("session_id", "")),
                started_at=str(data.get("started_at", "")),
                uptime_seconds=float(data.get("uptime_seconds", 0.0)),
                tick_count=int(data.get("tick_count", 0)),
                tick_interval_seconds=int(data.get("tick_interval_seconds", 300)),
                last_tick_at=str(data.get("last_tick_at", "")),
                attached_clients=int(data.get("attached_clients", 0)),
                socket_path=str(data.get("socket_path", "")),
            )
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            return None

    def send_shutdown(self) -> bool:
        """Send shutdown signal to the daemon. Returns True if acknowledged."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(self.socket_path))
            data = self.send(sock, {"type": "shutdown"})
            sock.close()
            return bool(data and data.get("ok"))
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            return False

    @staticmethod
    def send(
        sock: socket.socket, msg: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Send one JSON message and return the parsed response."""
        try:
            sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
            buf = b""
            sock.settimeout(120.0)
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    line, _ = buf.split(b"\n", 1)
                    return json.loads(line.decode("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_socket_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "nova.sock"


def _send(conn: socket.socket, msg: dict[str, Any]) -> None:
    try:
        conn.sendall((json.dumps(msg) + "\n").encode("utf-8"))
    except OSError:
        pass


def _extract_tool_name(tick) -> str:
    """Best-effort extraction of tool name from an OperationalTickRecord."""
    try:
        audit = getattr(tick, "adapter_audit", None) or {}
        if isinstance(audit, dict):
            return str(audit.get("tool_requested", "") or "")
    except Exception:
        pass
    return ""
