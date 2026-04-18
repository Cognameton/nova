"""CLI entrypoint for Nova 2.0."""

from __future__ import annotations

import argparse
from pathlib import Path

from nova.config import DEFAULT_CONFIG_PATH, load_config
from nova.eval.probes import BasicProbeRunner
from nova.inference.llama_cpp_backend import LlamaCppBackend
from nova.logging.traces import JsonlTraceLogger
from nova.memory.autobiographical import JsonlAutobiographicalMemoryStore
from nova.memory.engram import JsonEngramMemoryStore
from nova.memory.episodic import JsonlEpisodicMemoryStore
from nova.memory.graph import SqliteGraphMemoryStore
from nova.memory.retrieval import BasicMemoryEventFactory, BasicMemoryRouter
from nova.persona.store import JsonPersonaStore, JsonSelfStateStore
from nova.prompt.composer import NovaPromptComposer
from nova.prompt.retry import BasicRetryPolicy
from nova.prompt.validator import NovaOutputValidator
from nova.runtime import NovaRuntime
from nova.session import JsonlSessionStore


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_runtime(*, config_override: str | None = None) -> NovaRuntime:
    config = load_config(
        default_path=REPO_ROOT / DEFAULT_CONFIG_PATH,
        override_path=config_override,
    )

    data_dir = _resolve_path(config.app.data_dir)
    log_dir = _resolve_path(config.app.log_dir)
    sessions_dir = data_dir / "sessions"
    memory_dir = data_dir / "memory"
    traces_dir = log_dir / "traces"
    probes_path = log_dir / "probes.jsonl"

    persona_store = JsonPersonaStore(data_dir / "persona_state.json")
    self_state_store = JsonSelfStateStore(data_dir / "self_state.json")
    session_store = JsonlSessionStore(sessions_dir)
    trace_logger = JsonlTraceLogger(traces_dir, probes_path=probes_path)

    episodic_store = JsonlEpisodicMemoryStore(memory_dir / "episodic.jsonl")
    engram_store = JsonEngramMemoryStore(
        memory_dir / "engram.json",
        enabled=config.memory.engram_enabled,
    )
    graph_store = SqliteGraphMemoryStore(memory_dir / "graph.db")
    autobiographical_store = JsonlAutobiographicalMemoryStore(
        memory_dir / "autobiographical.jsonl"
    )
    memory_router = BasicMemoryRouter(
        episodic=episodic_store if config.memory.episodic_enabled else None,
        engram=engram_store if config.memory.engram_enabled else None,
        graph=graph_store if config.memory.graph_enabled else None,
        autobiographical=autobiographical_store if config.memory.autobiographical_enabled else None,
    )

    backend = LlamaCppBackend(config)
    composer = NovaPromptComposer(token_counter=backend.tokenize)
    validator = NovaOutputValidator(config.contract)
    retry_policy = BasicRetryPolicy()
    event_factory = BasicMemoryEventFactory()
    probe_runner = BasicProbeRunner() if config.eval.enable_probes else None

    return NovaRuntime(
        config=config,
        backend=backend,
        composer=composer,
        validator=validator,
        retry_policy=retry_policy,
        persona_store=persona_store,
        self_state_store=self_state_store,
        session_store=session_store,
        trace_logger=trace_logger,
        memory_router=memory_router,
        memory_event_factory=event_factory,
        probe_runner=probe_runner,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nova 2.0 text runtime")
    parser.add_argument(
        "--config",
        dest="config_override",
        help="Optional path to a YAML config override file.",
    )
    parser.add_argument(
        "--session-id",
        help="Resume or continue a specific session id.",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Force creation of a fresh session even if --session-id is provided.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra runtime details after each reply.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    runtime = build_runtime(config_override=args.config_override)
    session_id = None if args.new_session else args.session_id
    started_session = runtime.start(session_id=session_id)

    print("Nova 2.0")
    print(f"Session: {started_session}")
    print("Type 'exit' or 'quit' to stop.")

    try:
        while True:
            try:
                user_text = input("You: ").strip()
            except EOFError:
                print()
                break

            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                break

            turn = runtime.respond(user_text)
            print(f"Nova: {turn.final_answer}")
            if args.debug:
                print(
                    f"[debug] model={turn.model_id} prompt_tokens~={turn.prompt_token_estimate} "
                    f"completion_tokens={turn.completion_token_estimate} retries={turn.retry_count}"
                )
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
