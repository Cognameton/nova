"""Configuration loading for Nova 2.0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/nova.default.yaml")
VALID_BACKENDS = {"llama_cpp"}
VALID_PROMPT_ABLATION_MODES = {"current", "minimal", "state_summary", "action_boundary"}
VALID_TICK_HEARTBEAT_SAMPLING = {"recent", "stratified"}
VALID_REVERSI_OPPONENTS = {"random", "greedy"}


@dataclass(slots=True)
class AppConfig:
    name: str = "Nova"
    data_dir: str = "./data"
    log_dir: str = "./data/logs"


@dataclass(slots=True)
class ModelConfig:
    backend: str = "llama_cpp"
    model_path: str = ""
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    tensor_split: list[float] = field(default_factory=lambda: [0.5, 0.5])
    main_gpu: int = 0
    chat_format: str = ""
    system_prefix: str = ""


@dataclass(slots=True)
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    # Finding F14 — see GenerationRequest. Default 1.0 (off) keeps every
    # existing config and the whole suite on pre-2026-09-01 behaviour; the
    # live config opts in explicitly so the change is legible in one place.
    repeat_penalty: float = 1.0
    repeat_last_n: int = 64
    stop: list[str] = field(default_factory=lambda: ["User:", "\nUser:"])
    retries: int = 2
    # Phase 22 Stage 22.6 part 2 — experimental, default off. Reintroduces
    # genuine model deliberation on respond() only (never the tick loop);
    # see docs/plans/PHASE22_STAGE22_6_ORGANIC_CONTEMPLATION_PART2.txt.
    respond_enable_thinking: bool = False
    respond_thinking_max_tokens: int = 2048


@dataclass(slots=True)
class ContractConfig:
    forbid_think_tags: bool = True
    forbid_visible_reasoning: bool = True
    forbid_prompt_echo: bool = True


@dataclass(slots=True)
class PromptConfig:
    ablation_mode: str = "current"
    # Phase 22 Stage 22.7 part D — drive-dosage experiment, defaults
    # reproduce prior behavior exactly (drive line every tick, imperative
    # framing, standard grounding rule). Tick surface only; respond()
    # always carries the drive line. See
    # docs/plans/PHASE22_STAGE22_7_SATURATION_LOOP_CORRECTIVES.txt.
    tick_drive_injection_interval: int = 1
    tick_drive_descriptive: bool = False
    tick_soft_grounding: bool = False
    # Phase 22 Stage 22.8 part D — read-back proportional to write. Defaults
    # reproduce prior behavior exactly: last-3 recency window, no revision
    # markers. See docs/plans/PHASE22_STAGE22_8_SELF_MODEL_WRITE_LOOP.txt.
    tick_heartbeat_sampling: str = "recent"
    tick_self_model_revision_visibility: bool = False
    # Phase 22 Stage 22.16 — feedback closure and record-keeping. Defaults
    # reproduce 22.15 exactly. tick_tool_feedback: every tool outcome (applied,
    # rate-limited with the wait, queued, opened, closed + export result,
    # tool error, parse failure) is carried into her next prompts, not only
    # read results. tick_carryover_entries: how many carried entries are
    # kept (22.10 fixed this at 2). tick_log_prompt_text: store the rendered
    # tick prompt on the trace (block list + sha256 are always stored).
    tick_tool_feedback: bool = False
    tick_carryover_entries: int = 2
    tick_log_prompt_text: bool = False


@dataclass(slots=True)
class SelfModelConfig:
    """Phase 22 Stage 22.8 — Nova's own write access to inquiry-class fields.

    Default OFF so every existing config and the full test suite keep the
    pre-22.8 queue-only behavior; the live config enables it explicitly.
    """

    nova_writable_inquiry_fields: bool = False
    # 12 ticks at the 300s production cadence. Pacing, not a gate.
    revision_min_seconds: int = 3_600


@dataclass(slots=True)
class GameConfig:
    """Phase 22 Stage 22.13 — reversi as exogenous input on the tick surface.

    Default OFF: every existing config and the full suite keep the pre-22.13
    tick surface byte-identical. The live config opts in explicitly. When
    enabled, play_reversi joins the tool menu in both registers and a
    [Reversi] block is rendered on every tick.
    """

    reversi_enabled: bool = False
    reversi_opponent: str = "greedy"
    # None = a fresh seed per game (recorded on the game); set for replay.
    reversi_seed: int | None = None
    # Stage 22.15 — rest after each result before the next game may open.
    # 0 = none (22.14 behaviour). Live: 1800 = 6 ticks at the 300s cadence.
    reversi_rest_seconds: int = 0


@dataclass(slots=True)
class PersonaConfig:
    name: str = "Nova"
    tone: str = "grounded, calm, intelligent, attentive"
    core_description: str = ""
    values: list[str] = field(default_factory=list)
    commitments: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryConfig:
    episodic_enabled: bool = True
    engram_enabled: bool = True
    semantic_enabled: bool = True
    graph_enabled: bool = True
    autobiographical_enabled: bool = True


@dataclass(slots=True)
class SessionConfig:
    max_recent_turns: int = 12
    autosave: bool = True


@dataclass(slots=True)
class ConsoleConfig:
    pending_proposal_max_age_seconds: int = 900


@dataclass(slots=True)
class EvalConfig:
    enable_probes: bool = True
    orientation_stability_threshold: float = 0.72
    orientation_min_runs: int = 2


@dataclass(slots=True)
class CognitionConfig:
    enabled: bool = True
    pass_budget: int = 1
    revision_ceiling: int = 1


@dataclass(slots=True)
class NovaConfig:
    app: AppConfig = field(default_factory=AppConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    cognition: CognitionConfig = field(default_factory=CognitionConfig)
    self_model: SelfModelConfig = field(default_factory=SelfModelConfig)
    game: GameConfig = field(default_factory=GameConfig)

    def validate(self) -> None:
        if self.model.backend not in VALID_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{self.model.backend}'. Valid backends: {sorted(VALID_BACKENDS)}"
            )
        if not self.model.model_path:
            raise ValueError("model.model_path is required")
        if self.model.n_ctx <= 0:
            raise ValueError("model.n_ctx must be positive")
        if self.generation.max_tokens <= 0:
            raise ValueError("generation.max_tokens must be positive")
        if self.generation.retries < 0:
            raise ValueError("generation.retries must be non-negative")
        if self.prompt.ablation_mode not in VALID_PROMPT_ABLATION_MODES:
            raise ValueError(
                "prompt.ablation_mode must be one of "
                f"{sorted(VALID_PROMPT_ABLATION_MODES)}"
            )
        if self.prompt.tick_drive_injection_interval < 1:
            raise ValueError("prompt.tick_drive_injection_interval must be >= 1")
        if self.prompt.tick_carryover_entries < 1:
            raise ValueError("prompt.tick_carryover_entries must be >= 1")
        if self.prompt.tick_heartbeat_sampling not in VALID_TICK_HEARTBEAT_SAMPLING:
            raise ValueError(
                "prompt.tick_heartbeat_sampling must be one of "
                f"{sorted(VALID_TICK_HEARTBEAT_SAMPLING)}"
            )
        if self.self_model.revision_min_seconds < 0:
            raise ValueError("self_model.revision_min_seconds must be non-negative")
        if self.game.reversi_rest_seconds < 0:
            raise ValueError("game.reversi_rest_seconds must be non-negative")
        if self.game.reversi_opponent not in VALID_REVERSI_OPPONENTS:
            raise ValueError(
                "game.reversi_opponent must be one of "
                f"{sorted(VALID_REVERSI_OPPONENTS)}"
            )
        if not self.app.data_dir:
            raise ValueError("app.data_dir is required")
        if not self.app.log_dir:
            raise ValueError("app.log_dir is required")
        if self.console.pending_proposal_max_age_seconds <= 0:
            raise ValueError("console.pending_proposal_max_age_seconds must be positive")
        if not 0.0 <= self.eval.orientation_stability_threshold <= 1.0:
            raise ValueError("eval.orientation_stability_threshold must be between 0.0 and 1.0")
        if self.eval.orientation_min_runs <= 0:
            raise ValueError("eval.orientation_min_runs must be positive")
        if self.cognition.pass_budget < 0:
            raise ValueError("cognition.pass_budget must be non-negative")
        if self.cognition.revision_ceiling < 0:
            raise ValueError("cognition.revision_ceiling must be non-negative")

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in config: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _section(section_type: type, payload: dict[str, Any] | None) -> Any:
    return section_type(**(payload or {}))


def load_config(
    *,
    default_path: str | Path = DEFAULT_CONFIG_PATH,
    override_path: str | Path | None = None,
) -> NovaConfig:
    default_cfg_path = Path(default_path).expanduser()
    if not default_cfg_path.is_absolute():
        default_cfg_path = Path.cwd() / default_cfg_path
    if not default_cfg_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_cfg_path}")

    payload = _read_yaml(default_cfg_path)

    if override_path is not None:
        user_cfg_path = Path(override_path).expanduser()
        if not user_cfg_path.is_absolute():
            user_cfg_path = Path.cwd() / user_cfg_path
        if not user_cfg_path.exists():
            raise FileNotFoundError(f"Override config not found: {user_cfg_path}")
        payload = _deep_merge(payload, _read_yaml(user_cfg_path))

    config = NovaConfig(
        app=_section(AppConfig, payload.get("app")),
        model=_section(ModelConfig, payload.get("model")),
        generation=_section(GenerationConfig, payload.get("generation")),
        contract=_section(ContractConfig, payload.get("contract")),
        prompt=_section(PromptConfig, payload.get("prompt")),
        persona=_section(PersonaConfig, payload.get("persona")),
        memory=_section(MemoryConfig, payload.get("memory")),
        session=_section(SessionConfig, payload.get("session")),
        console=_section(ConsoleConfig, payload.get("console")),
        eval=_section(EvalConfig, payload.get("eval")),
        cognition=_section(CognitionConfig, payload.get("cognition")),
        self_model=_section(SelfModelConfig, payload.get("self_model")),
        game=_section(GameConfig, payload.get("game")),
    )
    config.validate()
    return config
