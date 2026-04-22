"""Configuration loading for Nova 2.0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/nova.default.yaml")
VALID_BACKENDS = {"llama_cpp"}


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


@dataclass(slots=True)
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: list[str] = field(default_factory=lambda: ["User:", "\nUser:"])
    retries: int = 2


@dataclass(slots=True)
class ContractConfig:
    forbid_think_tags: bool = True
    forbid_visible_reasoning: bool = True
    forbid_prompt_echo: bool = True


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
    semantic_enabled: bool = False
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
class NovaConfig:
    app: AppConfig = field(default_factory=AppConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

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
        persona=_section(PersonaConfig, payload.get("persona")),
        memory=_section(MemoryConfig, payload.get("memory")),
        session=_section(SessionConfig, payload.get("session")),
        console=_section(ConsoleConfig, payload.get("console")),
        eval=_section(EvalConfig, payload.get("eval")),
    )
    config.validate()
    return config
