"""Typed, YAML-backed configuration.

Every knob lives here so experiments are reproducible from one file. Load with
`load_config(path, overrides=["steering.layer=14", ...])`; dotted overrides are
parsed as YAML scalars, so numbers, booleans and lists work as expected.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dtype: str = "auto"          # auto | float32 | bfloat16 | float16
    device: str = "auto"         # auto | cpu | cuda | cuda:1 | mps
    trust_remote_code: bool = False


@dataclass
class DataConfig:
    pairs: str = "data/contrastive_pairs.jsonl"
    holdout_pairs: str | None = None  # separate held-out file; when set, all of `pairs` is used for training
    holdout_fraction: float = 0.2     # used only without holdout_pairs: stratified split of `pairs` by domain
    seed: int = 42


@dataclass
class ExtractionConfig:
    batch_size: int = 8
    max_norm_samples: int = 50_000  # tokens kept per layer for norm calibration


@dataclass
class SteeringConfig:
    layer: int = 12              # decoder layer whose OUTPUT is steered (0-based)
    readout_layer: int = 18      # decoder layer whose output is read for explanations
    scale_fraction: float = 0.2  # alpha=1 adds this fraction of a typical token norm
    positions: str = "all"       # all | generated  (steer prompt tokens too, or only new ones)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 90
    batch_size: int = 8
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 1234


@dataclass
class JudgeConfig:
    backend: str = "auto"        # auto | anthropic | hf | none
    anthropic_model: str = "claude-haiku-5-5"
    hf_model: str = "Qwen/Qwen2.5-7B-Instruct"
    hf_dtype: str = "half"       # half precision: a 7B judge needs ~15 GB instead of ~30 GB
    cache: bool = True


@dataclass
class EvalConfig:
    alphas: list[float] = field(default_factory=lambda: [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    max_new_tokens: int = 75
    include_holdout: bool = True
    holdout_limit: int | None = None  # evaluate only the first N held-out prompts (balanced across styles) for speed
    prompts_file: str | None = None  # your own prompts: .txt (one per line), .jsonl or .csv with a `prompt` column


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "runs"
    source_path: str | None = None  # where this config was loaded from (set automatically)

    @property
    def artifact_dir(self) -> Path:
        """Per-model directory holding extracted vectors and diagnostics."""
        return Path(self.output_dir) / model_slug(self.model.id)

    def resolve_path(self, path: str | None) -> Path | None:
        """Resolve a path relative to the CWD, falling back to the config file's project root."""
        if path is None:
            return None
        p = Path(path).expanduser()
        if p.is_absolute() or p.exists() or self.source_path is None:
            return p
        cfg_dir = Path(self.source_path).resolve().parent
        for base in (cfg_dir, cfg_dir.parent):
            if (base / p).exists():
                return base / p
        return p

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("source_path", None)
        return d


def model_slug(model_id: str) -> str:
    """'org/name' -> 'org__name'; a local model directory -> its folder name."""
    p = Path(model_id).expanduser()
    if p.is_dir():
        return "local__" + p.resolve().name
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_id.strip("/"))


def _merge(dc: Any, values: dict[str, Any], prefix: str = "") -> None:
    known = {f.name: f for f in fields(dc)}
    for key, val in values.items():
        if key not in known:
            raise KeyError(f"Unknown config key: {prefix}{key}")
        cur = getattr(dc, key)
        if is_dataclass(cur):
            if not isinstance(val, dict):
                raise TypeError(f"{prefix}{key} must be a mapping")
            _merge(cur, val, prefix=f"{prefix}{key}.")
        else:
            setattr(dc, key, val)


def _set_dotted(cfg: Config, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    nested: dict[str, Any] = value
    for part in reversed(parts):
        nested = {part: nested}
    _merge(cfg, nested)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        _merge(cfg, raw)
        cfg.source_path = str(path)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, val = item.split("=", 1)
        _set_dotted(cfg, key.strip(), yaml.safe_load(val))
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if not 0 < cfg.data.holdout_fraction < 1:
        raise ValueError("data.holdout_fraction must be between 0 and 1")
    if cfg.steering.positions not in {"all", "generated"}:
        raise ValueError("steering.positions must be 'all' or 'generated'")
    if cfg.judge.backend not in {"auto", "anthropic", "hf", "none"}:
        raise ValueError("judge.backend must be auto, anthropic, hf or none")


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
