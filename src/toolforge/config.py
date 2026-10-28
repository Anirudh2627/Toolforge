"""Configuration loading with strict key validation.

Configs are small frozen dataclasses fed from YAML (``configs/base.yaml``).
Unknown keys raise :class:`ConfigError` instead of being ignored: a typo in a
config file must never silently change experiment behavior. Later phases extend
this module with training (SFT/GRPO) sections.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised for malformed config files or unknown/invalid keys."""


@dataclass(frozen=True)
class PathSettings:
    data_dir: str = "data"
    tasks_file: str = "data/tasks/benchmark_v1.jsonl"
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"


@dataclass(frozen=True)
class EnvSettings:
    default_max_steps: int = 8
    repeat_limit: int = 3
    sql_max_rows: int = 50
    sql_progress_limit: int = 1_000_000


@dataclass(frozen=True)
class BenchmarkSettings:
    seed: int = 20260301
    version: str = "v1"
    held_out_families: tuple[str, ...] = ("crm",)
    train_fraction: float = 0.7
    val_fraction: float = 0.1


@dataclass(frozen=True)
class LoggingSettings:
    level: str = "INFO"


@dataclass(frozen=True)
class ToolForgeConfig:
    project: str = "toolforge"
    seed: int = 20260301
    paths: PathSettings = field(default_factory=PathSettings)
    env: EnvSettings = field(default_factory=EnvSettings)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


_SECTION_TYPES: dict[str, type] = {
    "paths": PathSettings,
    "env": EnvSettings,
    "benchmark": BenchmarkSettings,
    "logging": LoggingSettings,
}

_TOP_LEVEL_SCALARS = {"project": str, "seed": int}


def _deep_merge(base: dict, override: Mapping[str, Any], where: str) -> dict:
    merged = dict(base)
    for key, value in override.items():
        path = f"{where}.{key}" if where else key
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value, path)
        else:
            merged[key] = value
    return merged


def _build_section(cls: type, raw: Mapping[str, Any], where: str) -> Any:
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown key(s) in {where}: {sorted(unknown)} (known: {sorted(known)})")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in raw:
            continue
        value = raw[f.name]
        if f.name == "held_out_families":
            if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
                raise ConfigError(f"{where}.held_out_families must be a list of strings, got {value!r}")
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _as_dict(data: Any) -> dict:
    """Convert config dataclasses back to plain dicts (for run metadata)."""
    if is_dataclass(data) and not isinstance(data, type):
        return {f.name: _as_dict(getattr(data, f.name)) for f in fields(data)}
    if isinstance(data, dict):
        return {k: _as_dict(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_as_dict(v) for v in data]
    return data


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> ToolForgeConfig:
    """Load configuration from YAML, layered over built-in defaults.

    Args:
        path: YAML file (e.g. ``configs/base.yaml``). ``None`` uses defaults.
        overrides: nested dict merged on top of the file (highest priority),
            useful for notebook/CLI tweaks without editing files.
    """
    defaults = _as_dict(ToolForgeConfig())
    raw: dict = {}
    if path is not None:
        file = Path(path)
        if not file.exists():
            raise ConfigError(f"Config file not found: {file}")
        loaded = yaml.safe_load(file.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"Config root must be a mapping, got {type(loaded).__name__} in {file}")
        raw = loaded
    if overrides:
        raw = _deep_merge(raw, overrides, "")

    merged = _deep_merge(defaults, raw, "")

    unknown_top = set(merged) - set(_SECTION_TYPES) - set(_TOP_LEVEL_SCALARS)
    if unknown_top:
        raise ConfigError(
            f"Unknown top-level config key(s): {sorted(unknown_top)} "
            f"(known: {sorted(set(_SECTION_TYPES) | set(_TOP_LEVEL_SCALARS))})"
        )
    for key, expected in _TOP_LEVEL_SCALARS.items():
        if key in merged and not isinstance(merged[key], expected):
            raise ConfigError(f"Config key '{key}' must be {expected.__name__}, got {merged[key]!r}")

    sections = {name: _build_section(cls, merged.get(name, {}), name) for name, cls in _SECTION_TYPES.items()}
    bench: BenchmarkSettings = sections["benchmark"]
    if not (0.0 < bench.train_fraction < 1.0 and 0.0 <= bench.val_fraction < 1.0):
        raise ConfigError("train_fraction must be in (0,1) and val_fraction in [0,1)")
    if bench.train_fraction + bench.val_fraction >= 1.0:
        raise ConfigError("train_fraction + val_fraction must be < 1.0 (remainder is test)")
    return ToolForgeConfig(
        project=merged.get("project", "toolforge"),
        seed=merged.get("seed", 20260301),
        **sections,
    )


def config_to_dict(config: ToolForgeConfig) -> dict:
    """Serialize a config for experiment metadata files."""
    return _as_dict(config)
