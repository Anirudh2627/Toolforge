"""ToolForge: execution-verified tool-use training and evaluation for small LMs.

Phase 1 (this package's current scope) covers the CPU-only core: action grammar
and schema validation (:mod:`toolforge.schemas`), six deterministic mock tool
environments (:mod:`toolforge.environments`), the episode executor
(:mod:`toolforge.executor`), the state-based verifier (:mod:`toolforge.verifier`)
and the reproducible task benchmark (:mod:`toolforge.tasks`).

Training (SFT/GRPO) and model-backed evaluation are added in later phases; the
core never imports torch/transformers so it runs anywhere.
"""

from __future__ import annotations

import logging

__version__ = "0.1.0"

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a package logger; configures a stderr handler exactly once.

    Library code must not call ``basicConfig`` globally beyond this guard, so
    embedding applications (demo, notebooks) keep control of logging.
    """
    logger = logging.getLogger(name if name.startswith("toolforge") else f"toolforge.{name}")
    root = logging.getLogger("toolforge")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logger


from toolforge.schemas import (  # noqa: E402  (after get_logger definition)
    Action,
    ActionKind,
    ParseFailure,
    ToolSpec,
    ValidationIssue,
    parse_action,
    validate_against_schema,
)

__all__ = [
    "__version__",
    "get_logger",
    "Action",
    "ActionKind",
    "ParseFailure",
    "ToolSpec",
    "ValidationIssue",
    "parse_action",
    "validate_against_schema",
]
