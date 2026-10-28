"""Base class and shared types for the deterministic mock tool environments.

Conventions every environment follows:

- State is a plain JSON-serializable dict; ``snapshot()`` returns a deep copy
  so verifiers and traces never alias live state.
- ``current_date`` lives in state ("2026-03-02" by default); environments
  never read the system clock.
- Tools are methods named ``_tool_<local_name>`` taking the validated
  arguments dict and returning a :class:`ToolResult`.
- Tool methods signal user-facing problems by returning failure results, not
  by raising. Unexpected exceptions are caught once in :meth:`Environment.execute`,
  logged with traceback, and surfaced as ``internal_error`` results — an
  environment bug must crash the pipeline in tests (which assert on error
  codes) rather than silently corrupt a rollout.
- ``volatile_keys`` names leaf fields whose mutation does not count as a
  task-relevant side effect (e.g. email ``read`` flags). The verifier ignores
  exactly these when checking for forbidden state changes.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from toolforge import get_logger
from toolforge.schemas import ToolForgeError, ToolSpec

logger = get_logger("env")

# Structured tool error codes (kept small and stable; metrics group by these).
ERR_NOT_FOUND = "not_found"
ERR_ALREADY_EXISTS = "already_exists"
ERR_INVALID_ARGUMENT = "invalid_argument"
ERR_AMBIGUOUS = "ambiguous"
ERR_UNSAFE = "unsafe_operation"
ERR_UNSUPPORTED = "unsupported"
ERR_LIMIT = "limit_exceeded"
ERR_UNKNOWN_TOOL = "unknown_tool"
ERR_INTERNAL = "internal_error"


class InvalidFixtureError(ToolForgeError):
    """Raised when a task fixture does not match the environment's state shape."""


class ToolResult:
    """Outcome of one tool execution: either ``ok`` with output, or an error."""

    __slots__ = ("ok", "output", "error")

    def __init__(self, ok: bool, output: Any = None, error: dict[str, str] | None = None) -> None:
        self.ok = ok
        self.output = output
        self.error = error

    @classmethod
    def success(cls, output: Any = None) -> "ToolResult":
        return cls(True, output=output)

    @classmethod
    def failure(cls, code: str, message: str) -> "ToolResult":
        return cls(False, error={"code": code, "message": message})

    def to_dict(self) -> dict[str, Any]:
        if self.ok:
            return {"ok": True, "output": self.output}
        return {"ok": False, "error": dict(self.error or {})}

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ToolResult(ok={self.ok}, output={self.output!r}, error={self.error!r})"


def require(condition: bool, message: str) -> None:
    """Fixture validation helper: raise InvalidFixtureError when condition fails."""
    if not condition:
        raise InvalidFixtureError(message)


def fresh_id(existing_ids: set[str], prefix: str, start: int) -> str:
    """Deterministic id allocation: first ``prefix_NNN`` (N >= start) not taken."""
    n = start
    while f"{prefix}_{n:03d}" in existing_ids:
        n += 1
    return f"{prefix}_{n:03d}"


class Environment(ABC):
    """One isolated, deterministic tool family with in-memory state."""

    #: family identifier used in qualified tool names ("calendar.create_event")
    family: ClassVar[str] = ""
    #: leaf field names whose changes are not task-relevant side effects
    volatile_keys: ClassVar[frozenset[str]] = frozenset()
    #: tool specifications; concrete classes build this once at import time
    tool_specs: ClassVar[list[ToolSpec]] = []

    def __init__(self, fixture: dict[str, Any] | None = None, seed: int = 0, **options: Any) -> None:
        self._seed = seed
        self._options = options
        self._state: dict[str, Any] = {}
        self.reset(fixture=fixture, seed=seed)

    # -- lifecycle ---------------------------------------------------------

    def reset(self, fixture: dict[str, Any] | None = None, seed: int | None = None) -> None:
        """Restore state from an explicit fixture, or regenerate it from seed."""
        if seed is not None:
            self._seed = seed
        if fixture is not None:
            self._state = self.normalize_state(fixture)
        else:
            self._state = self.default_state(self._seed)
        self._post_reset()

    @abstractmethod
    def default_state(self, seed: int) -> dict[str, Any]:
        """Deterministically generate a fresh state from ``seed``."""

    @abstractmethod
    def normalize_state(self, fixture: dict[str, Any]) -> dict[str, Any]:
        """Validate a fixture and fill in defaults; raise InvalidFixtureError."""

    def _post_reset(self) -> None:
        """Hook for derived resources (e.g. the SQL environment's connection)."""

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def snapshot(self) -> dict[str, Any]:
        """Deep, JSON-serializable copy of the current state."""
        return copy.deepcopy(self._state)

    # -- tools ---------------------------------------------------------------

    @property
    def tools(self) -> list[ToolSpec]:
        return list(self.tool_specs)

    def tool_names(self) -> list[str]:
        return [spec.name for spec in self.tool_specs]

    def execute(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a local tool name (no family prefix) with validated arguments.

        Argument-schema validation happens in the executor *before* this call;
        environments additionally guard their own invariants and return
        structured failures.
        """
        method = getattr(self, f"_tool_{tool}", None)
        if method is None or not callable(method):
            return ToolResult.failure(ERR_UNKNOWN_TOOL,
                                      f"Tool '{self.family}.{tool}' does not exist. "
                                      f"Available: {', '.join(t.local_name for t in self.tool_specs)}")
        try:
            return method(arguments)
        except Exception as exc:  # environment bug: log loudly, surface structure
            logger.exception("Internal error executing %s.%s", self.family, tool)
            return ToolResult.failure(ERR_INTERNAL, f"{type(exc).__name__}: {exc}")

    # -- helpers for concrete environments ------------------------------------

    @staticmethod
    def _ok(output: Any = None) -> ToolResult:
        return ToolResult.success(output)

    @staticmethod
    def _fail(code: str, message: str) -> ToolResult:
        return ToolResult.failure(code, message)
