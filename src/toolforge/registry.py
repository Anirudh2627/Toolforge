"""Environment registry and tool routing.

:class:`EnvironmentRegistry` maps family names to environment factories and
owns tool-spec lookup by qualified name (``family.tool``). :class:`ToolRouter`
binds a set of live environment instances to an episode so the executor can
dispatch qualified tool calls — this is what makes multi-family tasks ("check
the calendar, then email the attendee") work with the same code path as
single-family ones.

The MCP adapter (later phase) will wrap these same ToolSpecs; the core
benchmark never imports MCP packages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from toolforge.schemas import ToolForgeError, ToolSpec

if TYPE_CHECKING:  # runtime import would cycle via environments/__init__ -> environments.registry
    from toolforge.environments.base import Environment


class UnknownFamilyError(ToolForgeError):
    """Raised when a family name is not registered."""


class UnknownToolError(ToolForgeError):
    """Raised when a qualified tool name cannot be resolved."""


EnvironmentFactory = Callable[..., "Environment"]


class EnvironmentRegistry:
    """Registry of environment factories keyed by family name."""

    def __init__(self) -> None:
        self._factories: dict[str, EnvironmentFactory] = {}

    def register(self, family: str, factory: EnvironmentFactory) -> None:
        if not family or not family.replace("_", "").isalnum() or family != family.lower():
            raise ValueError(f"Family name must be lowercase alphanumeric/underscore: {family!r}")
        if family in self._factories:
            raise ValueError(f"Family {family!r} is already registered")
        self._factories[family] = factory

    def families(self) -> list[str]:
        return list(self._factories)

    def create(self, family: str, fixture: dict[str, Any] | None = None,
               seed: int = 0, **options: Any) -> Environment:
        if family not in self._factories:
            raise UnknownFamilyError(
                f"Unknown environment family {family!r}; registered: {sorted(self._factories)}")
        return self._factories[family](fixture=fixture, seed=seed, **options)

    def tool_specs(self, families: list[str] | None = None) -> list[ToolSpec]:
        """All tool specs for the given families (default: every registered family).

        Instantiates one environment per family with a throwaway seed because
        tool specs are class-level and state-independent.
        """
        wanted = families if families is not None else self.families()
        specs: list[ToolSpec] = []
        for family in wanted:
            env = self.create(family, seed=0)
            specs.extend(env.tools)
        return specs

    def tool_spec(self, qualified_name: str) -> ToolSpec:
        for spec in self.tool_specs():
            if spec.name == qualified_name:
                return spec
        raise UnknownToolError(f"No tool named {qualified_name!r}")


class ToolRouter:
    """Routes qualified tool names to live environment instances for one episode."""

    def __init__(self, environments: dict[str, Environment]) -> None:
        if not environments:
            raise ValueError("ToolRouter requires at least one environment")
        self._environments = environments
        self._tools: dict[str, tuple[Environment, str]] = {}
        for family, env in environments.items():
            for spec in env.tools:
                self._tools[spec.name] = (env, spec.local_name)

    @property
    def environments(self) -> dict[str, Environment]:
        return self._environments

    def tool_specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for env in self._environments.values():
            specs.extend(env.tools)
        return specs

    def spec_for(self, qualified_name: str) -> ToolSpec | None:
        entry = self._tools.get(qualified_name)
        if entry is None:
            return None
        env, local = entry
        for spec in env.tools:
            if spec.local_name == local:
                return spec
        return None  # unreachable: router built from these specs

    def known_tool_names(self) -> list[str]:
        return sorted(self._tools)

    def resolve(self, qualified_name: str) -> tuple[Environment, str]:
        if qualified_name not in self._tools:
            raise UnknownToolError(
                f"Unknown tool {qualified_name!r}; available: {self.known_tool_names()}")
        return self._tools[qualified_name]

    def snapshots(self) -> dict[str, dict[str, Any]]:
        return {family: env.snapshot() for family, env in self._environments.items()}

    def volatile_keys(self) -> dict[str, frozenset[str]]:
        return {family: env.volatile_keys for family, env in self._environments.items()}


def default_registry() -> EnvironmentRegistry:
    """A fresh registry containing all six built-in families."""
    from toolforge.environments.registry import register_default_environments
    return register_default_environments(EnvironmentRegistry())
