"""Default environment registration.

Kept separate from :mod:`toolforge.registry` (the generic machinery) so the
set of shipped environments can evolve without touching the router/executor.
"""

from __future__ import annotations

from toolforge.environments.base import Environment
from toolforge.environments.calendar import CalendarEnvironment
from toolforge.environments.crm import CrmEnvironment
from toolforge.environments.email import EmailEnvironment
from toolforge.environments.files import FilesEnvironment
from toolforge.environments.sql import SqlEnvironment
from toolforge.environments.todo import TodoEnvironment
from toolforge.registry import EnvironmentRegistry

#: family -> environment class, in the canonical benchmark order.
ENVIRONMENT_CLASSES: dict[str, type[Environment]] = {
    CalendarEnvironment.family: CalendarEnvironment,
    EmailEnvironment.family: EmailEnvironment,
    FilesEnvironment.family: FilesEnvironment,
    CrmEnvironment.family: CrmEnvironment,
    SqlEnvironment.family: SqlEnvironment,
    TodoEnvironment.family: TodoEnvironment,
}

ALL_FAMILIES: tuple[str, ...] = tuple(ENVIRONMENT_CLASSES)


def register_default_environments(registry: EnvironmentRegistry) -> EnvironmentRegistry:
    """Register all six built-in families on ``registry`` (mutates and returns it)."""
    for family, cls in ENVIRONMENT_CLASSES.items():
        registry.register(family, cls)
    return registry
