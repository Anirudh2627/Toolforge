"""The six deterministic mock tool environments plus registration helpers.

Importing this package does not register anything globally; call
:func:`toolforge.registry.default_registry` (or
:func:`toolforge.environments.registry.register_default_environments`) to get
a wired registry. Every environment is pure-Python, in-memory, and CPU-only.
"""

from __future__ import annotations

from toolforge.environments.base import (
    ERR_ALREADY_EXISTS,
    ERR_AMBIGUOUS,
    ERR_INTERNAL,
    ERR_INVALID_ARGUMENT,
    ERR_LIMIT,
    ERR_NOT_FOUND,
    ERR_UNSAFE,
    ERR_UNKNOWN_TOOL,
    ERR_UNSUPPORTED,
    Environment,
    InvalidFixtureError,
    ToolResult,
)
from toolforge.environments.calendar import CalendarEnvironment
from toolforge.environments.crm import CrmEnvironment
from toolforge.environments.email import EmailEnvironment
from toolforge.environments.files import FilesEnvironment
from toolforge.environments.registry import ALL_FAMILIES, ENVIRONMENT_CLASSES
from toolforge.environments.sql import SqlEnvironment
from toolforge.environments.todo import TodoEnvironment

__all__ = [
    "Environment", "ToolResult", "InvalidFixtureError",
    "CalendarEnvironment", "EmailEnvironment", "FilesEnvironment",
    "CrmEnvironment", "SqlEnvironment", "TodoEnvironment",
    "ENVIRONMENT_CLASSES", "ALL_FAMILIES",
    "ERR_ALREADY_EXISTS", "ERR_AMBIGUOUS", "ERR_INTERNAL", "ERR_INVALID_ARGUMENT",
    "ERR_LIMIT", "ERR_NOT_FOUND", "ERR_UNSAFE", "ERR_UNKNOWN_TOOL", "ERR_UNSUPPORTED",
]
