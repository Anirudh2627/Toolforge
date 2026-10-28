"""Shared fixtures for the CPU-only Phase 1 test suite (no model downloads)."""

from __future__ import annotations

import pytest

from toolforge.config import EnvSettings
from toolforge.environments import (
    CalendarEnvironment,
    CrmEnvironment,
    EmailEnvironment,
    FilesEnvironment,
    SqlEnvironment,
    TodoEnvironment,
)
from toolforge.executor import Executor
from toolforge.registry import default_registry
from toolforge.tasks import ExpectedOutcome, Task

ALL_ENV_CLASSES = [
    CalendarEnvironment, EmailEnvironment, FilesEnvironment,
    CrmEnvironment, SqlEnvironment, TodoEnvironment,
]


@pytest.fixture(scope="session")
def registry():
    return default_registry()


@pytest.fixture
def executor(registry):
    return Executor(registry, settings=EnvSettings(default_max_steps=8, repeat_limit=3))


@pytest.fixture
def small_calendar_fixture():
    """A tiny explicit calendar state used across executor/verifier tests."""
    return {
        "current_date": "2026-03-02",
        "user_email": "user@example.com",
        "next_event_num": 3,
        "events": [
            {"id": "evt_001", "title": "Standup", "date": "2026-03-03", "start": "09:00",
             "end": "09:30", "attendees": ["alice@example.com"], "location": "Zoom",
             "notes": "", "status": "scheduled"},
            {"id": "evt_002", "title": "Review", "date": "2026-03-03", "start": "11:00",
             "end": "12:00", "attendees": ["bob@example.com"], "location": "Room A",
             "notes": "", "status": "scheduled"},
        ],
    }


def make_calendar_task(task_id: str = "test-cal-001", request: str = "Create an event.",
                       checks: list | None = None, terminal: str = "final_answer",
                       forbid: bool = False, max_steps: int = 8,
                       fixture: dict | None = None,
                       small_calendar_fixture: dict | None = None) -> Task:
    """Build a minimal calendar Task for unit tests."""
    fx = fixture if fixture is not None else {
        "current_date": "2026-03-02",
        "user_email": "user@example.com",
        "next_event_num": 1,
        "events": [],
    }
    return Task(
        task_id=task_id, family="calendar", request=request, category="single_step_selection",
        difficulty="easy", fixture=fx, fixture_seed=0,
        expected=ExpectedOutcome(terminal_action=terminal, forbid_state_changes=forbid,
                                 checks=checks or [], max_invalid_calls=None),
        tool_families=["calendar"], max_steps=max_steps, tags=["unit-test"],
        source="handwritten", benchmark_seed=0, benchmark_version="test",
    )
