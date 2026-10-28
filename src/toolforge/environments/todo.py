"""Task-management environment ("todo" family).

Named ``todo`` to avoid clashing with the benchmark module
:mod:`toolforge.tasks`. State shape::

    {
      "current_date": "2026-03-02",
      "users": ["user","alice","bob","carol","dave"],
      "tasks": [{"id","title","description","status","owner","assignee",
                 "deadline","priority","project","created"}],
      "next_task_num": int
    }

``status`` in open|in_progress|blocked|done|cancelled; ``priority`` in
low|medium|high. "Overdue" = deadline < current_date and status is one of
open/in_progress/blocked.
"""

from __future__ import annotations

import random
from typing import Any, ClassVar

from toolforge.environments.base import (
    ERR_INVALID_ARGUMENT,
    ERR_NOT_FOUND,
    Environment,
    ToolResult,
    require,
)
from toolforge.environments.pools import pick
from toolforge.schemas import ToolSpec
from toolforge.timeutils import (
    DEFAULT_CURRENT_DATE,
    TimeFormatError,
    add_days,
    is_valid_date,
    parse_date,
)

STATUSES = ["open", "in_progress", "blocked", "done", "cancelled"]
OPEN_STATUSES = ("open", "in_progress", "blocked")
PRIORITIES = ["low", "medium", "high"]
USERS = ["user", "alice", "bob", "carol", "dave"]
PROJECTS = ["website", "billing", "hiring", "infra"]

_SEED_TITLES = [
    ("Fix checkout button overlap on mobile", "website", "high"),
    ("Draft Q2 hiring plan", "hiring", "medium"),
    ("Migrate billing DB to new cluster", "billing", "high"),
    ("Write onboarding email sequence", "website", "low"),
    ("Review vendor security questionnaire", "infra", "medium"),
    ("Set up staging for payments service", "billing", "high"),
    ("Publish January metrics summary", "website", "medium"),
    ("Schedule panel interviews", "hiring", "medium"),
    ("Rotate expired TLS certificates", "infra", "high"),
    ("Clean up stale feature flags", "website", "low"),
]


class TodoEnvironment(Environment):
    family: ClassVar[str] = "todo"
    volatile_keys: ClassVar[frozenset[str]] = frozenset()

    # ------------------------------------------------------------------ state

    def default_state(self, seed: int) -> dict[str, Any]:
        rng = random.Random(seed)
        tasks = []
        for i, (title, project, priority) in enumerate(_SEED_TITLES):
            status = pick(rng, ["open", "open", "in_progress", "blocked", "done"])
            # Two deterministic overdue tasks (deadlines before current_date).
            if i in (2, 8):
                deadline = add_days(DEFAULT_CURRENT_DATE, -(1 + i % 5))
                status = pick(rng, ["open", "in_progress"])
            else:
                deadline = add_days(DEFAULT_CURRENT_DATE, rng.randrange(-1, 21))
            tasks.append({
                "id": f"tsk_{i + 1:03d}",
                "title": title,
                "description": "",
                "status": status,
                "owner": "user",
                "assignee": pick(rng, USERS),
                "deadline": deadline,
                "priority": priority,
                "project": project,
                "created": add_days(DEFAULT_CURRENT_DATE, -(10 - i)),
            })
        return {
            "current_date": DEFAULT_CURRENT_DATE,
            "users": list(USERS),
            "tasks": tasks,
            "next_task_num": len(tasks) + 1,
        }

    def normalize_state(self, fixture: dict[str, Any]) -> dict[str, Any]:
        require(isinstance(fixture, dict), "todo fixture must be an object")
        users = fixture.get("users", USERS)
        require(isinstance(users, list) and all(isinstance(u, str) and u for u in users),
                "users must be a list of non-empty strings")
        state: dict[str, Any] = {
            "current_date": str(fixture.get("current_date", DEFAULT_CURRENT_DATE)),
            "users": list(users),
            "tasks": [],
            "next_task_num": 1,
        }
        require(is_valid_date(state["current_date"]), "current_date must be YYYY-MM-DD")
        raw = fixture.get("tasks", [])
        require(isinstance(raw, list), "tasks must be a list")
        seen_ids: set[str] = set()
        for i, t in enumerate(raw):
            require(isinstance(t, dict), f"tasks[{i}] must be an object")
            for key in ("id", "title", "status"):
                require(isinstance(t.get(key), str) and t[key], f"tasks[{i}] missing {key!r}")
            require(t["id"] not in seen_ids, f"duplicate task id {t['id']!r}")
            seen_ids.add(t["id"])
            require(t["status"] in STATUSES, f"tasks[{i}].status must be one of {STATUSES}")
            priority = t.get("priority", "medium")
            require(priority in PRIORITIES, f"tasks[{i}].priority must be one of {PRIORITIES}")
            deadline = t.get("deadline")
            require(deadline is None or (isinstance(deadline, str) and is_valid_date(deadline)),
                    f"tasks[{i}].deadline must be null or YYYY-MM-DD")
            assignee = t.get("assignee")
            require(assignee is None or (isinstance(assignee, str) and assignee in users),
                    f"tasks[{i}].assignee {assignee!r} not in users")
            state["tasks"].append({
                "id": t["id"], "title": t["title"],
                "description": str(t.get("description", "")),
                "status": t["status"], "owner": str(t.get("owner", "user")),
                "assignee": assignee, "deadline": deadline, "priority": priority,
                "project": str(t.get("project", "")),
                "created": str(t.get("created", state["current_date"])),
            })
        next_num = fixture.get("next_task_num")
        if next_num is None:
            next_num = len(state["tasks"]) + 1
        require(isinstance(next_num, int) and next_num >= 1, "next_task_num must be a positive int")
        state["next_task_num"] = next_num
        return state

    # ------------------------------------------------------------------ tools

    _deadline_schema: ClassVar[dict] = {
        "type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$",
        "description": "Deadline date (YYYY-MM-DD).",
    }

    tool_specs: ClassVar[list[ToolSpec]] = [
        ToolSpec(
            "todo.create_task",
            "Create a task. Status starts as 'open'. Assignee must be a known user.",
            {"type": "object", "properties": {
                "title": {"type": "string", "minLength": 1, "maxLength": 200},
                "description": {"type": "string", "maxLength": 2000},
                "assignee": {"type": "string", "enum": USERS},
                "deadline": _deadline_schema,
                "priority": {"type": "string", "enum": PRIORITIES},
                "project": {"type": "string", "maxLength": 80},
            }, "required": ["title"], "additionalProperties": False},
        ),
        ToolSpec(
            "todo.list_tasks",
            "List tasks with optional filters (AND-combined). overdue_only selects "
            "tasks past their deadline that are not done/cancelled.",
            {"type": "object", "properties": {
                "status": {"type": "string", "enum": STATUSES},
                "assignee": {"type": "string"},
                "project": {"type": "string"},
                "overdue_only": {"type": "boolean"},
            }, "additionalProperties": False},
        ),
        ToolSpec(
            "todo.assign_task",
            "Assign (or reassign) a task to a known user.",
            {"type": "object", "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "assignee": {"type": "string", "enum": USERS},
            }, "required": ["task_id", "assignee"], "additionalProperties": False},
        ),
        ToolSpec(
            "todo.update_task_status",
            "Set a task's status: open | in_progress | blocked | done | cancelled.",
            {"type": "object", "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "status": {"type": "string", "enum": STATUSES},
            }, "required": ["task_id", "status"], "additionalProperties": False},
        ),
        ToolSpec(
            "todo.set_task_deadline",
            "Set or change a task's deadline (YYYY-MM-DD).",
            {"type": "object", "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "deadline": _deadline_schema,
            }, "required": ["task_id", "deadline"], "additionalProperties": False},
        ),
        ToolSpec(
            "todo.cancel_task",
            "Cancel a task (sets status to 'cancelled'; the record is kept).",
            {"type": "object", "properties": {"task_id": {"type": "string", "minLength": 1}},
             "required": ["task_id"], "additionalProperties": False},
        ),
    ]

    # ------------------------------------------------------------ tool bodies

    def _find_task(self, task_id: Any) -> dict[str, Any] | None:
        if not isinstance(task_id, str):
            return None
        for t in self._state["tasks"]:
            if t["id"] == task_id:
                return t
        return None

    def _is_overdue(self, t: dict[str, Any]) -> bool:
        return (t["deadline"] is not None
                and t["deadline"] < self._state["current_date"]
                and t["status"] in OPEN_STATUSES)

    def _tool_create_task(self, args: dict[str, Any]) -> ToolResult:
        if args.get("deadline") is not None:
            try:
                parse_date(args["deadline"])
            except TimeFormatError as exc:
                return self._fail(ERR_INVALID_ARGUMENT, str(exc))
        task = {
            "id": f"tsk_{self._state['next_task_num']:03d}",
            "title": args["title"],
            "description": args.get("description", ""),
            "status": "open",
            "owner": "user",
            "assignee": args.get("assignee"),
            "deadline": args.get("deadline"),
            "priority": args.get("priority", "medium"),
            "project": args.get("project", ""),
            "created": self._state["current_date"],
        }
        self._state["next_task_num"] += 1
        self._state["tasks"].append(task)
        return self._ok({"task": task})

    def _tool_list_tasks(self, args: dict[str, Any]) -> ToolResult:
        tasks = self._state["tasks"]
        if args.get("status"):
            tasks = [t for t in tasks if t["status"] == args["status"]]
        if args.get("assignee"):
            tasks = [t for t in tasks if t["assignee"] == args["assignee"]]
        if args.get("project"):
            tasks = [t for t in tasks if t["project"] == args["project"]]
        if args.get("overdue_only"):
            tasks = [t for t in tasks if self._is_overdue(t)]
        tasks = sorted(tasks, key=lambda t: t["id"])
        return self._ok({"tasks": tasks, "count": len(tasks)})

    def _tool_assign_task(self, args: dict[str, Any]) -> ToolResult:
        task = self._find_task(args["task_id"])
        if task is None:
            return self._fail(ERR_NOT_FOUND, f"No task with id {args['task_id']!r}")
        task["assignee"] = args["assignee"]
        return self._ok({"task": task})

    def _tool_update_task_status(self, args: dict[str, Any]) -> ToolResult:
        task = self._find_task(args["task_id"])
        if task is None:
            return self._fail(ERR_NOT_FOUND, f"No task with id {args['task_id']!r}")
        task["status"] = args["status"]
        return self._ok({"task": task})

    def _tool_set_task_deadline(self, args: dict[str, Any]) -> ToolResult:
        task = self._find_task(args["task_id"])
        if task is None:
            return self._fail(ERR_NOT_FOUND, f"No task with id {args['task_id']!r}")
        try:
            parse_date(args["deadline"])
        except TimeFormatError as exc:
            return self._fail(ERR_INVALID_ARGUMENT, str(exc))
        task["deadline"] = args["deadline"]
        return self._ok({"task": task})

    def _tool_cancel_task(self, args: dict[str, Any]) -> ToolResult:
        task = self._find_task(args["task_id"])
        if task is None:
            return self._fail(ERR_NOT_FOUND, f"No task with id {args['task_id']!r}")
        task["status"] = "cancelled"
        return self._ok({"task": task})
