"""Environment tests: determinism, isolation, state transitions, error handling.

All CPU-only; no model downloads. Each environment is exercised through its
public contract (default_state/reset/snapshot/execute) plus family-specific
behavior.
"""

from __future__ import annotations

import copy
import json

import pytest

from toolforge.environments import (
    ENVIRONMENT_CLASSES,
    ERR_ALREADY_EXISTS,
    ERR_AMBIGUOUS,
    ERR_INVALID_ARGUMENT,
    ERR_NOT_FOUND,
    ERR_UNKNOWN_TOOL,
    ERR_UNSAFE,
    CalendarEnvironment,
    CrmEnvironment,
    EmailEnvironment,
    FilesEnvironment,
    InvalidFixtureError,
    SqlEnvironment,
    TodoEnvironment,
)
from toolforge.environments.pools import EMAILS
from toolforge.schemas import ToolSpec, validate_against_schema

ALL_ENV_CLASSES = list(ENVIRONMENT_CLASSES.values())


# --------------------------------------------------------------------------
# Contract tests applied to every environment
# --------------------------------------------------------------------------

class TestCommonContract:
    @pytest.mark.parametrize("cls", ALL_ENV_CLASSES)
    def test_default_state_is_deterministic(self, cls):
        a = cls(seed=42).snapshot()
        b = cls(seed=42).snapshot()
        assert a == b

    @pytest.mark.parametrize("cls", ALL_ENV_CLASSES)
    def test_different_seeds_differ(self, cls):
        # Every generator mixes the seed into content (ids may coincide).
        a = cls(seed=1).snapshot()
        b = cls(seed=999_999).snapshot()
        assert a != b

    @pytest.mark.parametrize("cls", ALL_ENV_CLASSES)
    def test_state_is_json_serializable(self, cls):
        json.dumps(cls(seed=7).snapshot())

    @pytest.mark.parametrize("cls", ALL_ENV_CLASSES)
    def test_instances_are_isolated(self, cls):
        env1, env2 = cls(seed=5), cls(seed=5)
        before = env2.snapshot()
        # Mutate env1 through whichever write tool exists; env2 must not change.
        _apply_one_write(env1)
        assert env2.snapshot() == before
        assert env1.snapshot() != before

    @pytest.mark.parametrize("cls", ALL_ENV_CLASSES)
    def test_reset_restores_initial_state(self, cls):
        env = cls(seed=11)
        initial = env.snapshot()
        _apply_one_write(env)
        assert env.snapshot() != initial
        env.reset()
        assert env.snapshot() == initial

    @pytest.mark.parametrize("cls", ALL_ENV_CLASSES)
    def test_reset_with_fixture_roundtrips(self, cls):
        env = cls(seed=3)
        fixture = env.snapshot()
        env2 = cls(fixture=copy.deepcopy(fixture))
        assert env2.snapshot() == fixture

    @pytest.mark.parametrize("cls", ALL_ENV_CLASSES)
    def test_snapshot_is_deep_copy(self, cls):
        env = cls(seed=8)
        before = env.snapshot()
        snap = env.snapshot()
        _deep_mutate(snap)
        # Mutating the snapshot must never alias into live environment state.
        assert env.snapshot() == before

    @pytest.mark.parametrize("cls", ALL_ENV_CLASSES)
    def test_unknown_tool_is_structured_error(self, cls):
        result = cls(seed=1).execute("definitely_not_a_tool", {})
        assert not result.ok
        assert result.error["code"] == ERR_UNKNOWN_TOOL

    @pytest.mark.parametrize("cls", ALL_ENV_CLASSES)
    def test_tool_specs_are_valid_and_neutral(self, cls):
        env = cls(seed=1)
        assert env.tools, "every environment must expose tools"
        for spec in env.tools:
            assert isinstance(spec, ToolSpec)
            assert spec.family == env.family
            assert ToolSpec.from_dict(spec.to_dict()) == spec
            # No hidden reward/verifier information in model-visible text.
            lowered = (spec.description + json.dumps(spec.parameters)).lower()
            for forbidden in ("reward", "verifier", "expected outcome", "ground truth",
                              "grpo", "benchmark"):
                assert forbidden not in lowered

    @pytest.mark.parametrize("cls", ALL_ENV_CLASSES)
    def test_current_date_is_fixed_not_system_clock(self, cls):
        state = cls(seed=1).snapshot()
        assert state.get("current_date") == "2026-03-02"


def _deep_mutate(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            _deep_mutate(v)
        obj["__mutated__"] = True
    elif isinstance(obj, list):
        for v in obj:
            _deep_mutate(v)
        obj.append("__mutated__")


def _apply_one_write(env):
    """Perform one family-specific state mutation (used by isolation/reset tests)."""
    if isinstance(env, CalendarEnvironment):
        env.execute("create_event", {"title": "X", "date": "2026-03-10",
                                     "start": "10:00", "end": "11:00"})
    elif isinstance(env, EmailEnvironment):
        env.execute("send_email", {"to": "someone@example.com", "subject": "hi", "body": "b"})
    elif isinstance(env, FilesEnvironment):
        env.execute("create_file", {"path": "/tmp-x/new.md", "content": "n"})
    elif isinstance(env, CrmEnvironment):
        env.execute("add_note", {"contact_id": env.state["contacts"][0]["id"], "text": "n"})
    elif isinstance(env, SqlEnvironment):
        # SQL is read-only by design: simulate "write" via fixture re-load check instead.
        env._state["tables"]["customers"].append(
            {"customer_id": "cus_x", "name": "X", "region": "north",
             "tier": "basic", "signup_date": "2026-01-01"})
    elif isinstance(env, TodoEnvironment):
        env.execute("create_task", {"title": "X"})
    else:  # pragma: no cover
        raise AssertionError(f"unhandled environment {type(env)}")


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------

class TestCalendar:
    @pytest.fixture
    def env(self, small_calendar_fixture):
        return CalendarEnvironment(fixture=small_calendar_fixture)

    def test_list_events_filters(self, env):
        out = env.execute("list_events", {"date": "2026-03-03"}).output
        assert out["count"] == 2
        out = env.execute("list_events", {"attendee": "alice@example.com"}).output
        assert [e["id"] for e in out["events"]] == ["evt_001"]

    def test_create_event_and_conflict_warning(self, env):
        res = env.execute("create_event", {"title": "Overlap", "date": "2026-03-03",
                                           "start": "09:15", "end": "09:45"})
        assert res.ok
        assert res.output["conflicts"] == ["evt_001"]
        assert env.execute("list_events", {}).output["count"] == 3

    def test_create_event_validates_times(self, env):
        res = env.execute("create_event", {"title": "Bad", "date": "2026-03-03",
                                           "start": "12:00", "end": "11:00"})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT
        res = env.execute("create_event", {"title": "Bad", "date": "2026-13-99",
                                           "start": "10:00", "end": "11:00"})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT

    def test_free_slots_computation(self, env):
        out = env.execute("find_free_slots",
                          {"date": "2026-03-03", "duration_minutes": 60}).output
        # Busy 09:00-09:30 and 11:00-12:00 inside 09:00-17:00 ->
        # gaps: 09:30-11:00 (90m) and 12:00-17:00 (300m).
        assert out["free_slots"] == [{"start": "09:30", "end": "11:00"},
                                     {"start": "12:00", "end": "17:00"}]

    def test_free_slots_respects_custom_window(self, env):
        out = env.execute("find_free_slots", {"date": "2026-03-03", "duration_minutes": 30,
                                              "work_start": "09:00", "work_end": "10:00"}).output
        assert out["free_slots"] == [{"start": "09:30", "end": "10:00"}]

    def test_find_conflicts(self, env):
        env.execute("create_event", {"title": "Clash", "date": "2026-03-03",
                                     "start": "11:30", "end": "12:30"})
        out = env.execute("find_conflicts", {"date": "2026-03-03"}).output
        assert out["count"] == 1
        pair = out["conflicts"][0]
        assert pair["overlap_start"] == "11:30" and pair["overlap_end"] == "12:00"

    def test_update_requires_fields_and_existing_id(self, env):
        res = env.execute("update_event", {"event_id": "evt_001"})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT
        res = env.execute("update_event", {"event_id": "evt_404", "title": "x"})
        assert not res.ok and res.error["code"] == ERR_NOT_FOUND

    def test_update_moves_and_reports_conflicts(self, env):
        res = env.execute("update_event", {"event_id": "evt_001", "start": "11:30", "end": "12:30"})
        assert res.ok and res.output["conflicts"] == ["evt_002"]

    def test_cancel_is_idempotent_and_frees_time(self, env):
        assert env.execute("cancel_event", {"event_id": "evt_001"}).ok
        again = env.execute("cancel_event", {"event_id": "evt_001"})
        assert again.ok and "already" in again.output["note"].lower()
        out = env.execute("find_free_slots",
                          {"date": "2026-03-03", "duration_minutes": 60}).output
        assert out["free_slots"][0] == {"start": "09:00", "end": "11:00"}

    def test_bad_fixture_raises(self):
        with pytest.raises(InvalidFixtureError):
            CalendarEnvironment(fixture={"events": [{"id": "e", "title": "t",
                                                    "date": "not-a-date", "start": "09:00",
                                                    "end": "10:00"}]})


# --------------------------------------------------------------------------
# Email
# --------------------------------------------------------------------------

class TestEmail:
    @pytest.fixture
    def env(self):
        fixture = {
            "current_date": "2026-03-02", "user_email": "user@example.com",
            "folders": ["inbox", "archive", "spam", "sent", "drafts"], "next_message_num": 3,
            "messages": [
                {"id": "msg_001", "from": "alice@example.com", "to": "user@example.com",
                 "subject": "Budget review", "body": "Q1 budget attached",
                 "date": "2026-03-01T09:00", "folder": "inbox", "read": False,
                 "in_reply_to": None},
                {"id": "msg_002", "from": "bob@example.com", "to": "user@example.com",
                 "subject": "Re: Budget review", "body": "Looks fine",
                 "date": "2026-03-01T10:00", "folder": "inbox", "read": True,
                 "in_reply_to": None},
            ],
        }
        return EmailEnvironment(fixture=fixture)

    def test_search_filters(self, env):
        out = env.execute("search_messages", {"query": "budget"}).output
        assert {m["id"] for m in out["messages"]} == {"msg_001", "msg_002"}
        out = env.execute("search_messages", {"unread_only": True}).output
        assert [m["id"] for m in out["messages"]] == ["msg_001"]
        out = env.execute("search_messages", {"from": "bob@"}).output
        assert [m["id"] for m in out["messages"]] == ["msg_002"]

    def test_read_marks_read_and_returns_body(self, env):
        out = env.execute("read_message", {"message_id": "msg_001"}).output
        assert out["body"] == "Q1 budget attached"
        assert env.execute("search_messages", {"unread_only": True}).output["count"] == 0

    def test_read_missing(self, env):
        res = env.execute("read_message", {"message_id": "msg_404"})
        assert not res.ok and res.error["code"] == ERR_NOT_FOUND

    def test_move_validates_folder(self, env):
        res = env.execute("move_message", {"message_id": "msg_001", "folder": "nowhere"})
        assert not res.ok and res.error["code"] == ERR_NOT_FOUND
        assert env.execute("move_message", {"message_id": "msg_001", "folder": "archive"}).ok

    def test_draft_reply_normalizes_subject(self, env):
        out = env.execute("draft_reply", {"message_id": "msg_002", "body": "Thanks!"}).output
        draft = out["draft"]
        assert draft["subject"] == "Re: Budget review"  # not "Re: Re: ..."
        assert draft["to"] == "bob@example.com"
        assert draft["folder"] == "drafts" and draft["in_reply_to"] == "msg_002"

    def test_send_email_validates_address(self, env):
        res = env.execute("send_email", {"to": "not-an-email", "subject": "s", "body": "b"})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT

    def test_send_email_stays_in_sandbox(self, env):
        out = env.execute("send_email", {"to": "alice@example.com", "subject": "Hi",
                                         "body": "There"}).output
        assert out["message"]["folder"] == "sent"
        assert "simulated" in out["note"].lower()

    def test_send_draft_flow(self, env):
        draft_id = env.execute("draft_reply",
                               {"message_id": "msg_001", "body": "ok"}).output["draft"]["id"]
        res = env.execute("send_draft", {"draft_id": draft_id})
        assert res.ok and res.output["message"]["folder"] == "sent"
        # Sending a non-draft fails.
        res = env.execute("send_draft", {"draft_id": "msg_001"})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT

    def test_read_flag_is_volatile(self):
        assert "read" in EmailEnvironment.volatile_keys


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------

class TestFiles:
    @pytest.fixture
    def env(self):
        return FilesEnvironment(seed=3)

    def test_list_and_directory_filter(self, env):
        out = env.execute("list_files", {"directory": "/notes"}).output
        assert out["count"] >= 3
        assert all(f["path"].startswith("/notes/") for f in out["files"])

    def test_read_and_search(self, env):
        out = env.execute("read_file", {"path": "/reports/q1-budget.md"}).output
        assert "45000" in out["content"]
        hits = env.execute("search_files", {"query": "45000"}).output["results"]
        assert {h["path"] for h in hits} >= {"/reports/q1-budget.md"}
        assert all(h["matches"][0]["line_number"] >= 1 for h in hits)

    def test_create_conflict_and_update_modes(self, env):
        res = env.execute("create_file", {"path": "/notes/todo.md", "content": "x"})
        assert not res.ok and res.error["code"] == ERR_ALREADY_EXISTS
        res = env.execute("update_file", {"path": "/notes/todo.md"})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT
        res = env.execute("update_file", {"path": "/notes/todo.md", "content": "a", "append": "b"})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT
        assert env.execute("update_file", {"path": "/notes/todo.md", "append": "REVIEWED"}).ok
        content = env.execute("read_file", {"path": "/notes/todo.md"}).output["content"]
        assert content.endswith("REVIEWED")

    def test_delete(self, env):
        assert env.execute("delete_file", {"path": "/notes/todo.md"}).ok
        assert not env.execute("delete_file", {"path": "/notes/todo.md"}).ok

    def test_missing_file(self, env):
        res = env.execute("read_file", {"path": "/nope.md"})
        assert not res.ok and res.error["code"] == ERR_NOT_FOUND


# --------------------------------------------------------------------------
# CRM
# --------------------------------------------------------------------------

class TestCrm:
    @pytest.fixture
    def env(self):
        return CrmEnvironment(seed=17)

    def test_search_and_get_company(self, env):
        contacts = env.execute("search_contacts", {"tag": "vip"}).output["contacts"]
        assert all("vip" in c["tags"] for c in contacts)
        res = env.execute("get_company", {"name": "Northwind"})
        assert res.ok and res.output["company"]["name"] == "Northwind Traders"
        res = env.execute("get_company", {"name": "nonexistent"})
        assert not res.ok and res.error["code"] == ERR_NOT_FOUND

    def test_get_company_requires_arguments(self, env):
        res = env.execute("get_company", {})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT

    def test_get_company_ambiguous_name(self, env):
        # 'a' matches several company names -> structured ambiguity with candidates.
        res = env.execute("get_company", {"name": "a"})
        assert not res.ok and res.error["code"] == ERR_AMBIGUOUS
        assert "matches" in res.error["message"]

    def test_create_contact_validation(self, env):
        existing = env.state["contacts"][0]["email"]
        res = env.execute("create_contact", {"name": "Dup", "email": existing})
        assert not res.ok and res.error["code"] == ERR_ALREADY_EXISTS
        res = env.execute("create_contact", {"name": "Bad", "email": "nope"})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT
        res = env.execute("create_contact",
                          {"name": "New", "email": "new@example.com", "company_id": "cmp_404"})
        assert not res.ok and res.error["code"] == ERR_NOT_FOUND
        assert env.execute("create_contact", {"name": "New", "email": "new@example.com"}).ok

    def test_update_contact_needs_a_field(self, env):
        cid = env.state["contacts"][0]["id"]
        res = env.execute("update_contact", {"contact_id": cid})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT

    def test_note_and_interaction_require_contact(self, env):
        res = env.execute("add_note", {"contact_id": "ctc_404", "text": "x"})
        assert not res.ok and res.error["code"] == ERR_NOT_FOUND
        res = env.execute("log_interaction",
                          {"contact_id": "ctc_404", "type": "call", "summary": "x"})
        assert not res.ok and res.error["code"] == ERR_NOT_FOUND

    def test_interaction_date_validation(self, env):
        cid = env.state["contacts"][0]["id"]
        res = env.execute("log_interaction",
                          {"contact_id": cid, "type": "call", "summary": "x",
                           "date": "March 1st"})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT

    def test_ids_are_sequential_and_unique(self, env):
        cid = env.state["contacts"][0]["id"]
        n1 = env.execute("add_note", {"contact_id": cid, "text": "a"}).output["note"]["id"]
        n2 = env.execute("add_note", {"contact_id": cid, "text": "b"}).output["note"]["id"]
        assert n1 != n2


# --------------------------------------------------------------------------
# Todo
# --------------------------------------------------------------------------

class TestTodo:
    @pytest.fixture
    def env(self):
        return TodoEnvironment(seed=23)

    def test_create_defaults_and_validation(self, env):
        res = env.execute("create_task", {"title": "New"})
        assert res.ok
        task = res.output["task"]
        assert task["status"] == "open" and task["priority"] == "medium"
        res = env.execute("create_task", {"title": "Bad", "deadline": "tomorrow"})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT

    def test_list_filters_and_overdue(self, env):
        overdue = env.execute("list_tasks", {"overdue_only": True}).output["tasks"]
        assert overdue, "default fixture guarantees overdue tasks"
        for t in overdue:
            assert t["deadline"] < env.state["current_date"]
            assert t["status"] in ("open", "in_progress", "blocked")
        by_owner = env.execute("list_tasks", {"assignee": "alice"}).output["tasks"]
        assert all(t["assignee"] == "alice" for t in by_owner)

    def test_status_transitions(self, env):
        tid = env.state["tasks"][0]["id"]
        assert env.execute("update_task_status", {"task_id": tid, "status": "done"}).ok
        assert env.execute("cancel_task", {"task_id": tid}).ok
        assert env.execute("update_task_status",
                           {"task_id": "tsk_404", "status": "done"}).error["code"] == ERR_NOT_FOUND

    def test_assign_and_deadline(self, env):
        tid = env.state["tasks"][0]["id"]
        assert env.execute("assign_task", {"task_id": tid, "assignee": "carol"}).ok
        assert env.execute("set_task_deadline",
                           {"task_id": tid, "deadline": "2026-04-01"}).ok
        res = env.execute("set_task_deadline", {"task_id": tid, "deadline": "04/01/2026"})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT

    def test_assignee_enum_is_schema_enforced(self, env):
        spec = next(s for s in env.tools if s.name == "todo.assign_task")
        issues = validate_against_schema({"task_id": "tsk_001", "assignee": "mallory"},
                                         spec.parameters)
        assert any(i.kind == "enum_violation" for i in issues)


# --------------------------------------------------------------------------
# SQL (functional; safety has its own file)
# --------------------------------------------------------------------------

class TestSqlFunctional:
    @pytest.fixture
    def env(self):
        return SqlEnvironment(seed=31)

    def test_list_and_describe(self, env):
        tables = {t["table"]: t for t in env.execute("list_tables", {}).output["tables"]}
        assert set(tables) == {"customers", "orders", "products"}
        assert tables["orders"]["row_count"] == 20
        desc = env.execute("describe_table", {"table": "products"}).output
        assert [c["name"] for c in desc["columns"]] == \
               ["product_id", "name", "category", "price", "stock"]
        assert desc["sample_row"] is not None
        assert env.execute("describe_table", {"table": "nope"}).error["code"] == ERR_NOT_FOUND

    def test_query_results(self, env):
        out = env.execute("run_query",
                          {"sql": "SELECT COUNT(*) AS n FROM orders"}).output
        assert out["columns"] == ["n"] and out["rows"] == [[20]]
        out = env.execute("run_query", {
            "sql": "SELECT name FROM customers WHERE region = ? ORDER BY name",
            "params": ["north"]}).output
        assert out["row_count"] >= 1

    def test_join_and_group(self, env):
        out = env.execute("run_query", {"sql": """
            SELECT c.name, SUM(o.quantity * o.unit_price) AS revenue
            FROM orders o JOIN customers c ON c.customer_id = o.customer_id
            WHERE o.status = 'completed'
            GROUP BY c.name ORDER BY revenue DESC LIMIT 3
        """}).output
        assert out["columns"] == ["name", "revenue"]
        assert len(out["rows"]) <= 3

    def test_truncation_flag(self):
        env = SqlEnvironment(seed=31, max_rows=5)
        out = env.execute("run_query", {"sql": "SELECT * FROM orders"}).output
        assert out["truncated"] is True and out["row_count"] == 5

    def test_syntax_error_is_structured(self, env):
        # Passes the SELECT prefix check but fails inside SQLite.
        res = env.execute("run_query", {"sql": "SELECT * FROMM orders"})
        assert not res.ok and res.error["code"] == ERR_INVALID_ARGUMENT
        # Fails the prefix check itself (typo'd keyword is simply not a SELECT).
        res = env.execute("run_query", {"sql": "SELCT * FROM orders"})
        assert not res.ok and res.error["code"] == ERR_UNSAFE


# --------------------------------------------------------------------------
# Tool-name / family consistency across the registry
# --------------------------------------------------------------------------

class TestToolNames:
    def test_qualified_names_unique(self, registry):
        names = [s.name for s in registry.tool_specs()]
        assert len(names) == len(set(names))
        assert len(names) >= 30  # six families, 3-6 tools each

    def test_families(self, registry):
        assert set(registry.families()) == {"calendar", "email", "files", "crm", "sql", "todo"}

    def test_email_addresses_are_example_domain(self, registry):
        for spec in registry.tool_specs():
            for email in EMAILS:
                assert email.endswith("@example.com")
