"""Reference-solution compiler: derive action scripts from expected outcomes.

The benchmark's expected outcomes are declarative and were computed from each
fixture at build time — which means for most template patterns the *correct
actions* can be compiled from the same specification:

- ``record_exists`` on a creatable collection  -> create-tool call with the
  ``where`` fields as arguments (missing event times are computed from the
  fixture via ``find_free_slots`` so ``no_overlaps`` expectations hold).
- ``field_equals``                              -> the family's update tool,
  merged per record; skipped when the fixture already satisfies the check.
- ``tool_called``                               -> emit the tool when valid
  arguments are derivable (``arguments_match``, ids parsed from the request,
  safe read-only defaults); creation synthesis otherwise covers it.
- ``answer_contains_all``                       -> first emit *evidence reads*
  whose executed outputs actually contain the expected strings (verified
  against the fixture at compile time), then a final answer embedding them.
  Retrieval trajectories therefore always observe the answer before stating
  it — the reference data never teaches hallucination.
- ``clarify`` / refusal tasks                   -> terminal action with text
  satisfying the expected phrase checks.

Runtime-dependent ids (draft -> send flows, create -> update flows) use
``{"$prev": "<path>"}`` placeholders resolved at execution time by
:class:`toolforge.data_generation.SolutionPolicy` from the latest tool output.

Honesty rules: the compiler returns ``None`` whenever it cannot guarantee a
pattern (coverage is reported per category, never forced), and *every*
compiled script is executed and verifier-checked downstream — only verified
episodes become reference trajectories. ``last_compilation_failure_reason()``
exposes diagnostics for coverage reports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from toolforge.environments.registry import ENVIRONMENT_CLASSES
from toolforge.schemas import validate_against_schema
from toolforge.tasks import (
    CAT_AMBIGUOUS,
    CAT_CONFLICT,
    CAT_IMPOSSIBLE,
    CAT_INVALID_TOOL,
    CAT_NO_ACTION,
    CAT_RETRIEVE_ACT,
    Task,
)
from toolforge.timeutils import minutes_to_time, time_to_minutes

_ID_RE = re.compile(r"\b(evt|msg|tsk|ctc|cmp|nte|act|ord|prd|cus)_(\d{3})\b")
_ID_PREFIX_TO_PARAM = {"evt": "event_id", "msg": "message_id", "tsk": "task_id",
                       "ctc": "contact_id", "cmp": "company_id"}
_WRITE_CHECKS = ("record_exists", "field_equals", "record_absent")

_INABILITY_TEXT = ("I cannot complete this request with the available tools: "
                   "there is no tool for that action, so it is not possible here.")
_READONLY_TEXT = ("I cannot do that: this database is read-only and only SELECT "
                  "queries are allowed, so the data cannot be modified.")
_MISSING_INFO_TEXT = ("I cannot act on this yet — which item do you mean? I did not find "
                      "enough information, and there is no clear match to proceed with.")
_EMPTY_MAILBOX_TEXT = ("I cannot find any such email — there is no matching message in the "
                       "mailbox. Which newsletter do you mean?")
_NO_SLOT_TEXT = ("I checked the calendar: there is no free slot available for that request, "
                 "so no event was created.")

_SAFE_CALL_ARGS: dict[str, dict[str, Any]] = {
    "calendar.list_events": {},
    "calendar.find_conflicts": {},
    "email.search_messages": {},
    "files.list_files": {},
    "crm.search_contacts": {},
    "sql.list_tables": {},
    "todo.list_tasks": {},
}

_FAMILY_LIST_TOOL = {
    "calendar": "calendar.list_events",
    "email": "email.search_messages",
    "files": "files.list_files",
    "crm": "crm.search_contacts",
    "sql": "sql.list_tables",
    "todo": "todo.list_tasks",
}

_last_reason: list[str] = [""]


def last_compilation_failure_reason() -> str:
    """Diagnostics for coverage reports (why the most recent compile failed)."""
    return _last_reason[0]


class CompilationError(Exception):
    """Internal signal: pattern not compilable (recorded in coverage stats)."""


@dataclass
class _Plan:
    reads: list[dict[str, Any]] = field(default_factory=list)
    writes: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    clarify_text: str | None = None


def _tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"action": "tool_call", "tool": tool_name, "arguments": arguments}


def _final(text: str) -> dict[str, Any]:
    return {"action": "final_answer", "content": text}


def _static_value(value: Any) -> Any:
    if isinstance(value, dict) and "op" in value:
        if value["op"] == "contains":
            return value["value"]
        raise CompilationError(f"cannot materialize op-value {value!r}")
    return value


def _natural_contains(kind: str, value: str) -> str:
    """Wrap a contains-op value into natural text that still contains it."""
    if kind == "note":
        return f"Regarding {value}."
    if kind == "email_body":
        return f"Hello — following up about {value}. Thanks!"
    return value


def _ids_in(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for prefix, num in _ID_RE.findall(text):
        param = _ID_PREFIX_TO_PARAM.get(prefix)
        if param:
            out[param] = f"{prefix}_{num}"
    return out


def _date_in(text: str) -> str | None:
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    return m.group(1) if m else None


def _output_text(result_output: Any) -> str:
    return json.dumps(result_output, ensure_ascii=False, default=str).lower()


def _add_unique(actions: list[dict[str, Any]], action: dict[str, Any]) -> None:
    if action not in actions:
        actions.append(action)


# --------------------------------------------------------------------------
# Evidence planning (answer_contains_all must be observed before being stated)
# --------------------------------------------------------------------------

def _coverage_after(env: Any, calls: list[dict[str, Any]], needed: set[str]) -> set[str]:
    remaining = set(needed)
    for call in calls:
        local = call["tool"].split(".", 1)[1]
        res = env.execute(local, call["arguments"])
        if res.ok:
            text = _output_text(res.output)
            remaining = {s for s in remaining if s not in text}
    return remaining


def _plan_evidence(task: Task, env: Any, strings: list[str],
                   any_groups: list[list[str]] | None = None) -> list[dict[str, Any]]:
    """Read-only calls whose executed outputs jointly contain all strings."""
    needed = {s.lower() for s in strings if s.lower() not in ("yes", "no")}
    calls: list[dict[str, Any]] = []
    if not needed and not any_groups:
        return calls

    def absorb(tool_name: str, call_args: dict[str, Any]) -> bool:
        local = tool_name.split(".", 1)[1]
        res = env.execute(local, call_args)
        if not res.ok:
            return False
        text = _output_text(res.output)
        found = {s for s in needed if s in text}
        if found:
            _add_unique(calls, _tool(tool_name, call_args))
            needed.difference_update(found)
        return not needed

    family = task.family
    if family == "calendar":
        date = _date_in(task.request)
        if not (date and absorb("calendar.list_events", {"date": date})):
            absorb("calendar.list_events", {})
        if needed and ("conflict" in task.request.lower() or "overlap" in task.request.lower()):
            absorb("calendar.find_conflicts", {"date": date} if date else {})
    elif family == "email":
        if "unread" in task.request.lower():
            absorb("email.search_messages", {"unread_only": True})
        else:
            absorb("email.search_messages", {})
        for msg in list(env.state["messages"]):
            if not needed:
                break
            blob = (msg["subject"] + " " + msg["body"]).lower()
            if any(s in blob for s in needed):
                absorb("email.read_message", {"message_id": msg["id"]})
    elif family == "files":
        absorb("files.list_files", {})
        for f in list(env.state["files"]):
            if not needed:
                break
            if any(s in f["content"].lower() or s in f["path"].lower() for s in needed):
                absorb("files.read_file", {"path": f["path"]})
    elif family == "crm":
        absorb("crm.search_contacts", {})
        for company in list(env.state["companies"]):
            if not needed:
                break
            if any(s in json.dumps(company).lower() for s in needed):
                absorb("crm.get_company", {"company_id": company["id"]})
    elif family == "todo":
        if "overdue" in task.request.lower():
            absorb("todo.list_tasks", {"overdue_only": True})
        absorb("todo.list_tasks", {})
    elif family == "sql":
        calls.extend(_plan_sql_evidence(
            env, needed,
            [[v.lower() for v in group] for group in (any_groups or [])]))
        needed = set()
    else:
        raise CompilationError(f"no evidence planner for family {family}")

    if needed:
        raise CompilationError(f"evidence incomplete; missing {sorted(needed)[:3]}")
    return calls


_SQL_CANDIDATE_QUERIES = [
    "SELECT COUNT(*) FROM orders",
    "SELECT COUNT(*) FROM customers",
    "SELECT COUNT(*) FROM products",
    "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
    "SELECT COUNT(*) FROM orders WHERE status = 'pending'",
    "SELECT COUNT(*) FROM orders WHERE status = 'shipped'",
    "SELECT COUNT(*) FROM orders WHERE order_date LIKE '2026-02%'",
    "SELECT COUNT(*) FROM customers WHERE tier = 'enterprise'",
    "SELECT COUNT(*) FROM products WHERE category = 'hardware'",
    "SELECT ROUND(SUM(quantity * unit_price), 2) FROM orders WHERE status = 'completed'",
    "SELECT ROUND(AVG(price), 2) FROM products WHERE category = 'software'",
    "SELECT ROUND(AVG(price), 2) FROM products WHERE category = 'hardware'",
    "SELECT region, COUNT(*) AS n FROM customers GROUP BY region ORDER BY n DESC, region LIMIT 1",
    "SELECT c.name, COUNT(o.order_id) AS n FROM orders o "
    "JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.name ORDER BY n DESC, c.name LIMIT 1",
    "SELECT p.name, SUM(o.quantity) AS q FROM orders o "
    "JOIN products p ON p.product_id = o.product_id WHERE o.status != 'cancelled' "
    "GROUP BY p.name ORDER BY q DESC, p.name LIMIT 1",
    "SELECT p.name, COUNT(*) AS n FROM orders o "
    "JOIN products p ON p.product_id = o.product_id GROUP BY p.name ORDER BY n DESC, p.name LIMIT 1",
    "SELECT c.name, c.tier, ROUND(SUM(o.quantity * o.unit_price), 2) AS spend "
    "FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE o.status != 'cancelled' "
    "GROUP BY c.name, c.tier ORDER BY spend DESC LIMIT 1",
    "SELECT order_id FROM orders WHERE status = 'pending' AND order_date > '2026-02-10' ORDER BY order_id",
    "SELECT name FROM customers WHERE tier = 'enterprise' ORDER BY name",
    "SELECT product_id FROM products WHERE stock < 10 ORDER BY product_id",
    "SELECT order_id FROM orders WHERE product_id IN "
    "(SELECT product_id FROM products WHERE stock < 10) ORDER BY order_id",
    "SELECT COUNT(*) FROM orders WHERE customer_id IN "
    "(SELECT customer_id FROM customers WHERE tier = 'enterprise')",
]


def _cell_variants(cell: Any) -> list[str]:
    if isinstance(cell, bool):
        return [str(cell).lower()]
    if isinstance(cell, float):
        out = [str(cell), f"{cell:.2f}", f"{cell:,.2f}"]
        if cell.is_integer():
            out.append(str(int(cell)))
        return [o.lower() for o in out]
    return [str(cell).lower()]


def _plan_sql_evidence(env: Any, needed: set[str],
                       any_groups: list[set[str]] | None = None) -> list[dict[str, Any]]:
    """Cover flat strings and at least one variant per any-group with real queries."""
    calls: list[dict[str, Any]] = []
    remaining = set(needed)
    remaining_groups = [set(g) for g in (any_groups or []) if g]
    for sql in _SQL_CANDIDATE_QUERIES:
        if not remaining and not remaining_groups:
            break
        res = env.execute("run_query", {"sql": sql})
        if not res.ok:
            continue
        cells = [v for row in res.output["rows"] for cell in row for v in _cell_variants(cell)]
        text = " ".join(cells) + " " + _output_text(res.output["columns"])
        found = {s for s in remaining if s in text}
        newly_covered_groups = [g for g in remaining_groups if any(v in text for v in g)]
        if found or newly_covered_groups:
            calls.append(_tool("sql.run_query", {"sql": sql}))
            remaining -= found
            for g in newly_covered_groups:
                remaining_groups.remove(g)
    if remaining or remaining_groups:
        raise CompilationError(
            f"sql evidence missing strings={sorted(remaining)[:3]} "
            f"groups={[sorted(g)[:2] for g in remaining_groups][:2]}")
    return calls


# --------------------------------------------------------------------------
# Required tool_called reads
# --------------------------------------------------------------------------

def _plan_required_reads(task: Task, env: Any, plan: _Plan) -> None:
    request_ids = _ids_in(task.request)
    specs = {s.name: s for s in env.tools}
    for check in task.expected.checks:
        if check["check"] != "tool_called":
            continue
        tool = check["tool"]
        spec = specs.get(tool)
        if spec is None:
            raise CompilationError(f"tool_called on unknown tool {tool}")
        args = dict(check.get("arguments_match") or {})
        if not args:
            args = dict(_SAFE_CALL_ARGS.get(tool, {}))
            for param in spec.parameters.get("properties", {}):
                if param in request_ids:
                    args[param] = request_ids[param]
            if tool == "crm.get_company" and not args:
                for company in env.state["companies"]:
                    if company["name"].lower() in task.request.lower():
                        args = {"name": company["name"]}
                        break
            if tool == "sql.describe_table" and "table" not in args:
                for table in env.state.get("tables", {}):
                    if re.search(rf"\b{table}\b", task.request):
                        args = {"table": table}
                        break
            if tool == "calendar.find_free_slots":
                date = _date_in(task.request)
                if date:
                    args = {"date": date, "duration_minutes": 30}
        if validate_against_schema(args, spec.parameters):
            continue  # cannot satisfy required args; creation synthesis may cover the tool
        _add_unique(plan.reads, _tool(tool, args))
    # Realism: an explicit "read message X" in the request becomes a read call.
    m = re.search(r"[Rr]ead message (msg_\d{3})", task.request)
    if m and task.family == "email":
        _add_unique(plan.reads, _tool("email.read_message", {"message_id": m.group(1)}))


# --------------------------------------------------------------------------
# Write synthesis
# --------------------------------------------------------------------------

def _collection(fixture: dict, family: str, coll: str) -> list[dict]:
    if "." in coll:
        family, coll = coll.split(".", 1)
    value = fixture.get(coll)
    if not isinstance(value, list):
        raise CompilationError(f"collection {coll!r} not list-valued for family {family}")
    return value


def _plan_writes(task: Task, env: Any, plan: _Plan) -> None:
    family = task.family
    checks = task.expected.checks
    fixture = env.snapshot()
    created_keys = {(c["collection"], _record_key(c.get("where", {})))
                    for c in checks if c["check"] == "record_exists"}

    # --- field_equals, grouped per target record; files.content merged ---
    updates: dict[tuple[str, str], dict[str, Any]] = {}
    content_appends: dict[str, list[str]] = {}
    content_overwrites: dict[str, str] = {}
    for check in checks:
        if check["check"] != "field_equals":
            continue
        where = check.get("where", {})
        rec_key = where.get("id") or where.get("path")
        if rec_key is None:
            raise CompilationError("field_equals without id/path in where")
        coll = check["collection"]
        if (coll, rec_key) in created_keys:
            continue  # paired with a creation; merged there
        matches = [r for r in _collection(fixture, family, coll)
                   if all(r.get(k) == v for k, v in where.items())]
        if len(matches) != 1:
            raise CompilationError(f"field_equals target not unique ({len(matches)} matches)")
        record = matches[0]
        field_name = check["field"]
        raw_value = check["value"]
        if family == "files" and field_name == "content":
            if isinstance(raw_value, dict) and raw_value.get("op") == "contains":
                content_appends.setdefault(rec_key, []).append(raw_value["value"])
            else:
                content_overwrites[rec_key] = raw_value
            continue
        value = _static_value(raw_value)
        if record.get(field_name) == value:
            continue  # already satisfied by the fixture
        updates.setdefault((coll, rec_key), {})[field_name] = value

    for (coll, rec_key), fields_ in updates.items():
        plan.writes.extend(_update_calls(family, coll, rec_key, fields_))
    for path, pieces in content_appends.items():
        if path in content_overwrites:
            continue
        plan.writes.append(_tool("files.update_file",
                                 {"path": path, "append": "\n".join(f"- {p}" for p in pieces)}))
    for path, content in content_overwrites.items():
        plan.writes.append(_tool("files.update_file", {"path": path, "content": content}))

    # --- record_exists -> creation ---
    for check in checks:
        if check["check"] == "record_exists":
            plan.writes.extend(_creation_calls(task, env, fixture, check))

    # --- create-then-move pattern: a required update_event on a just-created
    # record means the request wanted an intermediate state ("create at 13:00,
    # then move to 15:00"). Rewrite create+update so both calls appear. ---
    required_tools = {c["tool"] for c in checks if c["check"] == "tool_called"}
    if family == "calendar" and "calendar.update_event" in required_tools:
        times = re.findall(r"\b(?:[01]\d|2[0-3]):[0-5]\d\b", task.request)
        for i, action in enumerate(plan.writes):
            if (action["tool"] == "calendar.create_event" and len(times) >= 4
                    and action["arguments"].get("start") == times[2]
                    and action["arguments"].get("end") == times[3]):
                action["arguments"] = {**action["arguments"],
                                       "start": times[0], "end": times[1]}
                plan.writes.insert(i + 1, _tool("calendar.update_event", {
                    "event_id": {"$prev": "event.id"},
                    "start": times[2], "end": times[3]}))
                break

    # --- record_absent -> deletion ---
    for check in checks:
        if check["check"] != "record_absent":
            continue
        where = check.get("where", {})
        coll = check["collection"]
        existing = [r for r in _collection(fixture, family, coll)
                    if all(r.get(k) == v for k, v in where.items())]
        if not existing:
            continue
        if family == "files" and "path" in where:
            plan.writes.append(_tool("files.delete_file", {"path": where["path"]}))
        else:
            raise CompilationError("record_absent on non-files collection")

    # --- record_count: only enforce when it is the whole task (no writes) ---
    if not any(c["check"] in _WRITE_CHECKS for c in checks):
        for check in checks:
            if check["check"] != "record_count":
                continue
            coll = check["collection"]
            where = check.get("where", {})
            n = sum(1 for r in _collection(fixture, family, coll)
                    if all(_loose_eq(r.get(k), v) for k, v in where.items()))
            op, expected = check.get("op", "=="), check["value"]
            ok = {"==": n == expected, "!=": n != expected, ">=": n >= expected,
                  "<=": n <= expected, ">": n > expected, "<": n < expected}[op]
            if not ok:
                raise CompilationError(
                    "record_count requires state changes the compiler cannot infer")


def _record_key(where: dict[str, Any]) -> Any:
    return where.get("id") or where.get("path")


def _loose_eq(actual: Any, expected: Any) -> bool:
    if isinstance(actual, list) and not isinstance(expected, list):
        return expected in actual
    return actual == expected


def _update_calls(family: str, coll: str, rec_key: str, fields_: dict[str, Any]) -> list[dict]:
    fields_ = dict(fields_)
    if family == "calendar" and coll == "events":
        calls = []
        status = fields_.pop("status", None)
        if status == "cancelled":
            calls.append(_tool("calendar.cancel_event", {"event_id": rec_key}))
        elif status is not None:
            raise CompilationError("cannot set non-cancelled status via update")
        if fields_:
            calls.append(_tool("calendar.update_event", {"event_id": rec_key, **fields_}))
        return calls
    if family == "email" and coll == "messages":
        if set(fields_) == {"folder"}:
            return [_tool("email.move_message",
                          {"message_id": rec_key, "folder": fields_["folder"]})]
        raise CompilationError(f"email field_equals on {sorted(fields_)}")
    if family == "crm" and coll == "contacts":
        return [_tool("crm.update_contact", {"contact_id": rec_key, **fields_})]
    if family == "todo" and coll == "tasks":
        calls = []
        if "status" in fields_:
            status = fields_.pop("status")
            if status == "cancelled":
                calls.append(_tool("todo.cancel_task", {"task_id": rec_key}))
            else:
                calls.append(_tool("todo.update_task_status",
                                   {"task_id": rec_key, "status": status}))
        if "assignee" in fields_:
            calls.append(_tool("todo.assign_task",
                               {"task_id": rec_key, "assignee": fields_.pop("assignee")}))
        if "deadline" in fields_:
            calls.append(_tool("todo.set_task_deadline",
                               {"task_id": rec_key, "deadline": fields_.pop("deadline")}))
        if fields_:
            raise CompilationError(f"todo field_equals on {sorted(fields_)}")
        return calls
    raise CompilationError(f"no update synthesis for {family}.{coll}")


def _creation_calls(task: Task, env: Any, fixture: dict, check: dict) -> list[dict]:
    family = task.family
    coll = check["collection"]
    where = dict(check.get("where", {}))
    existing = [r for r in _collection(fixture, family, coll)
                if all(_loose_eq(r.get(k), _static_value(v)) for k, v in where.items())]
    if existing:
        return []  # fixture already satisfies the expectation

    if family == "calendar" and coll == "events":
        args: dict[str, Any] = {k: _static_value(where[k])
                                for k in ("title", "date", "attendees", "location", "notes")
                                if k in where and where[k] is not None}
        if "start" in where and "end" in where:
            args["start"], args["end"] = where["start"], where["end"]
        else:
            lowered = task.request.lower()
            duration = 60 if ("one-hour" in lowered or "one hour" in lowered
                              or "1 hour" in lowered) else 30
            slots = env.execute("find_free_slots",
                                {"date": where["date"], "duration_minutes": duration}
                                ).output["free_slots"]
            if not slots:
                raise CompilationError("no free slot for conflict-style creation")
            args["start"] = slots[0]["start"]
            args["end"] = minutes_to_time(time_to_minutes(slots[0]["start"]) + duration)
        return [_tool("calendar.create_event", args)]

    if family == "email" and coll == "messages":
        folder = where.get("folder")
        body_raw = where.get("body")
        body = (_natural_contains("email_body", body_raw["value"])
                if isinstance(body_raw, dict) and body_raw.get("op") == "contains"
                else body_raw)
        if folder == "drafts":
            return [_tool("email.draft_reply",
                          {"message_id": where["in_reply_to"],
                           "body": body or "Thanks — following up."})]
        if folder == "sent":
            if "in_reply_to" in where:
                return [
                    _tool("email.draft_reply",
                          {"message_id": where["in_reply_to"], "body": body or "Thanks!"}),
                    _tool("email.send_draft", {"draft_id": {"$prev": "draft.id"}}),
                ]
            subject_raw = where.get("subject", "Update")
            subject = (subject_raw["value"] if isinstance(subject_raw, dict)
                       and subject_raw.get("op") == "contains" else subject_raw)
            return [_tool("email.send_email",
                          {"to": where["to"], "subject": subject, "body": body or "Hello"})]
        raise CompilationError("email record_exists without sent/drafts folder")

    if family == "files" and coll == "files":
        content = where.get("content")
        if content is None:
            pieces = [c["value"]["value"] for c in task.expected.checks
                      if c["check"] == "field_equals"
                      and c.get("where", {}).get("path") == where.get("path")
                      and c["field"] == "content"
                      and isinstance(c.get("value"), dict) and c["value"].get("op") == "contains"]
            exact = [c["value"] for c in task.expected.checks
                     if c["check"] == "field_equals"
                     and c.get("where", {}).get("path") == where.get("path")
                     and c["field"] == "content" and not isinstance(c.get("value"), dict)]
            if exact:
                content = exact[0]
            elif pieces:
                content = "\n".join(f"- {p}" for p in pieces) + "\n"
            else:
                raise CompilationError("files creation without content constraint")
        else:
            content = _static_value(content)
        args = {"path": where["path"], "content": content}
        if where.get("tags") is not None:
            args["tags"] = where["tags"]
        return [_tool("files.create_file", args)]

    if family == "crm":
        if coll == "notes":
            text_raw = where["text"]
            text = (_natural_contains("note", text_raw["value"])
                    if isinstance(text_raw, dict) and text_raw.get("op") == "contains"
                    else text_raw)
            return [_tool("crm.add_note", {"contact_id": where["contact_id"], "text": text})]
        if coll == "interactions":
            args = {"contact_id": where["contact_id"], "type": where["type"],
                    "summary": _static_value(where["summary"])}
            if "date" in where:
                args["date"] = where["date"]
            return [_tool("crm.log_interaction", args)]
        if coll == "contacts":
            args = {k: where[k] for k in ("name", "email", "company_id", "title", "phone",
                                          "owner", "tags") if k in where and where[k] is not None}
            return [_tool("crm.create_contact", args)]
        raise CompilationError(f"crm creation for {coll}")

    if family == "todo" and coll == "tasks":
        args = {k: where[k] for k in ("title", "description", "assignee", "deadline",
                                      "priority", "project")
                if k in where and where[k] is not None}
        calls = [_tool("todo.create_task", args)]
        status = where.get("status", "open")
        if status in ("done", "in_progress", "blocked"):
            calls.append(_tool("todo.update_task_status",
                               {"task_id": {"$prev": "task.id"}, "status": status}))
        elif status == "cancelled":
            calls.append(_tool("todo.cancel_task", {"task_id": {"$prev": "task.id"}}))
        elif status != "open":
            raise CompilationError(f"unknown todo status {status!r}")
        return calls

    raise CompilationError(f"no creation synthesis for {family}.{coll}")


# --------------------------------------------------------------------------
# Discovery, final texts, assembly
# --------------------------------------------------------------------------

def _maybe_prepend_discovery(task: Task, plan: _Plan) -> None:
    """Insert a lookup when writes reference record ids/paths the request omits."""
    mentioned = set(_ids_in(task.request).values())
    mentioned |= set(re.findall(r"(/[A-Za-z0-9_\-./]+\.[A-Za-z0-9]+)", task.request))
    needs_discovery = False
    for action in plan.writes:
        for value in action["arguments"].values():
            if isinstance(value, str) and (
                    re.match(r"^(evt|msg|tsk|ctc|nte|act)_\d{3}$", value) and value not in mentioned
                    or value.startswith("/") and value not in mentioned):
                needs_discovery = True
    if needs_discovery:
        list_tool = _FAMILY_LIST_TOOL[task.family]
        _add_unique(plan.reads, _tool(list_tool, dict(_SAFE_CALL_ARGS.get(list_tool, {}))))
        plan.reads.sort(key=lambda a: 0 if a["tool"] == list_tool else 1)  # discovery first


def _clarify_question(task: Task) -> str:
    for tag in task.tags:
        if tag.startswith(("missing:", "ambiguous:")):
            items = tag.split(":", 1)[1].replace("_", " ").split(",")
            return ("Could you clarify " + ", ".join(items) + "? I want to make sure I "
                    "change the right thing before acting.")
    return ("Could you clarify exactly what you want me to do? I want to make sure I act "
            "on the right item before changing anything.")


def _default_read(task: Task, env: Any) -> dict[str, Any]:
    """A family-default observation call so retrieval answers are grounded."""
    family = task.family
    date = _date_in(task.request)
    if family == "calendar":
        return _tool("calendar.list_events", {"date": date} if date else {})
    if family == "email":
        return _tool("email.search_messages", {})
    if family == "files":
        return _tool("files.list_files", {})
    if family == "crm":
        return _tool("crm.search_contacts", {})
    if family == "todo":
        return _tool("todo.list_tasks", {})
    if family == "sql":
        return _tool("sql.run_query", {"sql": _SQL_CANDIDATE_QUERIES[0]})
    raise CompilationError(f"no default read for family {family}")


def _report_text(answer_all: list[str], answer_any: list[list[str]]) -> str:
    parts = []
    for lst in answer_any:
        lowered = [s.lower() for s in lst]
        if "yes" in lowered:
            parts.append("Yes.")
        elif "no" in lowered:
            parts.append("No.")
        elif any(s in lowered for s in ("cannot", "can't", "unable", "no tool")):
            parts.append(_INABILITY_TEXT)
        else:
            parts.append(lst[0].capitalize() + ".")
    if answer_all:
        parts.append("Here is what you asked for: " + ", ".join(answer_all) + ".")
    return " ".join(parts) if parts else "Done."


def _completion_text(plan: _Plan) -> str:
    verbs = [a["tool"].split(".", 1)[1].replace("_", " ") for a in plan.writes]
    if not verbs:
        return "Done — no changes were required."
    return "Done: " + ", ".join(dict.fromkeys(verbs)) + "."


def _refusal_text(task: Task) -> str:
    if "unsafe_request" in task.tags or "write_blocked" in task.tags:
        return _READONLY_TEXT
    if "empty_state" in task.tags:
        return _EMPTY_MAILBOX_TEXT
    if task.category in (CAT_IMPOSSIBLE, CAT_INVALID_TOOL):
        return _INABILITY_TEXT
    return _MISSING_INFO_TEXT


def _assemble(task: Task, plan: _Plan, answer_all: list[str],
              answer_any: list[list[str]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if plan.clarify_text:
        actions.append({"action": "clarify", "question": plan.clarify_text})
    else:
        actions.extend(plan.reads)
        actions.extend(plan.writes)
        text = plan.final_text
        lowered = text.lower()
        for s in answer_all:
            if s.lower() not in lowered:
                raise CompilationError(f"compiled answer misses contains_all string {s!r}")
        for lst in answer_any:
            if not any(s.lower() in lowered for s in lst):
                raise CompilationError(f"compiled answer misses contains_any list {lst[:2]}...")
        actions.append(_final(text))
    required = {c["tool"] for c in task.expected.checks if c["check"] == "tool_called"}
    emitted = {a["tool"] for a in actions if a["action"] == "tool_call"}
    if not required <= emitted:
        raise CompilationError(f"missing required tool calls: {sorted(required - emitted)}")
    if len(actions) > task.max_steps:
        raise CompilationError(f"script needs {len(actions)} turns > max_steps {task.max_steps}")
    return actions


# --------------------------------------------------------------------------
# Top-level compiler
# --------------------------------------------------------------------------

def compile_reference_solution(task: Task) -> list[dict[str, Any]] | None:
    """Compile a task's reference action script, or ``None`` when not compilable."""
    global _last_reason
    if len(task.tool_families) != 1:
        _last_reason[0] = "multi-family task"
        return None
    env = ENVIRONMENT_CLASSES[task.family](fixture=json.loads(json.dumps(task.fixture)))
    expected = task.expected
    plan = _Plan()
    try:
        answer_all = [s for c in expected.checks if c["check"] == "answer_contains_all"
                      for s in c["strings"]]
        answer_any = [c["strings"] for c in expected.checks if c["check"] == "answer_contains_any"]
        has_writes = any(c["check"] in _WRITE_CHECKS for c in expected.checks)

        # 1) Clarification tasks.
        if expected.terminal_action == "clarify":
            if task.category != CAT_AMBIGUOUS:
                raise CompilationError("clarify terminal on non-ambiguous task")
            plan.clarify_text = _clarify_question(task)
            return _assemble(task, plan, [], [])

        # 2) Read-only / refusal tasks (no state-change checks).
        if not has_writes:
            _plan_required_reads(task, env, plan)
            if task.category in (CAT_IMPOSSIBLE, CAT_INVALID_TOOL, CAT_NO_ACTION):
                plan.final_text = _refusal_text(task)
            elif task.category == CAT_CONFLICT:
                plan.final_text = _NO_SLOT_TEXT
            else:
                flat = [s for s in answer_all if s.lower() not in ("yes", "no")]
                # Numeric/variant any-groups need query evidence too (a stated
                # aggregate must be derived from an executed query, never guessed).
                groups = [lst for lst in answer_any
                          if not any(s.lower() in ("yes", "no") for s in lst)
                          and not any(s.lower() in ("cannot", "can't", "unable", "no tool",
                                                    "read-only", "read only") for s in lst)]
                remaining = {s.lower() for s in flat}
                if plan.reads:
                    remaining = _coverage_after(env, plan.reads, remaining)
                if remaining or (task.family == "sql" and groups):
                    plan.reads.extend(_plan_evidence(task, env, sorted(remaining),
                                                     any_groups=groups))
                if task.category == CAT_RETRIEVE_ACT and not plan.reads:
                    # Observe-before-answer: retrieval tasks always get a read call.
                    plan.reads.append(_default_read(task, env))
                plan.final_text = _report_text(answer_all, answer_any)
            return _assemble(task, plan, answer_all, answer_any)

        # 3) State-changing tasks.
        _plan_required_reads(task, env, plan)
        _plan_writes(task, env, plan)
        _maybe_prepend_discovery(task, plan)
        if answer_all:
            remaining = _coverage_after(env, plan.reads,
                                        {s.lower() for s in answer_all
                                         if s.lower() not in ("yes", "no")})
            if remaining:
                plan.reads.extend(_plan_evidence(task, env, sorted(remaining)))
            plan.final_text = _completion_text(plan) + " " + _report_text(answer_all, answer_any)
        else:
            plan.final_text = _completion_text(plan)
        return _assemble(task, plan, answer_all, answer_any)
    except (CompilationError, AssertionError, KeyError, TypeError, ValueError) as exc:
        _last_reason[0] = f"{type(exc).__name__}: {exc}"
        return None
