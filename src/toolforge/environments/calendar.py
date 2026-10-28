"""Calendar environment: events, free slots, conflicts. All state in-memory.

State shape::

    {
      "current_date": "2026-03-02",
      "user_email": "user@example.com",
      "events": [{"id","title","date","start","end","attendees","location","notes","status"}],
      "next_event_num": int
    }

``status`` is "scheduled" or "cancelled"; only scheduled events block time.
``create_event``/``update_event`` succeed even when they overlap other events
but report the conflicting ids — detecting conflicts before booking is the
agent's job (and the point of several benchmark categories).
"""

from __future__ import annotations

import random
from typing import Any, ClassVar

from toolforge.environments.base import (
    ERR_INVALID_ARGUMENT,
    ERR_NOT_FOUND,
    Environment,
    InvalidFixtureError,
    ToolResult,
    require,
)
from toolforge.environments.pools import EMAILS, USER_EMAIL, pick, sample
from toolforge.schemas import ToolSpec
from toolforge.timeutils import (
    DEFAULT_CURRENT_DATE,
    TimeFormatError,
    add_days,
    find_free_ranges,
    minutes_to_time,
    parse_date,
    parse_time,
    ranges_overlap,
    time_to_minutes,
)

EVENT_TITLES = [
    "Team standup", "Design review", "1:1 with manager", "Sprint planning",
    "Customer call", "Budget review", "Interview panel", "Product sync",
    "Architecture discussion", "Weekly report", "Vendor demo", "Retro",
]

LOCATIONS = ["Room A", "Room B", "Zoom", "HQ lobby", ""]


def _date_schema(desc: str) -> dict:
    return {"type": "string", "description": desc, "pattern": r"^\d{4}-\d{2}-\d{2}$"}


def _time_schema(desc: str) -> dict:
    return {"type": "string", "description": desc, "pattern": r"^([01]\d|2[0-3]):[0-5]\d$"}


class CalendarEnvironment(Environment):
    family: ClassVar[str] = "calendar"
    volatile_keys: ClassVar[frozenset[str]] = frozenset()  # no read-side effects

    # ------------------------------------------------------------------ state

    def default_state(self, seed: int) -> dict[str, Any]:
        rng = random.Random(seed)
        events: list[dict[str, Any]] = []
        # 9 events spread over the next 7 days.
        for i in range(9):
            day = add_days(DEFAULT_CURRENT_DATE, i % 7)
            start_hour = 9 + (i * 5 + rng.randrange(3)) % 7  # 09:00..15:00-ish
            duration = pick(rng, [30, 45, 60, 60, 90])
            start = start_hour * 60 + pick(rng, [0, 0, 30])
            end = min(start + duration, 17 * 60)
            attendees = sample(rng, EMAILS, rng.randrange(1, 3))
            events.append({
                "id": f"evt_{i + 1:03d}",
                "title": EVENT_TITLES[i % len(EVENT_TITLES)],
                "date": day,
                "start": minutes_to_time(start),
                "end": minutes_to_time(end),
                "attendees": attendees,
                "location": pick(rng, LOCATIONS),
                "notes": "",
                "status": "scheduled",
            })
        # Guarantee one discoverable conflict: a deliberate overlap tomorrow.
        tomorrow = add_days(DEFAULT_CURRENT_DATE, 1)
        events.append({
            "id": "evt_010",
            "title": "Design review (extended)",
            "date": tomorrow,
            "start": "10:00", "end": "11:30",
            "attendees": [EMAILS[0]],
            "location": "Room B", "notes": "", "status": "scheduled",
        })
        events.append({
            "id": "evt_011",
            "title": "Customer call",
            "date": tomorrow,
            "start": "10:30", "end": "11:00",
            "attendees": [EMAILS[1]],
            "location": "Zoom", "notes": "", "status": "scheduled",
        })
        events.sort(key=lambda e: (e["date"], e["start"], e["id"]))
        return {
            "current_date": DEFAULT_CURRENT_DATE,
            "user_email": USER_EMAIL,
            "events": events,
            "next_event_num": len(events) + 1,
        }

    def normalize_state(self, fixture: dict[str, Any]) -> dict[str, Any]:
        require(isinstance(fixture, dict), "calendar fixture must be an object")
        state: dict[str, Any] = {
            "current_date": fixture.get("current_date", DEFAULT_CURRENT_DATE),
            "user_email": fixture.get("user_email", USER_EMAIL),
            "events": [],
            "next_event_num": 1,
        }
        require(isinstance(state["current_date"], str), "current_date must be a string")
        raw_events = fixture.get("events", [])
        require(isinstance(raw_events, list), "events must be a list")
        seen_ids: set[str] = set()
        for i, ev in enumerate(raw_events):
            require(isinstance(ev, dict), f"events[{i}] must be an object")
            for key in ("id", "title", "date", "start", "end"):
                require(isinstance(ev.get(key), str) and ev.get(key) != "",
                        f"events[{i}] missing/invalid string field {key!r}")
            require(ev["id"] not in seen_ids, f"duplicate event id {ev['id']!r}")
            seen_ids.add(ev["id"])
            try:
                parse_date(ev["date"])
                parse_time(ev["start"])
                parse_time(ev["end"])
            except TimeFormatError as exc:
                raise InvalidFixtureError(f"events[{i}]: {exc}") from exc
            attendees = ev.get("attendees", [])
            require(isinstance(attendees, list) and all(isinstance(a, str) for a in attendees),
                    f"events[{i}].attendees must be a list of strings")
            status = ev.get("status", "scheduled")
            require(status in ("scheduled", "cancelled"),
                    f"events[{i}].status must be scheduled|cancelled, got {status!r}")
            state["events"].append({
                "id": ev["id"],
                "title": ev["title"],
                "date": ev["date"],
                "start": ev["start"],
                "end": ev["end"],
                "attendees": list(attendees),
                "location": str(ev.get("location", "")),
                "notes": str(ev.get("notes", "")),
                "status": status,
            })
        next_num = fixture.get("next_event_num")
        if next_num is None:
            next_num = len(state["events"]) + 1
        require(isinstance(next_num, int) and next_num >= 1, "next_event_num must be a positive int")
        state["next_event_num"] = next_num
        state["events"].sort(key=lambda e: (e["date"], e["start"], e["id"]))
        return state

    # ------------------------------------------------------------------ tools

    tool_specs: ClassVar[list[ToolSpec]] = [
        ToolSpec(
            "calendar.list_events",
            "List calendar events, optionally filtered by date and/or attendee email. "
            "Returns events sorted by date and start time.",
            {"type": "object", "properties": {
                "date": _date_schema("Only events on this date (YYYY-MM-DD). Omit for all dates."),
                "attendee": {"type": "string", "description": "Only events whose attendees include this email."},
                "status": {"type": "string", "enum": ["scheduled", "cancelled"],
                           "description": "Filter by status. Omit for all."},
            }, "additionalProperties": False},
        ),
        ToolSpec(
            "calendar.find_free_slots",
            "Find free time slots on a date inside a working window. Returns maximal "
            "gaps of at least duration_minutes between scheduled events.",
            {"type": "object", "properties": {
                "date": _date_schema("Date to search (YYYY-MM-DD)."),
                "duration_minutes": {"type": "integer", "minimum": 5, "maximum": 480,
                                     "description": "Minimum slot length in minutes."},
                "work_start": _time_schema("Start of working window (default 09:00)."),
                "work_end": _time_schema("End of working window (default 17:00)."),
            }, "required": ["date", "duration_minutes"], "additionalProperties": False},
        ),
        ToolSpec(
            "calendar.create_event",
            "Create a new event. Succeeds even if it overlaps other events; the result "
            "lists conflicting event ids so the caller can react.",
            {"type": "object", "properties": {
                "title": {"type": "string", "minLength": 1, "maxLength": 120},
                "date": _date_schema("Event date (YYYY-MM-DD)."),
                "start": _time_schema("Start time (HH:MM)."),
                "end": _time_schema("End time (HH:MM), must be after start."),
                "attendees": {"type": "array", "items": {"type": "string"}, "maxItems": 20,
                              "description": "Attendee emails (optional)."},
                "location": {"type": "string", "maxLength": 80},
                "notes": {"type": "string", "maxLength": 500},
            }, "required": ["title", "date", "start", "end"], "additionalProperties": False},
        ),
        ToolSpec(
            "calendar.update_event",
            "Update fields of an existing event. Provide only the fields to change; "
            "at least one field is required. Returns conflicts for the new time if any.",
            {"type": "object", "properties": {
                "event_id": {"type": "string", "minLength": 1},
                "title": {"type": "string", "minLength": 1, "maxLength": 120},
                "date": _date_schema("New date (YYYY-MM-DD)."),
                "start": _time_schema("New start time (HH:MM)."),
                "end": _time_schema("New end time (HH:MM)."),
                "attendees": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                "location": {"type": "string", "maxLength": 80},
                "notes": {"type": "string", "maxLength": 500},
            }, "required": ["event_id"], "additionalProperties": False},
        ),
        ToolSpec(
            "calendar.cancel_event",
            "Cancel an event by id. The record remains with status 'cancelled'.",
            {"type": "object", "properties": {"event_id": {"type": "string", "minLength": 1}},
             "required": ["event_id"], "additionalProperties": False},
        ),
        ToolSpec(
            "calendar.find_conflicts",
            "Detect pairs of scheduled events that overlap in time, optionally "
            "filtered by date and/or attendee.",
            {"type": "object", "properties": {
                "date": _date_schema("Only check this date (YYYY-MM-DD)."),
                "attendee": {"type": "string", "description": "Only events including this attendee email."},
            }, "additionalProperties": False},
        ),
    ]

    # ------------------------------------------------------------ tool bodies

    def _find_event(self, event_id: Any) -> dict[str, Any] | None:
        if not isinstance(event_id, str):
            return None
        for ev in self._state["events"]:
            if ev["id"] == event_id:
                return ev
        return None

    def _validate_dt(self, args: dict, keys: tuple[str, ...]) -> ToolResult | None:
        """Return a failure result if any present date/time key is malformed."""
        try:
            for key in keys:
                if key in args and args[key] is not None:
                    if key == "date":
                        parse_date(args[key])
                    else:
                        parse_time(args[key])
        except TimeFormatError as exc:
            return self._fail(ERR_INVALID_ARGUMENT, str(exc))
        return None

    def _conflicts_for(self, candidate: dict[str, Any]) -> list[str]:
        c_start, c_end = time_to_minutes(candidate["start"]), time_to_minutes(candidate["end"])
        out = []
        for ev in self._state["events"]:
            if ev["id"] == candidate["id"] or ev["status"] != "scheduled" or ev["date"] != candidate["date"]:
                continue
            if ranges_overlap(c_start, c_end, time_to_minutes(ev["start"]), time_to_minutes(ev["end"])):
                out.append(ev["id"])
        return sorted(out)

    def _tool_list_events(self, args: dict[str, Any]) -> ToolResult:
        if (bad := self._validate_dt(args, ("date",))):
            return bad
        events = self._state["events"]
        if args.get("date"):
            events = [e for e in events if e["date"] == args["date"]]
        if args.get("attendee"):
            events = [e for e in events if args["attendee"] in e["attendees"]]
        if args.get("status"):
            events = [e for e in events if e["status"] == args["status"]]
        return self._ok({"events": events, "count": len(events)})

    def _tool_find_free_slots(self, args: dict[str, Any]) -> ToolResult:
        if (bad := self._validate_dt(args, ("date", "work_start", "work_end"))):
            return bad
        date = args["date"]
        duration = args["duration_minutes"]
        work_start = args.get("work_start", "09:00")
        work_end = args.get("work_end", "17:00")
        ws, we = time_to_minutes(work_start), time_to_minutes(work_end)
        if we <= ws:
            return self._fail(ERR_INVALID_ARGUMENT,
                              f"work_end ({work_end}) must be after work_start ({work_start})")
        busy = [
            (time_to_minutes(e["start"]), time_to_minutes(e["end"]))
            for e in self._state["events"]
            if e["date"] == date and e["status"] == "scheduled"
        ]
        free = find_free_ranges(busy, ws, we, duration)
        return self._ok({
            "date": date,
            "duration_minutes": duration,
            "free_slots": [{"start": minutes_to_time(s), "end": minutes_to_time(e)} for s, e in free],
        })

    def _tool_create_event(self, args: dict[str, Any]) -> ToolResult:
        if (bad := self._validate_dt(args, ("date", "start", "end"))):
            return bad
        if time_to_minutes(args["end"]) <= time_to_minutes(args["start"]):
            return self._fail(ERR_INVALID_ARGUMENT,
                              f"end ({args['end']}) must be after start ({args['start']})")
        event = {
            "id": f"evt_{self._state['next_event_num']:03d}",
            "title": args["title"],
            "date": args["date"],
            "start": args["start"],
            "end": args["end"],
            "attendees": list(args.get("attendees", [])),
            "location": args.get("location", ""),
            "notes": args.get("notes", ""),
            "status": "scheduled",
        }
        self._state["next_event_num"] += 1
        conflicts = self._conflicts_for(event)
        self._state["events"].append(event)
        self._state["events"].sort(key=lambda e: (e["date"], e["start"], e["id"]))
        return self._ok({"event": event, "conflicts": conflicts,
                         "warning": "Event overlaps existing events." if conflicts else None})

    def _tool_update_event(self, args: dict[str, Any]) -> ToolResult:
        if (bad := self._validate_dt(args, ("date", "start", "end"))):
            return bad
        updatable = ("title", "date", "start", "end", "attendees", "location", "notes")
        changes = {k: v for k, v in args.items() if k in updatable and v is not None}
        if not changes:
            return self._fail(ERR_INVALID_ARGUMENT,
                              "update_event requires at least one field to change "
                              f"among {list(updatable)}")
        event = self._find_event(args["event_id"])
        if event is None:
            return self._fail(ERR_NOT_FOUND, f"No event with id {args['event_id']!r}")
        candidate = {**event, **changes}
        if time_to_minutes(candidate["end"]) <= time_to_minutes(candidate["start"]):
            return self._fail(ERR_INVALID_ARGUMENT,
                              f"resulting end ({candidate['end']}) must be after start ({candidate['start']})")
        event.update(changes)
        if "attendees" in changes:
            event["attendees"] = list(changes["attendees"])
        conflicts = self._conflicts_for(event)
        self._state["events"].sort(key=lambda e: (e["date"], e["start"], e["id"]))
        return self._ok({"event": event, "conflicts": conflicts})

    def _tool_cancel_event(self, args: dict[str, Any]) -> ToolResult:
        event = self._find_event(args["event_id"])
        if event is None:
            return self._fail(ERR_NOT_FOUND, f"No event with id {args['event_id']!r}")
        if event["status"] == "cancelled":
            return self._ok({"event": event, "note": "Event was already cancelled."})
        event["status"] = "cancelled"
        return self._ok({"event": event})

    def _tool_find_conflicts(self, args: dict[str, Any]) -> ToolResult:
        if (bad := self._validate_dt(args, ("date",))):
            return bad
        events = [e for e in self._state["events"] if e["status"] == "scheduled"]
        if args.get("date"):
            events = [e for e in events if e["date"] == args["date"]]
        if args.get("attendee"):
            events = [e for e in events if args["attendee"] in e["attendees"]]
        conflicts = []
        for i, a in enumerate(events):
            for b in events[i + 1:]:
                if a["date"] != b["date"]:
                    continue
                a_s, a_e = time_to_minutes(a["start"]), time_to_minutes(a["end"])
                b_s, b_e = time_to_minutes(b["start"]), time_to_minutes(b["end"])
                if ranges_overlap(a_s, a_e, b_s, b_e):
                    conflicts.append({
                        "event_a": a["id"], "event_b": b["id"], "date": a["date"],
                        "overlap_start": minutes_to_time(max(a_s, b_s)),
                        "overlap_end": minutes_to_time(min(a_e, b_e)),
                        "titles": [a["title"], b["title"]],
                    })
        conflicts.sort(key=lambda c: (c["date"], c["overlap_start"], c["event_a"]))
        return self._ok({"conflicts": conflicts, "count": len(conflicts)})
