"""Deterministic benchmark construction: templates + handwritten cases.

Design rules enforced here:

- Every task stores its **full explicit fixture**, so the benchmark JSONL is
  self-contained and expected outcomes can never drift if default generators
  change between versions.
- Expected values (counts, ids, timestamps, SQL answers) are **computed from
  the fixture at build time**, never hardcoded prose — templates draw random
  fixtures and derive ground truth programmatically. ``_draw`` retries until a
  fixture satisfies the task's preconditions (e.g. "exactly one conflict
  tomorrow"), keeping every emitted task unambiguous.
- Lexical answer checks are used only where state cannot express correctness
  (information retrieval, refusals). They use computed ground-truth values or
  a documented inability-phrase list.
- Build order is fixed and all randomness flows from one seeded ``random.Random``,
  so ``build_benchmark(seed)`` is byte-identical across runs.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Callable

from toolforge.environments.calendar import CalendarEnvironment
from toolforge.environments.crm import CrmEnvironment
from toolforge.environments.email import EmailEnvironment
from toolforge.environments.files import FilesEnvironment
from toolforge.environments.pools import EMAILS, USER_EMAIL
from toolforge.environments.sql import SqlEnvironment
from toolforge.environments.todo import TodoEnvironment
from toolforge.tasks import (
    CATEGORIES,
    CAT_AMBIGUOUS,
    CAT_ARGUMENTS,
    CAT_CONFLICT,
    CAT_DEPENDENT,
    CAT_ERROR_RECOVERY,
    CAT_IMPOSSIBLE,
    CAT_INDIRECT,
    CAT_INVALID_TOOL,
    CAT_MULTI_STEP,
    CAT_NO_ACTION,
    CAT_RETRIEVE_ACT,
    CAT_SINGLE_STEP,
    DEFAULT_MAX_STEPS_BY_CATEGORY,
    DIFFICULTY_BY_CATEGORY,
    ExpectedOutcome,
    Task,
    TaskError,
    _CATEGORY_ABBR,
    assign_splits,
    balance_category_coverage,
    check_no_leakage,
)
from toolforge.timeutils import (
    DEFAULT_CURRENT_DATE,
    add_days,
    minutes_to_time,
    shift_time,
    time_to_minutes,
)

TOMORROW = add_days(DEFAULT_CURRENT_DATE, 1)

#: Documented lexical markers for "this is not possible with my tools".
#: Used ONLY for impossible/bait/refusal tasks, case-insensitively.
INABILITY_PHRASES = [
    "cannot", "can't", "unable", "not able", "no tool", "not possible",
    "isn't possible", "don't have", "do not have", "unsupported",
    "not supported", "no available tool", "does not support", "doesn't support",
    "no way to", "not something i can",
]
READONLY_REFUSAL_PHRASES = INABILITY_PHRASES + ["read-only", "read only", "only select", "select queries"]

#: Markers that an answer surfaces missing information instead of fabricating
#: success. Used for missing_info_no_action tasks, where structural checks
#: alone cannot distinguish an honest "I can't without X" from a false claim.
MISSING_INFO_PHRASES = [
    "which", "what", "cannot", "can't", "unable", "not able", "no ", "none",
    "not find", "didn't find", "did not find", "couldn't find", "missing",
    "need more", "need the", "need to know", "please ", "could you", "specify",
    "unclear", "ambiguous", "?", "don't have", "do not have", "not possible",
    "isn't possible", "no such", "there is no", "more information", "who ",
]

_FAMILY_ABBR = {"calendar": "cal", "email": "eml", "files": "fs", "crm": "crm", "sql": "db", "todo": "td"}


# --------------------------------------------------------------------------
# Builder scaffolding
# --------------------------------------------------------------------------

class _Alloc:
    """Stable sequential task ids: tf-<family>-<category>-<n>."""

    def __init__(self) -> None:
        self._counters: Counter = Counter()

    def next_id(self, family: str, category: str) -> str:
        key = (family, category)
        self._counters[key] += 1
        return f"tf-{_FAMILY_ABBR[family]}-{_CATEGORY_ABBR[category]}-{self._counters[key]:03d}"


def _draw(rng: random.Random, make: Callable[[int], Any],
          predicate: Callable[[Any], bool], what: str,
          attempts: int = 200) -> tuple[int, Any]:
    """Draw fixture seeds until ``predicate(make(seed))`` holds."""
    for _ in range(attempts):
        seed = rng.randrange(1, 10_000_000)
        candidate = make(seed)
        if predicate(candidate):
            return seed, candidate
    raise TaskError(f"Could not draw a fixture satisfying precondition: {what}")


def _number_variants(value: float | int) -> list[str]:
    """Accepted surface forms of a computed numeric answer."""
    variants: set[str] = set()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    variants.add(str(value))
    if isinstance(value, int):
        variants.add(f"{value:,}")
        variants.add(f"{float(value):.2f}")
    else:
        variants.add(f"{value:.2f}")
        variants.add(f"{round(value, 2):g}")
    return sorted(variants)


class _FamilyBuilder:
    """Per-family task factory bundling id allocation and common defaults."""

    def __init__(self, family: str, rng: random.Random, alloc: _Alloc,
                 benchmark_seed: int, version: str) -> None:
        self.family = family
        self.rng = rng
        self.alloc = alloc
        self.benchmark_seed = benchmark_seed
        self.version = version

    def task(self, category: str, request: str, fixture: dict[str, Any], fixture_seed: int,
             checks: list[dict[str, Any]], *, terminal: str = "final_answer",
             forbid: bool = False, max_invalid: int | None = None,
             tags: tuple[str, ...] = (), max_steps: int | None = None,
             source: str = "template", tool_families: list[str] | None = None) -> Task:
        checks = list(checks)
        if (category == CAT_NO_ACTION and terminal == "any"
                and not any(str(c.get("check", "")).startswith("answer_") for c in checks)):
            # Central invariant: structural checks alone cannot distinguish an
            # honest "I can't do this without X" from a fabricated success
            # claim, so no-action tasks must also surface the missing info.
            checks.append({"check": "answer_contains_any", "strings": MISSING_INFO_PHRASES})
        return Task(
            task_id=self.alloc.next_id(self.family, category),
            family=self.family,
            request=request,
            category=category,
            difficulty=DIFFICULTY_BY_CATEGORY[category],
            fixture=fixture,
            fixture_seed=fixture_seed,
            expected=ExpectedOutcome(terminal_action=terminal, forbid_state_changes=forbid,
                                     checks=checks, max_invalid_calls=max_invalid),
            tool_families=list(tool_families or [self.family]),
            max_steps=max_steps or DEFAULT_MAX_STEPS_BY_CATEGORY[category],
            tags=list(tags),
            source=source,
            benchmark_seed=self.benchmark_seed,
            benchmark_version=self.version,
        )

    def refusal_checks(self, extra: list[dict[str, Any]] | None = None,
                       phrases: list[str] | None = None) -> list[dict[str, Any]]:
        checks = [{"check": "answer_contains_any", "strings": phrases or INABILITY_PHRASES}]
        checks.extend(extra or [])
        return checks


# ==========================================================================
# Calendar
# ==========================================================================

def _build_calendar(rng: random.Random, alloc: _Alloc, bseed: int, version: str) -> list[Task]:
    b = _FamilyBuilder("calendar", rng, alloc, bseed, version)
    tasks: list[Task] = []

    def fx(seed: int) -> dict[str, Any]:
        return CalendarEnvironment(seed=seed).snapshot()

    def env(fixture: dict[str, Any]) -> CalendarEnvironment:
        return CalendarEnvironment(fixture=fixture)

    def events_on(fixture: dict[str, Any], date: str) -> list[dict[str, Any]]:
        return [e for e in fixture["events"] if e["date"] == date and e["status"] == "scheduled"]

    # -- single-step selection ------------------------------------------------
    s, f = rng.randrange(1, 10**7), None
    f = fx(s)
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"Create a calendar event titled 'Project review' on {add_days(DEFAULT_CURRENT_DATE, 2)} "
        "from 13:00 to 14:00.",
        f, s,
        [{"check": "record_exists", "collection": "events",
          "where": {"title": "Project review", "date": add_days(DEFAULT_CURRENT_DATE, 2),
                    "start": "13:00", "end": "14:00", "status": "scheduled"}}],
        tags=["create"]))

    def _has_scheduled(fx_: dict) -> bool:
        return any(e["status"] == "scheduled" for e in fx_["events"])
    s, f = _draw(rng, fx, _has_scheduled, "at least one scheduled event")
    ev = rng.choice([e for e in f["events"] if e["status"] == "scheduled"])
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"Cancel my '{ev['title']}' event on {ev['date']} (id {ev['id']}).",
        f, s,
        [{"check": "field_equals", "collection": "events", "where": {"id": ev["id"]},
          "field": "status", "value": "cancelled"}],
        tags=["cancel"]))

    s, f = _draw(rng, fx, lambda x: len([e for e in x["events"] if e["date"] == TOMORROW]) >= 2,
                 ">=2 events tomorrow")
    n = len([e for e in f["events"] if e["date"] == TOMORROW])
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"How many calendar events (including cancelled ones) do I have on {TOMORROW}? "
        "Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n)]},
         {"check": "tool_called", "tool": "calendar.list_events", "min_times": 1}],
        forbid=True, tags=["count"]))

    def _has_locatable(fx_: dict) -> bool:
        return any(e["status"] == "scheduled" for e in fx_["events"])
    s, f = _draw(rng, fx, _has_locatable, "scheduled event")
    ev = rng.choice([e for e in f["events"] if e["status"] == "scheduled"])
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"Set the location of event {ev['id']} ('{ev['title']}' on {ev['date']}) to 'Room C'.",
        f, s,
        [{"check": "field_equals", "collection": "events", "where": {"id": ev["id"]},
          "field": "location", "value": "Room C"}],
        tags=["update"]))

    def _exactly_one_conflict(x: dict) -> bool:
        return len(env(x).execute("find_conflicts", {"date": TOMORROW}).output["conflicts"]) == 1
    s, f = _draw(rng, fx, _exactly_one_conflict, "exactly one conflict pair tomorrow")
    conflicts_tomorrow = env(f).execute("find_conflicts", {"date": TOMORROW}).output["conflicts"]
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"Do I have any overlapping events on {TOMORROW}? Answer yes or no and name the "
        "titles involved.",
        f, s,
        [{"check": "answer_contains_any", "strings": ["yes"]},
         {"check": "answer_contains_all",
          "strings": [t.lower() for t in conflicts_tomorrow[0]["titles"]]},
         {"check": "tool_called", "tool": "calendar.find_conflicts", "min_times": 1}],
        forbid=True, tags=["conflicts", "report"]))

    # -- argument construction -------------------------------------------------
    s, f = rng.randrange(1, 10**7), fx(0)
    f = fx(s)
    date = add_days(DEFAULT_CURRENT_DATE, 3)
    a1, a2 = EMAILS[0], EMAILS[2]
    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Create an event 'Roadmap planning' on {date} from 14:00 to 15:30 with "
        f"{a1} and {a2} in 'Room B'.",
        f, s,
        [{"check": "record_exists", "collection": "events",
          "where": {"title": "Roadmap planning", "date": date, "start": "14:00", "end": "15:30",
                    "attendees": [a1, a2], "location": "Room B"}}],
        tags=["create", "attendees"]))

    notes = "Agenda: https://wiki.example.com/agenda-42 — bring laptops."
    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Create an event 'Workshop prep' on {date} from 09:00 to 09:45 with exactly these "
        f"notes: {notes}",
        f, s,
        [{"check": "record_exists", "collection": "events",
          "where": {"title": "Workshop prep", "date": date, "start": "09:00", "end": "09:45",
                    "notes": notes}}],
        tags=["create", "notes"]))

    s, f = _draw(rng, fx, lambda x: len(events_on(x, TOMORROW)) >= 1, "event tomorrow")
    ev = events_on(f, TOMORROW)[0]
    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Update event {ev['id']}: retitle it to '{ev['title']} (v2)' and move it to "
        f"{add_days(DEFAULT_CURRENT_DATE, 4)}, 11:00-12:00.",
        f, s,
        [{"check": "field_equals", "collection": "events", "where": {"id": ev["id"]},
          "field": "title", "value": f"{ev['title']} (v2)"},
         {"check": "field_equals", "collection": "events", "where": {"id": ev["id"]},
          "field": "date", "value": add_days(DEFAULT_CURRENT_DATE, 4)},
         {"check": "field_equals", "collection": "events", "where": {"id": ev["id"]},
          "field": "start", "value": "11:00"},
         {"check": "field_equals", "collection": "events", "where": {"id": ev["id"]},
          "field": "end", "value": "12:00"}],
        max_steps=4, tags=["update"]))

    s, f = _draw(rng, fx, lambda x: any(
        len(e["attendees"]) == 1 and e["status"] == "scheduled" for e in x["events"]),
        "event with exactly one attendee")
    ev = next(e for e in f["events"] if len(e["attendees"]) == 1 and e["status"] == "scheduled")
    new_attendee = next(m for m in EMAILS if m not in ev["attendees"])
    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Add {new_attendee} to the attendees of event {ev['id']} ('{ev['title']}'), "
        "keeping the existing attendee.",
        f, s,
        [{"check": "field_equals", "collection": "events", "where": {"id": ev["id"]},
          "field": "attendees", "value": sorted(ev["attendees"] + [new_attendee])}],
        tags=["update", "attendees"]))

    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Create three events on {add_days(DEFAULT_CURRENT_DATE, 5)}: 'Interview 1' 09:00-09:45, "
        "'Interview 2' 10:00-10:45, and 'Debrief' 11:00-11:30.",
        f, s,
        [{"check": "record_exists", "collection": "events",
          "where": {"title": t, "date": add_days(DEFAULT_CURRENT_DATE, 5), "start": st, "end": en}}
         for t, st, en in [("Interview 1", "09:00", "09:45"), ("Interview 2", "10:00", "10:45"),
                           ("Debrief", "11:00", "11:30")]],
        max_steps=5, tags=["create", "multiple"]))

    # -- multi-step workflows ---------------------------------------------------
    s, f = rng.randrange(1, 10**7), fx(0)
    f = fx(s)
    tasks.append(b.task(
        CAT_MULTI_STEP,
        f"Find a free one-hour slot on {TOMORROW} between 09:00 and 17:00 and create an event "
        "'Focus block' in one of the free slots.",
        f, s,
        [{"check": "record_exists", "collection": "events",
          "where": {"title": "Focus block", "date": TOMORROW}},
         {"check": "no_overlaps", "collection": "events",
          "subject_where": {"title": "Focus block"}, "filter": {"status": "scheduled"}}],
        max_steps=4, tags=["free_slot", "create"]))

    date = add_days(DEFAULT_CURRENT_DATE, 2)
    tasks.append(b.task(
        CAT_MULTI_STEP,
        f"On {date}, first create 'Prep' from 08:30 to 09:00, then create 'Workshop' from "
        "09:00 to 11:00.",
        f, s,
        [{"check": "record_exists", "collection": "events",
          "where": {"title": "Prep", "date": date, "start": "08:30", "end": "09:00"}},
         {"check": "record_exists", "collection": "events",
          "where": {"title": "Workshop", "date": date, "start": "09:00", "end": "11:00"}}],
        max_steps=4, tags=["create", "sequence"]))

    bob = "bob@example.com"
    s, f = _draw(rng, fx,
                 lambda x: len([e for e in x["events"]
                                if e["date"] == TOMORROW and bob in e["attendees"]
                                and e["status"] == "scheduled"]) >= 2,
                 ">=2 events with bob tomorrow")
    targets = [e for e in f["events"]
               if e["date"] == TOMORROW and bob in e["attendees"] and e["status"] == "scheduled"]
    tasks.append(b.task(
        CAT_MULTI_STEP,
        f"Cancel all of my scheduled events with {bob} on {TOMORROW}.",
        f, s,
        [{"check": "field_equals", "collection": "events", "where": {"id": e["id"]},
          "field": "status", "value": "cancelled"} for e in targets]
        + [{"check": "record_count", "collection": "events",
            "where": {"date": TOMORROW, "attendees": bob, "status": "scheduled"},
            "op": "==", "value": 0}],
        max_steps=6, tags=["cancel", "batch"]))

    def _exactly_one_conflict_tomorrow(x: dict) -> bool:
        return len(env(x).execute("find_conflicts", {"date": TOMORROW}).output["conflicts"]) == 1
    s, f = _draw(rng, fx, _exactly_one_conflict_tomorrow, "exactly one conflict tomorrow")
    pair = env(f).execute("find_conflicts", {"date": TOMORROW}).output["conflicts"][0]
    ev_a = next(e for e in f["events"] if e["id"] == pair["event_a"])
    ev_b = next(e for e in f["events"] if e["id"] == pair["event_b"])
    later, earlier = (ev_a, ev_b) if ev_a["start"] > ev_b["start"] else (ev_b, ev_a)
    tasks.append(b.task(
        CAT_MULTI_STEP,
        f"Find the overlapping events on {TOMORROW} and cancel the one that starts later, "
        "keeping the earlier one.",
        f, s,
        [{"check": "field_equals", "collection": "events", "where": {"id": later["id"]},
          "field": "status", "value": "cancelled"},
         {"check": "field_equals", "collection": "events", "where": {"id": earlier["id"]},
          "field": "status", "value": "scheduled"}],
        max_steps=5, tags=["conflicts", "cancel"]))

    tasks.append(b.task(
        CAT_MULTI_STEP,
        f"Create a 'Placeholder' event on {add_days(DEFAULT_CURRENT_DATE, 6)} from 13:00 to "
        "14:00, then move it to 15:00-16:00.",
        f, s,
        [{"check": "record_exists", "collection": "events",
          "where": {"title": "Placeholder", "date": add_days(DEFAULT_CURRENT_DATE, 6),
                    "start": "15:00", "end": "16:00"}},
         {"check": "tool_called", "tool": "calendar.create_event", "min_times": 1},
         {"check": "tool_called", "tool": "calendar.update_event", "min_times": 1}],
        max_steps=5, tags=["create", "update"]))

    # -- retrieve then act/report ------------------------------------------------
    s, f = _draw(rng, fx, lambda x: len(x["events"]) >= 3, ">=3 events")
    ev = rng.choice(f["events"])
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        f"What time does my '{ev['title']}' event on {ev['date']} start and end?",
        f, s,
        [{"check": "answer_contains_all", "strings": [ev["start"], ev["end"]]}],
        forbid=True, tags=["report"]))

    s, f = _draw(rng, fx, lambda x: any(
        len(e["attendees"]) >= 2 for e in x["events"]), "event with >=2 attendees")
    ev = next(e for e in f["events"] if len(e["attendees"]) >= 2)
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        f"Who is attending '{ev['title']}' on {ev['date']}? List every attendee email.",
        f, s,
        [{"check": "answer_contains_all", "strings": list(ev["attendees"])}],
        forbid=True, tags=["report"]))

    s, f = _draw(rng, fx, lambda x: len(events_on(x, TOMORROW)) >= 2, ">=2 events tomorrow")
    first = sorted(events_on(f, TOMORROW), key=lambda e: (e["start"], e["id"]))[0]
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        f"What is my first event on {TOMORROW}? Give its title and start time.",
        f, s,
        [{"check": "answer_contains_all", "strings": [first["title"], first["start"]]}],
        forbid=True, tags=["report"]))

    s, f = _draw(rng, fx, lambda x: len(events_on(x, TOMORROW)) >= 1, ">=1 event tomorrow")
    busy = [(time_to_minutes(e["start"]), time_to_minutes(e["end"])) for e in events_on(f, TOMORROW)]
    lunch_free = not any(s_ < 12 * 60 + 30 and 12 * 60 < e_ for s_, e_ in busy)
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        f"Am I free between 12:00 and 13:00 on {TOMORROW}? Answer yes or no.",
        f, s,
        [{"check": "answer_contains_any", "strings": ["yes" if lunch_free else "no"]}],
        forbid=True, tags=["report"]))

    # -- dependent actions -------------------------------------------------------
    s, f = _draw(rng, fx, lambda x: len(events_on(x, TOMORROW)) >= 2, ">=2 events tomorrow")
    earliest = sorted(events_on(f, TOMORROW), key=lambda e: (e["start"], e["id"]))[0]
    tasks.append(b.task(
        CAT_DEPENDENT,
        f"Find my earliest event on {TOMORROW} and rename it to 'Standup (moved)'.",
        f, s,
        [{"check": "field_equals", "collection": "events", "where": {"id": earliest["id"]},
          "field": "title", "value": "Standup (moved)"},
         {"check": "tool_called", "tool": "calendar.list_events", "min_times": 1}],
        max_steps=5, tags=["lookup", "update"]))

    def _has_30_slot(x: dict) -> bool:
        slots = env(x).execute("find_free_slots",
                               {"date": TOMORROW, "duration_minutes": 30}).output["free_slots"]
        return len(slots) >= 1
    s, f = _draw(rng, fx, _has_30_slot, ">=1 free 30-min slot tomorrow")
    slot = env(f).execute("find_free_slots",
                          {"date": TOMORROW, "duration_minutes": 30}).output["free_slots"][0]
    exp_end = minutes_to_time(time_to_minutes(slot["start"]) + 30)
    tasks.append(b.task(
        CAT_DEPENDENT,
        f"Find the first free 30-minute slot on {TOMORROW} (working hours 09:00-17:00) and "
        "book it as 'Deep work'.",
        f, s,
        [{"check": "record_exists", "collection": "events",
          "where": {"title": "Deep work", "date": TOMORROW,
                    "start": slot["start"], "end": exp_end}}],
        max_steps=5, tags=["free_slot", "create"]))

    def _one_conflict_tomorrow(x: dict) -> bool:
        return len(env(x).execute("find_conflicts", {"date": TOMORROW}).output["conflicts"]) == 1
    s, f = _draw(rng, fx, _one_conflict_tomorrow, "exactly one conflict tomorrow")
    pair = env(f).execute("find_conflicts", {"date": TOMORROW}).output["conflicts"][0]
    anchor = next(e for e in f["events"] if e["id"] == pair["event_a"])
    other = next(e for e in f["events"] if e["id"] == pair["event_b"])
    tasks.append(b.task(
        CAT_DEPENDENT,
        f"Find the event that overlaps with '{anchor['title']}' ({anchor['id']}) on {TOMORROW} "
        "and set its notes to 'needs rescheduling'.",
        f, s,
        [{"check": "field_equals", "collection": "events", "where": {"id": other["id"]},
          "field": "notes", "value": "needs rescheduling"}],
        max_steps=5, tags=["conflicts", "update"]))

    # -- ambiguity / no-action / indirect ---------------------------------------
    s = rng.randrange(1, 10**7)
    f = fx(s)
    tasks.append(b.task(
        CAT_AMBIGUOUS,
        "Schedule a meeting with the team sometime next week.",
        f, s, [], terminal="clarify", forbid=True, tags=["missing:time,attendees,title"]))

    def _two_mornings(x: dict) -> bool:
        return len([e for e in x["events"]
                    if time_to_minutes(e["start"]) < 12 * 60 and e["status"] == "scheduled"]) >= 2
    s, f = _draw(rng, fx, _two_mornings, ">=2 morning events")
    tasks.append(b.task(
        CAT_AMBIGUOUS,
        "Reschedule my morning meeting to 14:00.",
        f, s, [], terminal="clarify", forbid=True, tags=["missing:which-event"]))

    tasks.append(b.task(
        CAT_NO_ACTION,
        "Add the event from that email to my calendar.",
        f, s, [], terminal="any", forbid=True, tags=["missing:context"]))

    def _unique_attendee_day(x: dict) -> bool:
        seen: Counter = Counter()
        for e in x["events"]:
            if e["status"] == "scheduled":
                for a in e["attendees"]:
                    seen[(e["date"], a)] += 1
        return any(v == 1 for v in seen.values())
    s, f = _draw(rng, fx, _unique_attendee_day, "unique (date, attendee) event")
    counter: Counter = Counter()
    for e in f["events"]:
        if e["status"] == "scheduled":
            for a in e["attendees"]:
                counter[(e["date"], a)] += 1
    (day, attendee), _ = next((k, v) for k, v in sorted(counter.items()) if v == 1)
    ev = next(e for e in f["events"]
              if e["date"] == day and attendee in e["attendees"] and e["status"] == "scheduled")
    if time_to_minutes(ev["end"]) + 30 <= 23 * 60 + 30:
        new_start, new_end = shift_time(ev["start"], 30), shift_time(ev["end"], 30)
        tasks.append(b.task(
            CAT_INDIRECT,
            f"Move my meeting with {attendee} on {day} 30 minutes later.",
            f, s,
            [{"check": "field_equals", "collection": "events", "where": {"id": ev["id"]},
              "field": "start", "value": new_start},
             {"check": "field_equals", "collection": "events", "where": {"id": ev["id"]},
              "field": "end", "value": new_end}],
            max_steps=5, tags=["lookup", "update"]))

    def _sequential_pair(x: dict) -> bool:
        by_date: dict[str, list[dict]] = {}
        for e in x["events"]:
            if e["status"] == "scheduled":
                by_date.setdefault(e["date"], []).append(e)
        for day, evs in by_date.items():
            ordered = sorted(evs, key=lambda e: (e["start"], e["id"]))
            if len(ordered) >= 2:
                return True
        return False
    s, f = _draw(rng, fx, _sequential_pair, "date with >=2 events")
    by_date: dict[str, list[dict]] = {}
    for e in f["events"]:
        if e["status"] == "scheduled":
            by_date.setdefault(e["date"], []).append(e)
    day = sorted(d for d, evs in by_date.items() if len(evs) >= 2)[0]
    ordered = sorted(by_date[day], key=lambda e: (e["start"], e["id"]))
    anchor, nxt = ordered[0], ordered[1]
    tasks.append(b.task(
        CAT_INDIRECT,
        f"On {day}, rename the event that starts right after '{anchor['title']}' to 'Follow-up'.",
        f, s,
        [{"check": "field_equals", "collection": "events", "where": {"id": nxt["id"]},
          "field": "title", "value": "Follow-up"}],
        max_steps=5, tags=["lookup", "update"]))

    # -- error recovery -----------------------------------------------------------
    def _late_event(x: dict) -> bool:
        return any(e["status"] == "scheduled" and time_to_minutes(e["end"]) > 16 * 60 + 30
                   for e in x["events"])
    s, f = _draw(rng, fx, _late_event, "event ending after 16:30")
    ev = next(e for e in f["events"]
              if e["status"] == "scheduled" and time_to_minutes(e["end"]) > 16 * 60 + 30)
    tasks.append(b.task(
        CAT_ERROR_RECOVERY,
        f"Change the start time of evt_999 to 16:00 — that's my '{ev['title']}' event on "
        f"{ev['date']}.",
        f, s,
        [{"check": "field_equals", "collection": "events", "where": {"id": ev["id"]},
          "field": "start", "value": "16:00"},
         {"check": "tool_called", "tool": "calendar.list_events", "min_times": 1}],
        max_steps=6, tags=["wrong_id"]))

    tasks.append(b.task(
        CAT_ERROR_RECOVERY,
        f"Cancel evt_888 — it's the '{ev['title']}' event on {ev['date']}.",
        f, s,
        [{"check": "field_equals", "collection": "events", "where": {"id": ev["id"]},
          "field": "status", "value": "cancelled"},
         {"check": "tool_called", "tool": "calendar.list_events", "min_times": 1}],
        max_steps=6, tags=["wrong_id"]))

    # -- conflicting constraints ----------------------------------------------------
    attendee = EMAILS[1]
    s, f = _draw(rng, fx, lambda x: len(events_on(x, TOMORROW)) >= 1, "event tomorrow")
    busy_ev = events_on(f, TOMORROW)[0]
    tasks.append(b.task(
        CAT_CONFLICT,
        f"Create an event 'Design sync' with {attendee} on {TOMORROW} at {busy_ev['start']} "
        "for one hour. If that time is taken, pick any free one-hour slot the same day "
        "between 09:00 and 17:00 instead.",
        f, s,
        [{"check": "record_exists", "collection": "events",
          "where": {"title": "Design sync", "date": TOMORROW}},
         {"check": "no_overlaps", "collection": "events",
          "subject_where": {"title": "Design sync"}, "filter": {"status": "scheduled"}}],
        max_steps=5, tags=["busy_slot", "fallback"]))

    def _busy_with_fallback(x: dict) -> bool:
        if not events_on(x, TOMORROW):
            return False
        slots = env(x).execute("find_free_slots",
                               {"date": TOMORROW, "duration_minutes": 30}).output["free_slots"]
        return len(slots) >= 1
    s, f = _draw(rng, fx, _busy_with_fallback, "busy slot + >=1 free 30-min slot")
    busy_ev = events_on(f, TOMORROW)[0]
    slot = env(f).execute("find_free_slots",
                          {"date": TOMORROW, "duration_minutes": 30}).output["free_slots"][0]
    tasks.append(b.task(
        CAT_CONFLICT,
        f"Book a 30-minute '1:1 with manager' on {TOMORROW} starting at {busy_ev['start']}. "
        "If that start time is already taken, use the earliest free 30-minute slot of the day "
        "(working hours 09:00-17:00) instead.",
        f, s,
        [{"check": "record_exists", "collection": "events",
          "where": {"title": "1:1 with manager", "date": TOMORROW, "start": slot["start"]}},
         {"check": "no_overlaps", "collection": "events",
          "subject_where": {"title": "1:1 with manager"}, "filter": {"status": "scheduled"}}],
        max_steps=5, tags=["busy_slot", "earliest_slot"]))

    # Fully booked day: correct behavior is to create nothing and report failure.
    packed_date = add_days(DEFAULT_CURRENT_DATE, 3)
    packed = {
        "current_date": DEFAULT_CURRENT_DATE, "user_email": USER_EMAIL, "next_event_num": 5,
        "events": [
            {"id": f"evt_{i + 1:03d}", "title": t, "date": packed_date, "start": st, "end": en,
             "attendees": [EMAILS[i]], "location": "Room A", "notes": "", "status": "scheduled"}
            for i, (t, st, en) in enumerate([
                ("Standup", "09:00", "10:00"), ("Workshop", "10:00", "12:00"),
                ("Lunch block", "12:00", "13:00"), ("Client calls", "13:00", "17:00")])
        ],
    }
    tasks.append(b.task(
        CAT_CONFLICT,
        f"Schedule a one-hour 'Team offsite prep' on {packed_date} between 09:00 and 17:00. "
        "If there is no free slot, do not create anything and tell me.",
        packed, 0,
        [{"check": "record_count", "collection": "events", "where": {"date": packed_date},
          "op": "==", "value": 4},
         {"check": "answer_contains_any",
          "strings": ["no free", "no available", "fully booked", "no slot", "not possible",
                      "cannot", "can't", "unable", "booked solid", "no one-hour"]}],
        max_steps=5, source="handwritten", tags=["no_slot", "report"]))

    # -- impossible / bait -----------------------------------------------------------
    tasks.append(b.task(
        CAT_IMPOSSIBLE,
        f"Book a table at Luigi's restaurant for the team dinner on {add_days(DEFAULT_CURRENT_DATE, 4)}.",
        fx(rng.randrange(1, 10**7)), 0,
        b.refusal_checks(), forbid=True, max_invalid=2, tags=["no_such_capability"]))

    tasks.append(b.task(
        CAT_INVALID_TOOL,
        f"Send an SMS reminder about '{ev['title']}' to all its attendees.",
        f, s,
        b.refusal_checks(), forbid=True, max_invalid=2, tags=["bait:sms"]))

    tasks.append(b.task(
        CAT_INVALID_TOOL,
        f"Print the agenda for {TOMORROW} and fax it to the front desk.",
        f, s,
        b.refusal_checks(), forbid=True, max_invalid=2, tags=["bait:fax,printer"]))

    return tasks


# ==========================================================================
# Email
# ==========================================================================

def _build_email(rng: random.Random, alloc: _Alloc, bseed: int, version: str) -> list[Task]:
    b = _FamilyBuilder("email", rng, alloc, bseed, version)
    tasks: list[Task] = []

    def fx(seed: int) -> dict[str, Any]:
        return EmailEnvironment(seed=seed).snapshot()

    def inbox(fixture: dict, folder: str = "inbox") -> list[dict]:
        return [m for m in fixture["messages"] if m["folder"] == folder]

    # -- single step -----------------------------------------------------------------
    s = rng.randrange(1, 10**7)
    f = fx(s)
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "Send an email to alice@example.com with subject 'Status update' and body "
        "'All green this week.'",
        f, s,
        [{"check": "record_exists", "collection": "messages",
          "where": {"folder": "sent", "to": "alice@example.com", "subject": "Status update"}}],
        tags=["send"]))

    s, f = _draw(rng, fx, lambda x: len(inbox(x)) >= 1, ">=1 inbox message")
    msg = inbox(f)[0]
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"Move message {msg['id']} ('{msg['subject']}') to the archive folder.",
        f, s,
        [{"check": "field_equals", "collection": "messages", "where": {"id": msg["id"]},
          "field": "folder", "value": "archive"}],
        tags=["move"]))

    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"Read message {msg['id']} and tell me who sent it (give the exact address).",
        f, s,
        [{"check": "answer_contains_all", "strings": [msg["from"]]},
         {"check": "tool_called", "tool": "email.read_message", "min_times": 1}],
        forbid=True, tags=["read", "report"]))

    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"Draft a reply to message {msg['id']} saying 'Thanks, sounds good.'",
        f, s,
        [{"check": "record_exists", "collection": "messages",
          "where": {"folder": "drafts", "in_reply_to": msg["id"], "body": "Thanks, sounds good."}}],
        tags=["draft"]))

    def _has_unread(x: dict) -> bool:
        return len([m for m in inbox(x) if not m["read"]]) >= 2
    s, f = _draw(rng, fx, _has_unread, ">=2 unread inbox messages")
    n_unread = len([m for m in inbox(f) if not m["read"]])
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "How many unread emails are in my inbox? Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n_unread)]},
         {"check": "tool_called", "tool": "email.search_messages", "min_times": 1}],
        forbid=True, tags=["count"]))

    # -- argument construction ----------------------------------------------------------
    body = "Report is attached.\nPlease review by Friday."
    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Send an email to {EMAILS[3]} with subject 'Weekly report' and exactly this body "
        f"(two lines): {body!r}",
        f, s,
        [{"check": "record_exists", "collection": "messages",
          "where": {"folder": "sent", "to": EMAILS[3], "subject": "Weekly report", "body": body}}],
        tags=["send", "exact_body"]))

    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Send an email to {EMAILS[0]} with subject 'Meeting {TOMORROW} 10:00' and body "
        "'Confirming our slot.'",
        f, s,
        [{"check": "record_exists", "collection": "messages",
          "where": {"folder": "sent", "to": EMAILS[0],
                    "subject": f"Meeting {TOMORROW} 10:00", "body": "Confirming our slot."}}],
        tags=["send", "exact_subject"]))

    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Move message {msg['id']} to the spam folder.",
        f, s,
        [{"check": "field_equals", "collection": "messages", "where": {"id": msg["id"]},
          "field": "folder", "value": "spam"}],
        tags=["move"]))

    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Draft a reply to {msg['id']} with exactly this body: 'I can confirm the numbers. "
        "The Q1 budget is fine.'",
        f, s,
        [{"check": "record_exists", "collection": "messages",
          "where": {"folder": "drafts", "in_reply_to": msg["id"],
                    "body": "I can confirm the numbers. The Q1 budget is fine."}}],
        tags=["draft", "exact_body"]))

    # -- multi-step ----------------------------------------------------------------------
    def _two_from_sender(x: dict) -> bool:
        senders = Counter(m["from"] for m in inbox(x))
        return any(n >= 2 for n in senders.values())
    s, f = _draw(rng, fx, _two_from_sender, ">=2 inbox mails from one sender")
    sender, n_inbox = Counter(m["from"] for m in inbox(f)).most_common(1)[0]
    inbox_ids = [m["id"] for m in inbox(f) if m["from"] == sender]
    tasks.append(b.task(
        CAT_MULTI_STEP,
        f"Find all emails from {sender} in my inbox and move every one of them to the archive.",
        f, s,
        [{"check": "field_equals", "collection": "messages", "where": {"id": mid},
          "field": "folder", "value": "archive"} for mid in inbox_ids]
        + [{"check": "record_count", "collection": "messages",
            "where": {"from": sender, "folder": "inbox"}, "op": "==", "value": 0}],
        max_steps=6, tags=["search", "move", "batch"]))

    reply_body = "Got it, thanks for the update."
    tasks.append(b.task(
        CAT_MULTI_STEP,
        f"Read message {msg['id']}, draft a reply saying '{reply_body}', and send that draft.",
        f, s,
        [{"check": "record_exists", "collection": "messages",
          "where": {"folder": "sent", "in_reply_to": msg["id"], "body": reply_body}},
         {"check": "tool_called", "tool": "email.draft_reply", "min_times": 1},
         {"check": "tool_called", "tool": "email.send_draft", "min_times": 1}],
        max_steps=5, tags=["read", "draft", "send"]))

    def _keyword_multi(x: dict) -> bool:
        return len([m for m in inbox(x) if "budget" in (m["subject"] + m["body"]).lower()]) >= 1
    s, f = _draw(rng, fx, _keyword_multi, ">=1 budget email in inbox")
    matches = [m for m in inbox(f) if "budget" in (m["subject"] + m["body"]).lower()]
    tasks.append(b.task(
        CAT_MULTI_STEP,
        "Search my inbox for emails mentioning 'budget' and move all of them to the archive.",
        f, s,
        [{"check": "field_equals", "collection": "messages", "where": {"id": m["id"]},
          "field": "folder", "value": "archive"} for m in matches]
        + [{"check": "tool_called", "tool": "email.search_messages", "min_times": 1}],
        max_steps=6, tags=["search", "move"]))

    s, f = _draw(rng, fx, lambda x: len(inbox(x)) >= 2, ">=2 inbox messages")
    latest = sorted(inbox(f), key=lambda m: (m["date"], m["id"]))[-1]
    tasks.append(b.task(
        CAT_MULTI_STEP,
        "Find the most recent email in my inbox, draft a reply to it saying 'Received, "
        "thanks.', send that draft, and tell me the subject of the email you replied to.",
        f, s,
        [{"check": "record_exists", "collection": "messages",
          "where": {"folder": "sent", "in_reply_to": latest["id"], "body": "Received, thanks."}},
         {"check": "answer_contains_all", "strings": [latest["subject"]]}],
        max_steps=6, tags=["search", "draft", "send", "report"]))

    s, f = _draw(rng, fx, _has_unread, ">=2 unread inbox")
    unread = sorted([m for m in inbox(f) if not m["read"]],
                    key=lambda m: (m["date"], m["id"]))
    newest_unread = unread[-1]
    tasks.append(b.task(
        CAT_MULTI_STEP,
        "Find my unread inbox emails, archive the most recent one, and tell me its subject.",
        f, s,
        [{"check": "field_equals", "collection": "messages", "where": {"id": newest_unread["id"]},
          "field": "folder", "value": "archive"},
         {"check": "answer_contains_all", "strings": [newest_unread["subject"]]}],
        max_steps=6, tags=["search", "move", "report"]))

    # -- retrieve then report -------------------------------------------------------------
    s, f = _draw(rng, fx, lambda x: len(inbox(x)) >= 1, "inbox message")
    msg = sorted(inbox(f), key=lambda m: (m["date"], m["id"]))[-1]
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        f"What is the subject of the most recent email in my inbox from {msg['from']}? "
        "Quote it exactly.",
        f, s,
        [{"check": "answer_contains_all", "strings": [msg["subject"]]}],
        forbid=True, tags=["search", "report"]))

    def _has_invoice(x: dict) -> bool:
        return any("INV-201" in m["body"] for m in x["messages"])
    s, f = _draw(rng, fx, _has_invoice, "invoice email present")
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "Which invoice number is overdue according to the billing email? Give the exact "
        "invoice id and amount.",
        f, s,
        [{"check": "answer_contains_all", "strings": ["INV-201", "4200"]}],
        forbid=True, tags=["read", "report"]))

    def _has_migration(x: dict) -> bool:
        return any("migration window" in m["body"].lower() for m in x["messages"])
    s, f = _draw(rng, fx, _has_migration, "migration email present")
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "According to the ops team's email, when is the server migration window? Give day "
        "and times.",
        f, s,
        [{"check": "answer_contains_all", "strings": ["saturday", "02:00-06:00"]}],
        forbid=True, tags=["read", "report"]))

    s, f = _draw(rng, fx, lambda x: len(inbox(x)) >= 3, ">=3 inbox messages")
    sender = inbox(f)[0]["from"]
    n_from = len([m for m in inbox(f) if m["from"] == sender])
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        f"How many emails from {sender} are in my inbox? Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n_from)]}],
        forbid=True, tags=["search", "count"]))

    # -- dependent -------------------------------------------------------------------------
    s, f = _draw(rng, fx, lambda x: len(inbox(x)) >= 1, "inbox message")
    target = inbox(f)[0]
    tasks.append(b.task(
        CAT_DEPENDENT,
        f"Find out who sent the email with subject '{target['subject']}' and draft them a "
        "reply asking 'Could you send over the report?'",
        f, s,
        [{"check": "record_exists", "collection": "messages",
          "where": {"folder": "drafts", "in_reply_to": target["id"],
                    "to": target["from"],
                    "body": {"op": "contains", "value": "report"}}}],
        max_steps=5, tags=["search", "draft"]))

    def _has_offsite(x: dict) -> bool:
        return any("offsite" in m["subject"].lower() for m in x["messages"])
    s, f = _draw(rng, fx, _has_offsite, "offsite email present")
    tasks.append(b.task(
        CAT_DEPENDENT,
        "Find the email about the team offsite and send a new email to carol@example.com "
        "with body 'FYI' and a subject that mentions the offsite.",
        f, s,
        [{"check": "record_exists", "collection": "messages",
          "where": {"folder": "sent", "to": "carol@example.com", "body": "FYI",
                    "subject": {"op": "contains", "value": "offsite", "case_insensitive": True}}}],
        max_steps=5, tags=["search", "send"]))

    # -- ambiguity / no action ----------------------------------------------------------------
    s, f = rng.randrange(1, 10**7), fx(0)
    f = fx(s)
    tasks.append(b.task(
        CAT_AMBIGUOUS,
        "Reply to that email and tell them yes.",
        f, s, [], terminal="clarify", forbid=True, tags=["missing:which-email"]))

    tasks.append(b.task(
        CAT_AMBIGUOUS,
        "Forward the invoice to accounting.",
        f, s, [], terminal="clarify", forbid=True, tags=["missing:recipient,which-invoice"]))

    tasks.append(b.task(
        CAT_NO_ACTION,
        "Send the signed contract to legal.",
        f, s, [], terminal="any", forbid=True, tags=["missing:address,content"]))

    tasks.append(b.task(
        CAT_NO_ACTION,
        "Email Bob the file he asked for.",
        f, s, [], terminal="any", forbid=True, tags=["missing:which-bob,which-file,no-attachments"]))

    # -- indirect -------------------------------------------------------------------------------
    def _unique_sender_latest(x: dict) -> bool:
        return len(inbox(x)) >= 2
    s, f = _draw(rng, fx, _unique_sender_latest, ">=2 inbox")
    sender = sorted(inbox(f), key=lambda m: (m["date"], m["id"]))[-1]["from"]
    latest = sorted([m for m in inbox(f) if m["from"] == sender],
                    key=lambda m: (m["date"], m["id"]))[-1]
    tasks.append(b.task(
        CAT_INDIRECT,
        f"Draft a reply to the latest email from {sender} saying 'I'll be there.' and send it.",
        f, s,
        [{"check": "record_exists", "collection": "messages",
          "where": {"folder": "sent", "in_reply_to": latest["id"], "body": "I'll be there."}}],
        max_steps=5, tags=["search", "draft", "send"]))

    def _has_migration_mail(x: dict) -> bool:
        return any("migration" in m["subject"].lower() for m in inbox(x))
    s, f = _draw(rng, fx, _has_migration_mail, "migration subject in inbox")
    mig = next(m for m in inbox(f) if "migration" in m["subject"].lower())
    tasks.append(b.task(
        CAT_INDIRECT,
        "Archive the email about the server migration.",
        f, s,
        [{"check": "field_equals", "collection": "messages", "where": {"id": mig["id"]},
          "field": "folder", "value": "archive"}],
        max_steps=5, tags=["search", "move"]))

    # -- error recovery --------------------------------------------------------------------------
    def _has_budget_mail(x: dict) -> bool:
        return any("budget" in m["subject"].lower() for m in x["messages"])
    s, f = _draw(rng, fx, _has_budget_mail, "budget email present")
    budget = next(m for m in f["messages"] if "budget" in m["subject"].lower())
    tasks.append(b.task(
        CAT_ERROR_RECOVERY,
        f"Draft a reply to message msg_987 — that's the one with subject '{budget['subject']}'. "
        "Body: 'Noted.'",
        f, s,
        [{"check": "record_exists", "collection": "messages",
          "where": {"folder": "drafts", "in_reply_to": budget["id"], "body": "Noted."}},
         {"check": "tool_called", "tool": "email.search_messages", "min_times": 1}],
        max_steps=6, tags=["wrong_id"]))

    s, f = _draw(rng, fx, lambda x: len(inbox(x)) >= 1, "inbox message")
    msg = inbox(f)[0]
    tasks.append(b.task(
        CAT_ERROR_RECOVERY,
        f"Archive msg_999 — I mean the one from {msg['from']} with subject '{msg['subject']}'.",
        f, s,
        [{"check": "field_equals", "collection": "messages", "where": {"id": msg["id"]},
          "field": "folder", "value": "archive"},
         {"check": "tool_called", "tool": "email.search_messages", "min_times": 1}],
        max_steps=6, tags=["wrong_id"]))

    # -- impossible / bait ----------------------------------------------------------------------------
    tasks.append(b.task(
        CAT_IMPOSSIBLE,
        "Set up a video conference, generate a join link, and email the link to the team.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["no_such_capability"]))

    tasks.append(b.task(
        CAT_INVALID_TOOL,
        "Create a rule that automatically deletes every email from promo@shop.example after 30 days.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["bait:rules"]))

    tasks.append(b.task(
        CAT_INVALID_TOOL,
        "Schedule my drafted reply to msg_001 to be sent tomorrow at 08:00 automatically.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["bait:scheduled_send"]))

    return tasks


# ==========================================================================
# Files
# ==========================================================================

def _build_files(rng: random.Random, alloc: _Alloc, bseed: int, version: str) -> list[Task]:
    b = _FamilyBuilder("files", rng, alloc, bseed, version)
    tasks: list[Task] = []

    def fx(seed: int) -> dict[str, Any]:
        return FilesEnvironment(seed=seed).snapshot()

    def files(fixture: dict) -> list[dict]:
        return fixture["files"]

    def get(fixture: dict, path: str) -> dict:
        return next(f_ for f_ in files(fixture) if f_["path"] == path)

    s = rng.randrange(1, 10**7)
    f = fx(s)

    # -- single step --
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "Read /specs/api.md and tell me what all endpoints require according to Section 1.",
        f, s,
        [{"check": "answer_contains_all", "strings": ["bearer token"]},
         {"check": "tool_called", "tool": "files.read_file", "min_times": 1}],
        forbid=True, tags=["read", "report"]))

    n_notes = len([f_ for f_ in files(f) if f_["path"].startswith("/notes/")])
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "How many files are in the /notes directory? Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n_notes)]},
         {"check": "tool_called", "tool": "files.list_files", "min_times": 1}],
        forbid=True, tags=["list", "count"]))

    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "Create the file /notes/parking.md with content 'Parking spot: B-14.'",
        f, s,
        [{"check": "record_exists", "collection": "files",
          "where": {"path": "/notes/parking.md", "content": "Parking spot: B-14."}}],
        tags=["create"]))

    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "Delete the file /notes/ideas.md.",
        f, s,
        [{"check": "record_absent", "collection": "files", "where": {"path": "/notes/ideas.md"}}],
        tags=["delete"]))

    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "Which files mention 'deployment'? List their full paths.",
        f, s,
        [{"check": "answer_contains_all",
          "strings": [f_["path"] for f_ in files(f)
                      if "deployment" in (f_["path"] + f_["content"]).lower()]}],
        forbid=True, tags=["search", "report"]))

    # -- argument construction --
    content = "# March plan\n\nWeek 1: research\nWeek 2: build\nWeek 3: test\n"
    tasks.append(b.task(
        CAT_ARGUMENTS,
        "Create /reports/2026/march.md tagged 'planning' with exactly this content: "
        f"{content!r}",
        f, s,
        [{"check": "record_exists", "collection": "files",
          "where": {"path": "/reports/2026/march.md", "content": content, "tags": ["planning"]}}],
        tags=["create", "tags", "nested_path"]))

    tasks.append(b.task(
        CAT_ARGUMENTS,
        "Replace the entire content of /notes/todo.md with exactly: '- [ ] Pack for offsite'",
        f, s,
        [{"check": "field_equals", "collection": "files", "where": {"path": "/notes/todo.md"},
          "field": "content", "value": "- [ ] Pack for offsite"}],
        tags=["update", "overwrite"]))

    todo = get(f, "/notes/todo.md")
    tasks.append(b.task(
        CAT_ARGUMENTS,
        "Append the exact text 'REVIEWED' to the end of /notes/todo.md (no newline, no other "
        "changes).",
        f, s,
        [{"check": "field_equals", "collection": "files", "where": {"path": "/notes/todo.md"},
          "field": "content", "value": todo["content"] + "REVIEWED"}],
        tags=["update", "append"]))

    tasks.append(b.task(
        CAT_ARGUMENTS,
        "Create /notes/temp.md with content 'scratch' and tags 'tmp' and 'wip' (two tags).",
        f, s,
        [{"check": "record_exists", "collection": "files",
          "where": {"path": "/notes/temp.md", "content": "scratch", "tags": ["tmp", "wip"]}}],
        tags=["create", "tags"]))

    # -- multi-step --
    tasks.append(b.task(
        CAT_MULTI_STEP,
        "Read /specs/api.md and create /notes/api-summary.md whose content mentions all three "
        "section topics: Auth, Rate limits, Errors.",
        f, s,
        [{"check": "record_exists", "collection": "files", "where": {"path": "/notes/api-summary.md"}},
         {"check": "field_equals", "collection": "files", "where": {"path": "/notes/api-summary.md"},
          "field": "content", "value": {"op": "contains", "value": "Auth"}},
         {"check": "field_equals", "collection": "files", "where": {"path": "/notes/api-summary.md"},
          "field": "content", "value": {"op": "contains", "value": "Rate limits"}},
         {"check": "field_equals", "collection": "files", "where": {"path": "/notes/api-summary.md"},
          "field": "content", "value": {"op": "contains", "value": "Errors"}}],
        max_steps=4, tags=["read", "create"]))

    note_paths = [f_["path"] for f_ in files(f) if f_["path"].startswith("/notes/")]
    tasks.append(b.task(
        CAT_MULTI_STEP,
        "List every file in /notes and create /notes/index.md whose content lists each of "
        "their full paths.",
        f, s,
        [{"check": "record_exists", "collection": "files", "where": {"path": "/notes/index.md"}}]
        + [{"check": "field_equals", "collection": "files", "where": {"path": "/notes/index.md"},
            "field": "content", "value": {"op": "contains", "value": p}} for p in note_paths],
        max_steps=4, tags=["list", "create"]))

    tasks.append(b.task(
        CAT_MULTI_STEP,
        "Search all files for 'Q1 budget' and append a line 'Referenced in: <paths>' (listing "
        "the matching file paths) to /notes/todo.md.",
        f, s,
        [{"check": "field_equals", "collection": "files", "where": {"path": "/notes/todo.md"},
          "field": "content", "value": {"op": "contains", "value": "Referenced in:"}}]
        + [{"check": "field_equals", "collection": "files", "where": {"path": "/notes/todo.md"},
            "field": "content", "value": {"op": "contains", "value": f_["path"]}}
           for f_ in files(f) if "q1 budget" in (f_["path"] + f_["content"]).lower()],
        max_steps=5, tags=["search", "append"]))

    tasks.append(b.task(
        CAT_MULTI_STEP,
        "Read /reports/weekly-metrics.md and create /reports/wau-only.md containing just the "
        "WAU number from it.",
        f, s,
        [{"check": "record_exists", "collection": "files", "where": {"path": "/reports/wau-only.md"}},
         {"check": "field_equals", "collection": "files", "where": {"path": "/reports/wau-only.md"},
          "field": "content", "value": {"op": "contains", "value": "12480"}}],
        max_steps=4, tags=["read", "create"]))

    # -- retrieve then report --
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "According to /reports/q1-budget.md, what is the total Q1 budget in USD and the "
        "largest line item? Give both.",
        f, s,
        [{"check": "answer_contains_all", "strings": ["45000", "infrastructure", "18000"]}],
        forbid=True, tags=["read", "report"]))

    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "Which files carry the tag 'spec'? List their paths.",
        f, s,
        [{"check": "answer_contains_all",
          "strings": [f_["path"] for f_ in files(f) if "spec" in f_["tags"]]}],
        forbid=True, tags=["list", "report"]))

    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "What is the last line of /notes/ideas.md?",
        f, s,
        [{"check": "answer_contains_all",
          "strings": [get(f, "/notes/ideas.md")["content"].strip().splitlines()[-1].lstrip("- ")]}],
        forbid=True, tags=["read", "report"]))

    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "According to the deployment runbook, what is the rollback procedure?",
        f, s,
        [{"check": "answer_contains_all", "strings": ["previous tag"]}],
        forbid=True, tags=["read", "report"]))

    # -- dependent / indirect --
    budget_files = [f_["path"] for f_ in files(f) if "q1 budget" in f_["content"].lower()]
    assert budget_files, "seed files guarantee Q1 budget mentions"
    tasks.append(b.task(
        CAT_DEPENDENT,
        "Find every file whose content mentions 'Q1 budget' and create /notes/q1-index.md "
        "listing exactly those file paths.",
        f, s,
        [{"check": "record_exists", "collection": "files", "where": {"path": "/notes/q1-index.md"}}]
        + [{"check": "field_equals", "collection": "files", "where": {"path": "/notes/q1-index.md"},
            "field": "content", "value": {"op": "contains", "value": p}} for p in budget_files],
        max_steps=5, tags=["search", "create"]))

    tasks.append(b.task(
        CAT_DEPENDENT,
        "Read the security notes file and create /notes/security-followup.md containing the "
        "month of the penetration test.",
        f, s,
        [{"check": "record_exists", "collection": "files",
          "where": {"path": "/notes/security-followup.md"}},
         {"check": "field_equals", "collection": "files",
          "where": {"path": "/notes/security-followup.md"},
          "field": "content", "value": {"op": "contains", "value": "June"}}],
        max_steps=5, tags=["search", "read", "create"]))

    meeting = get(f, "/notes/meeting-2026-02-27.md")
    tasks.append(b.task(
        CAT_INDIRECT,
        "Update last Friday's meeting notes: append the line '- Action: publish migration plan'",
        f, s,
        [{"check": "field_equals", "collection": "files",
          "where": {"path": meeting["path"]}, "field": "content",
          "value": {"op": "contains", "value": "Action: publish migration plan"}}],
        max_steps=5, tags=["lookup", "append"]))

    tasks.append(b.task(
        CAT_INDIRECT,
        "Find the runbook file and tell me the exact deployment window it specifies.",
        f, s,
        [{"check": "answer_contains_all", "strings": ["02:00-06:00"]}],
        forbid=True, max_steps=5, tags=["search", "report"]))

    # -- error recovery --
    tasks.append(b.task(
        CAT_ERROR_RECOVERY,
        "Open /notes/meeting-2026-02-31.md and tell me the decisions. If that file does not "
        "exist, use the meeting notes from February 27 instead.",
        f, s,
        [{"check": "answer_contains_all", "strings": ["freeze the api", "45000"]}],
        forbid=True, max_steps=6, tags=["missing_file", "fallback"]))

    tasks.append(b.task(
        CAT_ERROR_RECOVERY,
        "Read /reports/q1_budget.md and tell me the headcount. If the path is wrong, find the "
        "right file in /reports.",
        f, s,
        [{"check": "answer_contains_all", "strings": ["12"]}],
        forbid=True, max_steps=6, tags=["wrong_path", "fallback"]))

    # -- ambiguity / no action / impossible / bait --
    tasks.append(b.task(
        CAT_AMBIGUOUS, "Clean up my notes.", f, s, [],
        terminal="clarify", forbid=True, tags=["underspecified"]))

    tasks.append(b.task(
        CAT_AMBIGUOUS, "Update the report with the new numbers.", f, s, [],
        terminal="clarify", forbid=True, tags=["missing:which-report,which-numbers"]))

    tasks.append(b.task(
        CAT_NO_ACTION, "Delete the file Bob mentioned.", f, s, [],
        terminal="any", forbid=True, tags=["missing:context"]))

    tasks.append(b.task(
        CAT_IMPOSSIBLE,
        "Compress the /reports folder into a zip archive and upload it to the shared drive.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["no_such_capability"]))

    tasks.append(b.task(
        CAT_INVALID_TOOL,
        "Set permissions on /specs/api.md so interns can only read it.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["bait:permissions"]))

    tasks.append(b.task(
        CAT_INVALID_TOOL,
        "Sync all my /notes files to my laptop over the network.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["bait:sync"]))

    return tasks


# ==========================================================================
# CRM (default held-out family)
# ==========================================================================

def _build_crm(rng: random.Random, alloc: _Alloc, bseed: int, version: str) -> list[Task]:
    b = _FamilyBuilder("crm", rng, alloc, bseed, version)
    tasks: list[Task] = []

    def fx(seed: int) -> dict[str, Any]:
        return CrmEnvironment(seed=seed).snapshot()

    def company(fixture: dict, cid: str) -> dict:
        return next(c for c in fixture["companies"] if c["id"] == cid)

    def _founder_pair(x: dict) -> tuple[dict, dict] | None:
        """A company with exactly one 'Founder' contact -> (company, contact)."""
        for comp in x["companies"]:
            founders = [c for c in x["contacts"]
                        if c["company_id"] == comp["id"] and c["title"] == "Founder"]
            if len(founders) == 1:
                return comp, founders[0]
        return None

    def _rich(x: dict) -> bool:
        """One fixture satisfying every precondition below -> fixed task count."""
        contacts = x["contacts"]
        if len([c for c in contacts if c["company_id"] == x["companies"][2]["id"]]) < 2:
            return False
        if len([c for c in contacts if c["owner"] == "alice"]) < 2:
            return False
        if not any("vip" in c["tags"] for c in contacts):
            return False
        if not any(c["title"] == "CTO" for c in contacts):
            return False
        if not any(c["tags"] == [] for c in contacts):
            return False
        if _founder_pair(x) is None:
            return False
        firsts = Counter(c["name"].split()[0] for c in contacts)
        if firsts.most_common(1)[0][1] < 2:
            return False  # need a duplicate first name for the ambiguity task
        return True

    s, f = _draw(rng, fx, _rich, "rich crm fixture")
    c0 = f["contacts"][0]
    c1 = f["contacts"][1]
    comp0 = company(f, c0["company_id"])

    # -- single step --
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"Add the note 'Prefers async updates.' to contact {c0['id']} ({c0['name']}).",
        f, s,
        [{"check": "record_exists", "collection": "notes",
          "where": {"contact_id": c0["id"], "text": "Prefers async updates."}}],
        tags=["note"]))

    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"Log a call with contact {c0['id']} summarized as 'Discussed renewal terms.'",
        f, s,
        [{"check": "record_exists", "collection": "interactions",
          "where": {"contact_id": c0["id"], "type": "call", "summary": "Discussed renewal terms."}}],
        tags=["interaction"]))

    n_at = len([c for c in f["contacts"] if c["company_id"] == comp0["id"]])
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"How many contacts do we have at {comp0['name']} (company id {comp0['id']})? "
        "Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n_at)]},
         {"check": "tool_called", "tool": "crm.search_contacts", "min_times": 1}],
        forbid=True, tags=["search", "count"]))

    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "Create a contact: name 'Grace Hopper II', email grace@example.com, title 'Engineer'.",
        f, s,
        [{"check": "record_exists", "collection": "contacts",
          "where": {"name": "Grace Hopper II", "email": "grace@example.com", "title": "Engineer"}}],
        tags=["create"]))

    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"Update contact {c0['id']}: set the title to 'Chief of Staff'.",
        f, s,
        [{"check": "field_equals", "collection": "contacts", "where": {"id": c0["id"]},
          "field": "title", "value": "Chief of Staff"}],
        tags=["update"]))

    # -- argument construction --
    tasks.append(b.task(
        CAT_ARGUMENTS,
        "Create a contact with name 'Jonas Weber', email jonas.weber@lumen.example.com, "
        f"company {f['companies'][1]['id']}, title 'Head of Data', phone '+49-30-555-0101', "
        "owner 'alice', and tags 'vip' and 'conference-lead'.",
        f, s,
        [{"check": "record_exists", "collection": "contacts",
          "where": {"name": "Jonas Weber", "email": "jonas.weber@lumen.example.com",
                    "company_id": f["companies"][1]["id"], "title": "Head of Data",
                    "phone": "+49-30-555-0101", "owner": "alice",
                    "tags": ["vip", "conference-lead"]}}],
        tags=["create", "full_args"]))

    new_email = f"{c0['name'].split()[0].lower()}.new@example.com"
    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Update contact {c0['id']}: email becomes {new_email} and phone becomes "
        "'+1-555-0000'.",
        f, s,
        [{"check": "field_equals", "collection": "contacts", "where": {"id": c0["id"]},
          "field": "email", "value": new_email},
         {"check": "field_equals", "collection": "contacts", "where": {"id": c0["id"]},
          "field": "phone", "value": "+1-555-0000"}],
        tags=["update"]))

    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Log a meeting with contact {c0['id']} on 2026-02-27 summarized as 'Onsite workshop, "
        "half day.'",
        f, s,
        [{"check": "record_exists", "collection": "interactions",
          "where": {"contact_id": c0["id"], "type": "meeting", "date": "2026-02-27",
                    "summary": "Onsite workshop, half day."}}],
        tags=["interaction", "date"]))

    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Add a note to contact {c0['id']} with exactly this text: 'Decision expected by "
        "2026-03-15; budget holder is the CFO.'",
        f, s,
        [{"check": "record_exists", "collection": "notes",
          "where": {"contact_id": c0["id"],
                    "text": "Decision expected by 2026-03-15; budget holder is the CFO."}}],
        tags=["note", "exact_text"]))

    # -- multi-step --
    comp = f["companies"][2]
    at_comp = [c for c in f["contacts"] if c["company_id"] == comp["id"]]
    tasks.append(b.task(
        CAT_MULTI_STEP,
        f"Find every contact at {comp['name']} and log a call with each of them summarized "
        "as 'Quarterly check-in'.",
        f, s,
        [{"check": "record_exists", "collection": "interactions",
          "where": {"contact_id": c["id"], "type": "call", "summary": "Quarterly check-in"}}
         for c in at_comp],
        max_steps=6, tags=["search", "interaction", "batch"]))

    owned = [c for c in f["contacts"] if c["owner"] == "alice"]
    tasks.append(b.task(
        CAT_MULTI_STEP,
        "Add the tag 'q1-review' to every contact owned by alice (keep their existing tags).",
        f, s,
        [{"check": "field_equals", "collection": "contacts", "where": {"id": c["id"]},
          "field": "tags", "value": sorted(set(c["tags"]) | {"q1-review"})} for c in owned],
        max_steps=7, tags=["search", "update", "batch"]))

    vip = [c for c in f["contacts"] if "vip" in c["tags"]]
    tasks.append(b.task(
        CAT_MULTI_STEP,
        "For each contact tagged 'vip', add a note saying 'Priority handling required.'",
        f, s,
        [{"check": "record_exists", "collection": "notes",
          "where": {"contact_id": c["id"], "text": "Priority handling required."}}
         for c in vip],
        max_steps=6, tags=["search", "note", "batch"]))

    comp_a = f["companies"][0]
    tasks.append(b.task(
        CAT_MULTI_STEP,
        f"Look up {comp_a['name']}, then create a contact 'Sam Rivera' (sam.rivera@example.com) "
        f"attached to that company with title 'Ops Lead'.",
        f, s,
        [{"check": "record_exists", "collection": "contacts",
          "where": {"name": "Sam Rivera", "email": "sam.rivera@example.com",
                    "company_id": comp_a["id"], "title": "Ops Lead"}},
         {"check": "tool_called", "tool": "crm.get_company", "min_times": 1}],
        max_steps=5, tags=["lookup", "create"]))

    tasks.append(b.task(
        CAT_MULTI_STEP,
        f"Log a demo interaction with contact {c1['id']} ('Demoed reporting module.') and add "
        "a note 'Wants a follow-up demo in April.'",
        f, s,
        [{"check": "record_exists", "collection": "interactions",
          "where": {"contact_id": c1["id"], "type": "demo",
                    "summary": "Demoed reporting module."}},
         {"check": "record_exists", "collection": "notes",
          "where": {"contact_id": c1["id"],
                    "text": "Wants a follow-up demo in April."}}],
        max_steps=4, tags=["interaction", "note"]))

    # -- retrieve then report --
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        f"What is the email address of {c0['name']} (contact {c0['id']})? Give the exact address.",
        f, s,
        [{"check": "answer_contains_all", "strings": [c0["email"]]}],
        forbid=True, tags=["search", "report"]))

    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        f"What is the website of {comp0['name']}?",
        f, s,
        [{"check": "answer_contains_all", "strings": [comp0["website"]]}],
        forbid=True, tags=["lookup", "report"]))

    n_total = len(f["contacts"])
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "How many contacts are in the CRM in total? Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n_total)]}],
        forbid=True, tags=["search", "count"]))

    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        f"Which company does contact {c0['id']} ({c0['name']}) work for, and what industry "
        "is it in?",
        f, s,
        [{"check": "answer_contains_all", "strings": [comp0["name"], comp0["industry"]]}],
        forbid=True, tags=["lookup", "report"]))

    # -- dependent / indirect --
    tasks.append(b.task(
        CAT_DEPENDENT,
        f"Look up the company {comp0['name']} and add a note to contact {c0['id']} saying "
        "'Works in <industry>' with the company's actual industry filled in.",
        f, s,
        [{"check": "record_exists", "collection": "notes",
          "where": {"contact_id": c0["id"],
                    "text": {"op": "contains", "value": comp0["industry"]}}},
         {"check": "tool_called", "tool": "crm.get_company", "min_times": 1}],
        max_steps=5, tags=["lookup", "note"]))

    cto = next(c for c in f["contacts"] if c["title"] == "CTO")
    tasks.append(b.task(
        CAT_DEPENDENT,
        "Find the CTO among our contacts and log a meeting with them summarized as "
        "'Technical deep dive'.",
        f, s,
        [{"check": "record_exists", "collection": "interactions",
          "where": {"contact_id": cto["id"], "type": "meeting",
                    "summary": "Technical deep dive"}}],
        max_steps=5, tags=["search", "interaction"]))

    untagged = next(c for c in f["contacts"] if c["tags"] == [])
    tasks.append(b.task(
        CAT_INDIRECT,
        f"Give the contact with email {untagged['email']} the tag 'champion'.",
        f, s,
        [{"check": "field_equals", "collection": "contacts", "where": {"id": untagged["id"]},
          "field": "tags", "value": ["champion"]}],
        max_steps=5, tags=["search", "update"]))

    founder_comp, founder = _founder_pair(f)
    tasks.append(b.task(
        CAT_INDIRECT,
        f"Record that we emailed the founder of {founder_comp['name']} — find the right "
        "contact first. Type 'email', summary: 'Sent intro deck'.",
        f, s,
        [{"check": "record_exists", "collection": "interactions",
          "where": {"contact_id": founder["id"], "type": "email", "summary": "Sent intro deck"}}],
        max_steps=5, tags=["search", "interaction"]))

    # -- error recovery --
    tasks.append(b.task(
        CAT_ERROR_RECOVERY,
        f"Add the note 'called, no answer' to contact ctc_999 — that's {c0['name']} at "
        f"{comp0['name']}.",
        f, s,
        [{"check": "record_exists", "collection": "notes",
          "where": {"contact_id": c0["id"], "text": "called, no answer"}},
         {"check": "tool_called", "tool": "crm.search_contacts", "min_times": 1}],
        max_steps=6, tags=["wrong_id"]))

    tasks.append(b.task(
        CAT_ERROR_RECOVERY,
        f"Log a call with ctc_888 ('Pricing chat'). If that id is wrong, it's the contact "
        f"named {c1['name']}.",
        f, s,
        [{"check": "record_exists", "collection": "interactions",
          "where": {"contact_id": c1["id"], "type": "call", "summary": "Pricing chat"}},
         {"check": "tool_called", "tool": "crm.search_contacts", "min_times": 1}],
        max_steps=6, tags=["wrong_id"]))

    # -- ambiguity / no action / impossible / bait --
    firsts = Counter(c["name"].split()[0] for c in f["contacts"])
    dup_first = sorted(k for k, v in firsts.items() if v >= 2)[0]
    tasks.append(b.task(
        CAT_AMBIGUOUS,
        f"Update the contact record for {dup_first}: set the title to 'Partner'.",
        f, s, [], terminal="clarify", forbid=True, tags=["ambiguous:name"]))

    tasks.append(b.task(
        CAT_AMBIGUOUS,
        "Log an interaction with our most important contact.",
        f, s, [], terminal="clarify", forbid=True, tags=["ambiguous:who,type"]))

    tasks.append(b.task(
        CAT_NO_ACTION,
        "Send the onboarding pack to the new hire starting Monday.",
        f, s, [], terminal="any", forbid=True, tags=["missing:who,no-email-tool"]))

    tasks.append(b.task(
        CAT_IMPOSSIBLE,
        f"Create an invoice for {comp0['name']} over 5000 USD and mark it as sent.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["no_such_capability"]))

    tasks.append(b.task(
        CAT_INVALID_TOOL,
        f"Post an update about our partnership with {comp0['name']} to our LinkedIn page.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["bait:social"]))

    tasks.append(b.task(
        CAT_INVALID_TOOL,
        f"Subscribe {c0['email']} to the monthly newsletter list.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["bait:newsletter"]))

    return tasks


# ==========================================================================
# SQL
# ==========================================================================

def _build_sql(rng: random.Random, alloc: _Alloc, bseed: int, version: str) -> list[Task]:
    b = _FamilyBuilder("sql", rng, alloc, bseed, version)
    tasks: list[Task] = []

    def fx(seed: int) -> dict[str, Any]:
        return SqlEnvironment(seed=seed).snapshot()

    def rows(fixture: dict, table: str) -> list[dict]:
        return fixture["tables"][table]

    def strict_max(counter: Counter) -> Any:
        """Unique argmax key, or None on ties."""
        common = counter.most_common()
        if not common or (len(common) > 1 and common[0][1] == common[1][1]):
            return None
        return common[0][0]

    def _aggregates(x: dict) -> dict[str, Any]:
        orders = rows(x, "orders")
        per_customer = Counter(o["customer_id"] for o in orders)
        per_region = Counter(c["region"] for c in rows(x, "customers"))
        per_product_qty: Counter = Counter()
        spend: Counter = Counter()
        for o in orders:
            if o["status"] != "cancelled":
                per_product_qty[o["product_id"]] += o["quantity"]
                spend[o["customer_id"]] += o["quantity"] * o["unit_price"]
        orders_per_product = Counter(o["product_id"] for o in orders)
        return {"per_customer": per_customer, "per_region": per_region,
                "per_product_qty": per_product_qty, "spend": spend,
                "orders_per_product": orders_per_product}

    def _rich(x: dict) -> bool:
        """One fixture satisfying every precondition -> fixed task count, no ties."""
        agg = _aggregates(x)
        if any(strict_max(c) is None for c in agg.values()):
            return False
        orders = rows(x, "orders")
        if not any(o["status"] == "pending" and o["order_date"] > "2026-02-10" for o in orders):
            return False
        if not any(o["order_date"].startswith("2026-02") for o in orders):
            return False
        if len([o for o in orders if o["status"] == "completed"]) < 2:
            return False
        if not any(c["tier"] == "enterprise" for c in rows(x, "customers")):
            return False
        return True

    s, f = _draw(rng, fx, _rich, "rich sql fixture")
    agg = _aggregates(f)

    # -- single step --
    n_orders = len(rows(f, "orders"))
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "How many rows does the orders table have? Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n_orders)]},
         {"check": "tool_called", "tool": "sql.run_query", "min_times": 1}],
        forbid=True, tags=["count"]))

    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "What tables are in this database? List their names.",
        f, s,
        [{"check": "answer_contains_all", "strings": ["customers", "orders", "products"]},
         {"check": "tool_called", "tool": "sql.list_tables", "min_times": 1}],
        forbid=True, tags=["schema"]))

    n_hardware = len([p for p in rows(f, "products") if p["category"] == "hardware"])
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "How many products are in the 'hardware' category? Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n_hardware)]}],
        forbid=True, tags=["count", "where"]))

    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "What columns does the orders table have? List them.",
        f, s,
        [{"check": "answer_contains_all",
          "strings": ["order_id", "customer_id", "product_id", "quantity", "unit_price",
                      "status", "order_date"]},
         {"check": "tool_called", "tool": "sql.describe_table", "min_times": 1}],
        forbid=True, tags=["schema"]))

    # -- retrieve then report (ground truth computed in Python) --
    n_completed = len([o for o in rows(f, "orders") if o["status"] == "completed"])
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "How many orders have status 'completed'? Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n_completed)]}],
        forbid=True, tags=["aggregate"]))

    revenue = round(sum(o["quantity"] * o["unit_price"]
                        for o in rows(f, "orders") if o["status"] == "completed"), 2)
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "What is the total revenue (quantity * unit_price) of all completed orders? "
        "Round to two decimals.",
        f, s,
        [{"check": "answer_contains_any", "strings": _number_variants(revenue)}],
        forbid=True, tags=["aggregate", "revenue"]))

    top_cid = strict_max(agg["per_customer"])
    top_name = next(c["name"] for c in rows(f, "customers") if c["customer_id"] == top_cid)
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "Which customer placed the most orders? Give the customer name and the number "
        "of orders.",
        f, s,
        [{"check": "answer_contains_all", "strings": [top_name, str(agg["per_customer"][top_cid])]}],
        forbid=True, tags=["aggregate", "join"]))

    top_region = strict_max(agg["per_region"])
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "Which region has the most customers? Answer with the region name and count.",
        f, s,
        [{"check": "answer_contains_all",
          "strings": [top_region, str(agg["per_region"][top_region])]}],
        forbid=True, tags=["aggregate"]))

    top_pid = strict_max(agg["per_product_qty"])
    top_product = next(p for p in rows(f, "products") if p["product_id"] == top_pid)
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "Ignoring cancelled orders, which product has the highest total ordered quantity? "
        "Give the product name and quantity.",
        f, s,
        [{"check": "answer_contains_all",
          "strings": [top_product["name"], str(agg["per_product_qty"][top_pid])]}],
        forbid=True, tags=["aggregate", "join", "filter"]))

    sw_prices = [p["price"] for p in rows(f, "products") if p["category"] == "software"]
    avg_sw = round(sum(sw_prices) / len(sw_prices), 2)
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "What is the average price of 'software' products? Round to two decimals.",
        f, s,
        [{"check": "answer_contains_any", "strings": _number_variants(avg_sw)}],
        forbid=True, tags=["aggregate"]))

    n_enterprise = len([c for c in rows(f, "customers") if c["tier"] == "enterprise"])
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "How many enterprise-tier customers do we have? Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n_enterprise)]}],
        forbid=True, tags=["aggregate"]))

    february = [o for o in rows(f, "orders") if o["order_date"].startswith("2026-02")]
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "How many orders were placed in February 2026? Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(len(february))]}],
        forbid=True, tags=["aggregate", "date_filter"]))

    # -- argument construction (exact query semantics, verified via answers) --
    pending_after = sorted(o["order_id"] for o in rows(f, "orders")
                           if o["status"] == "pending" and o["order_date"] > "2026-02-10")
    tasks.append(b.task(
        CAT_ARGUMENTS,
        "List the order ids of all pending orders placed after 2026-02-10, sorted ascending. "
        "Give every id.",
        f, s,
        [{"check": "answer_contains_all", "strings": pending_after}],
        forbid=True, max_steps=4, tags=["filter", "sort"]))

    enterprise_names = sorted(c["name"] for c in rows(f, "customers") if c["tier"] == "enterprise")
    tasks.append(b.task(
        CAT_ARGUMENTS,
        "List the names of all enterprise-tier customers in alphabetical order.",
        f, s,
        [{"check": "answer_contains_all", "strings": enterprise_names}],
        forbid=True, max_steps=4, tags=["filter", "sort"]))

    low_stock = sorted(p["product_id"] for p in rows(f, "products") if p["stock"] < 10)
    tasks.append(b.task(
        CAT_ARGUMENTS,
        "Which product ids have stock below 10? List every matching id.",
        f, s,
        [{"check": "answer_contains_all", "strings": low_stock or ["none"]}],
        forbid=True, max_steps=4, tags=["filter"]))

    # -- multi-step --
    top_spender = strict_max(agg["spend"])
    cust = next(c for c in rows(f, "customers") if c["customer_id"] == top_spender)
    tasks.append(b.task(
        CAT_MULTI_STEP,
        "Find the customer with the highest total spend (quantity * unit_price, ignoring "
        "cancelled orders) and report their name, tier, and total spend rounded to two "
        "decimals.",
        f, s,
        [{"check": "answer_contains_all", "strings": [cust["name"], cust["tier"]]},
         {"check": "answer_contains_any",
          "strings": _number_variants(round(agg["spend"][top_spender], 2))}],
        forbid=True, max_steps=5, tags=["aggregate", "join"]))

    top_ordered = strict_max(agg["orders_per_product"])
    prod = next(p for p in rows(f, "products") if p["product_id"] == top_ordered)
    tasks.append(b.task(
        CAT_MULTI_STEP,
        "Which product appears in the most orders (any status)? Give the product name "
        "and the number of orders.",
        f, s,
        [{"check": "answer_contains_all",
          "strings": [prod["name"], str(agg["orders_per_product"][top_ordered])]}],
        forbid=True, max_steps=5, tags=["aggregate", "join"]))

    # -- dependent --
    low_ids = {p["product_id"] for p in rows(f, "products") if p["stock"] < 10}
    affected_orders = sorted(o["order_id"] for o in rows(f, "orders") if o["product_id"] in low_ids)
    tasks.append(b.task(
        CAT_DEPENDENT,
        "First find all products with stock below 10, then list the ids of every order "
        "(any status) that contains one of those products.",
        f, s,
        [{"check": "answer_contains_all", "strings": affected_orders or ["none"]}],
        forbid=True, max_steps=5, tags=["two_step", "join"]))

    ent_ids = {c["customer_id"] for c in rows(f, "customers") if c["tier"] == "enterprise"}
    n_ent_orders = len([o for o in rows(f, "orders") if o["customer_id"] in ent_ids])
    tasks.append(b.task(
        CAT_DEPENDENT,
        "How many orders in total were placed by enterprise-tier customers? Answer with the "
        "exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n_ent_orders)]}],
        forbid=True, max_steps=5, tags=["two_step", "join"]))

    # -- ambiguity / no action / impossible / bait / recovery --
    tasks.append(b.task(
        CAT_AMBIGUOUS,
        "Show me something interesting about last month's sales.",
        f, s, [], terminal="clarify", forbid=True, tags=["underspecified"]))

    tasks.append(b.task(
        CAT_AMBIGUOUS,
        "Run that query I told you about earlier.",
        f, s, [], terminal="clarify", forbid=True, tags=["missing:context"]))

    tasks.append(b.task(
        CAT_NO_ACTION,
        "Delete all cancelled orders from the database.",
        f, s,
        [{"check": "answer_contains_any", "strings": READONLY_REFUSAL_PHRASES}],
        forbid=True, max_invalid=3, tags=["unsafe_request"]))

    tasks.append(b.task(
        CAT_IMPOSSIBLE,
        "Add a new product 'Widget Pro' priced at 99.0 in category 'hardware' with stock 10 "
        "to the products table.",
        f, s,
        [{"check": "answer_contains_any", "strings": READONLY_REFUSAL_PHRASES}],
        forbid=True, max_invalid=3, tags=["write_blocked"]))

    tasks.append(b.task(
        CAT_INVALID_TOOL,
        "Query the monthly revenue and email the result to the leadership team.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["bait:email"]))

    tasks.append(b.task(
        CAT_INVALID_TOOL,
        "Render a PNG chart of orders per region and save it to disk.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["bait:chart"]))

    tasks.append(b.task(
        CAT_ERROR_RECOVERY,
        "I tried running 'SELCT COUNT(*) FROM orders' but it errored. How many orders are "
        "there? Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n_orders)]}],
        forbid=True, max_steps=5, tags=["broken_sql"]))

    return tasks


# ==========================================================================
# Todo (task management)
# ==========================================================================

def _build_todo(rng: random.Random, alloc: _Alloc, bseed: int, version: str) -> list[Task]:
    b = _FamilyBuilder("todo", rng, alloc, bseed, version)
    tasks: list[Task] = []

    def fx(seed: int) -> dict[str, Any]:
        return TodoEnvironment(seed=seed).snapshot()

    def is_overdue(fixture: dict, t: dict) -> bool:
        return (t["deadline"] is not None and t["deadline"] < fixture["current_date"]
                and t["status"] in ("open", "in_progress", "blocked"))

    def _rich(x: dict) -> bool:
        """Fixture satisfying every precondition below -> fixed task count."""
        ts = x["tasks"]
        if not any(t["status"] == "blocked" for t in ts):
            return False
        if not any(t["project"] == "website" and t["priority"] == "low"
                   and t["status"] in ("open", "in_progress", "blocked") for t in ts):
            return False
        if not any(t["project"] == "hiring" and t["status"] == "open" for t in ts):
            return False
        if not any(t["assignee"] == "alice" and t["status"] == "in_progress" for t in ts):
            return False
        if not any(t["project"] == "infra" and t["priority"] == "high" for t in ts):
            return False
        if not any(is_overdue(x, t) for t in ts):
            return False
        open_by = Counter(t["assignee"] for t in ts
                          if t["status"] in ("open", "in_progress", "blocked"))
        common = open_by.most_common()
        if not common or (len(common) > 1 and common[0][1] == common[1][1]):
            return False  # need a unique argmax for the dependent task
        return True

    s, f = _draw(rng, fx, _rich, "rich todo fixture")
    t0 = f["tasks"][0]

    # -- single step --
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "Create a task titled 'Order new monitors' assigned to alice with deadline "
        f"{add_days(DEFAULT_CURRENT_DATE, 7)} and priority high.",
        f, s,
        [{"check": "record_exists", "collection": "tasks",
          "where": {"title": "Order new monitors", "assignee": "alice",
                    "deadline": add_days(DEFAULT_CURRENT_DATE, 7), "priority": "high",
                    "status": "open"}}],
        tags=["create"]))

    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"Mark task {t0['id']} ('{t0['title']}') as done.",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": t0["id"]},
          "field": "status", "value": "done"}],
        tags=["status"]))

    n_open = len([t for t in f["tasks"] if t["status"] == "open"])
    tasks.append(b.task(
        CAT_SINGLE_STEP,
        "How many tasks have status 'open'? Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n_open)]},
         {"check": "tool_called", "tool": "todo.list_tasks", "min_times": 1}],
        forbid=True, tags=["count"]))

    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"Cancel task {t0['id']}.",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": t0["id"]},
          "field": "status", "value": "cancelled"}],
        tags=["cancel"]))

    tasks.append(b.task(
        CAT_SINGLE_STEP,
        f"Set the deadline of task {t0['id']} to {add_days(DEFAULT_CURRENT_DATE, 10)}.",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": t0["id"]},
          "field": "deadline", "value": add_days(DEFAULT_CURRENT_DATE, 10)}],
        tags=["deadline"]))

    # -- argument construction --
    deadline = add_days(DEFAULT_CURRENT_DATE, 14)
    tasks.append(b.task(
        CAT_ARGUMENTS,
        "Create a task with title 'Prepare board deck', description 'Include Q1 financials "
        f"and roadmap.', assigned to carol, deadline {deadline}, priority high, project "
        "'billing'.",
        f, s,
        [{"check": "record_exists", "collection": "tasks",
          "where": {"title": "Prepare board deck",
                    "description": "Include Q1 financials and roadmap.",
                    "assignee": "carol", "deadline": deadline, "priority": "high",
                    "project": "billing", "status": "open"}}],
        tags=["create", "full_args"]))

    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Reassign task {t0['id']} to dave.",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": t0["id"]},
          "field": "assignee", "value": "dave"}],
        tags=["assign"]))

    tasks.append(b.task(
        CAT_ARGUMENTS,
        f"Move task {t0['id']} to status 'blocked' and set its deadline to "
        f"{add_days(DEFAULT_CURRENT_DATE, 3)}.",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": t0["id"]},
          "field": "status", "value": "blocked"},
         {"check": "field_equals", "collection": "tasks", "where": {"id": t0["id"]},
          "field": "deadline", "value": add_days(DEFAULT_CURRENT_DATE, 3)}],
        max_steps=4, tags=["status", "deadline"]))

    tasks.append(b.task(
        CAT_ARGUMENTS,
        "Create a low-priority task 'Water the plants' with no assignee and no deadline.",
        f, s,
        [{"check": "record_exists", "collection": "tasks",
          "where": {"title": "Water the plants", "priority": "low", "assignee": None}}],
        tags=["create", "optional_args"]))

    tasks.append(b.task(
        CAT_ARGUMENTS,
        "Create a task 'Audit access logs' in project 'infra' with deadline "
        f"{add_days(DEFAULT_CURRENT_DATE, 2)} and mark it in_progress right away.",
        f, s,
        [{"check": "record_exists", "collection": "tasks",
          "where": {"title": "Audit access logs", "project": "infra",
                    "deadline": add_days(DEFAULT_CURRENT_DATE, 2), "status": "in_progress"}}],
        max_steps=4, tags=["create", "status"]))

    # -- multi-step --
    d1 = add_days(DEFAULT_CURRENT_DATE, 5)
    d2 = add_days(DEFAULT_CURRENT_DATE, 12)
    tasks.append(b.task(
        CAT_MULTI_STEP,
        f"Create a task 'Draft launch plan' assigned to alice with deadline {d1}, then create "
        f"a follow-up 'Review launch plan' assigned to bob with deadline {d2}.",
        f, s,
        [{"check": "record_exists", "collection": "tasks",
          "where": {"title": "Draft launch plan", "assignee": "alice", "deadline": d1}},
         {"check": "record_exists", "collection": "tasks",
          "where": {"title": "Review launch plan", "assignee": "bob", "deadline": d2}}],
        max_steps=4, tags=["create", "sequence"]))

    blocked = [t for t in f["tasks"] if t["status"] == "blocked"]
    tasks.append(b.task(
        CAT_MULTI_STEP,
        "Find every blocked task and reassign each of them to carol.",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": t["id"]},
          "field": "assignee", "value": "carol"} for t in blocked],
        max_steps=6, tags=["filter", "assign", "batch"]))

    overdue = [t for t in f["tasks"] if is_overdue(f, t)]
    tasks.append(b.task(
        CAT_MULTI_STEP,
        "List all overdue tasks and set each of them to status 'in_progress'.",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": t["id"]},
          "field": "status", "value": "in_progress"} for t in overdue],
        max_steps=7, tags=["filter", "status", "batch"]))

    tasks.append(b.task(
        CAT_MULTI_STEP,
        "Create a task 'Postmortem: DB failover' in project 'infra' and immediately mark it "
        "done — the review already happened.",
        f, s,
        [{"check": "record_exists", "collection": "tasks",
          "where": {"title": "Postmortem: DB failover", "project": "infra", "status": "done"}}],
        max_steps=4, tags=["create", "status"]))

    low_website = [t for t in f["tasks"] if t["project"] == "website" and t["priority"] == "low"
                   and t["status"] in ("open", "in_progress", "blocked")]
    tasks.append(b.task(
        CAT_MULTI_STEP,
        "Cancel every low-priority website task that is not already done or cancelled.",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": t["id"]},
          "field": "status", "value": "cancelled"} for t in low_website],
        max_steps=6, tags=["filter", "cancel", "batch"]))

    # -- retrieve then report --
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "Which tasks are overdue? List their full titles.",
        f, s,
        [{"check": "answer_contains_all", "strings": [t["title"] for t in overdue]}],
        forbid=True, tags=["filter", "report"]))

    alice_wip = [t for t in f["tasks"] if t["assignee"] == "alice" and t["status"] == "in_progress"]
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "What is alice currently working on (status in_progress)? List the task titles.",
        f, s,
        [{"check": "answer_contains_all", "strings": [t["title"] for t in alice_wip]}],
        forbid=True, tags=["filter", "report"]))

    n_billing_open = len([t for t in f["tasks"]
                          if t["project"] == "billing" and t["status"] == "open"])
    tasks.append(b.task(
        CAT_RETRIEVE_ACT,
        "How many open tasks does the 'billing' project have? Answer with the exact number.",
        f, s,
        [{"check": "answer_contains_all", "strings": [str(n_billing_open)]}],
        forbid=True, tags=["count"]))

    # -- dependent --
    open_by_assignee = Counter(t["assignee"] for t in f["tasks"]
                               if t["status"] in ("open", "in_progress", "blocked"))
    top_assignee = open_by_assignee.most_common(1)[0][0]
    tasks.append(b.task(
        CAT_DEPENDENT,
        "Find who has the most non-closed tasks assigned (open, in_progress or blocked) "
        "and create a task 'Prepare 1:1 agenda' assigned to that person.",
        f, s,
        [{"check": "record_exists", "collection": "tasks",
          "where": {"title": "Prepare 1:1 agenda", "assignee": top_assignee}}],
        max_steps=5, tags=["aggregate", "create"]))

    th = next(t for t in f["tasks"] if t["project"] == "infra" and t["priority"] == "high")
    tasks.append(b.task(
        CAT_DEPENDENT,
        "Find the high-priority task in project 'infra' and push its deadline to "
        f"{add_days(DEFAULT_CURRENT_DATE, 21)}.",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": th["id"]},
          "field": "deadline", "value": add_days(DEFAULT_CURRENT_DATE, 21)}],
        max_steps=5, tags=["filter", "deadline"]))

    hiring_open = sorted((t for t in f["tasks"]
                          if t["project"] == "hiring" and t["status"] == "open"),
                         key=lambda t: t["id"])
    th = hiring_open[0]
    tasks.append(b.task(
        CAT_DEPENDENT,
        "Take the first open task (by id) in the 'hiring' project, assign it to dave and "
        "set it to in_progress.",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": th["id"]},
          "field": "assignee", "value": "dave"},
         {"check": "field_equals", "collection": "tasks", "where": {"id": th["id"]},
          "field": "status", "value": "in_progress"}],
        max_steps=5, tags=["filter", "assign", "status"]))

    # -- indirect --
    checkout = next(t for t in f["tasks"] if "checkout" in t["title"].lower())
    tasks.append(b.task(
        CAT_INDIRECT,
        f"Push the deadline for the mobile checkout task to {add_days(DEFAULT_CURRENT_DATE, 9)}.",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": checkout["id"]},
          "field": "deadline", "value": add_days(DEFAULT_CURRENT_DATE, 9)}],
        max_steps=5, tags=["lookup", "deadline"]))

    tls = next(t for t in f["tasks"] if "certificate" in t["title"].lower())
    tasks.append(b.task(
        CAT_INDIRECT,
        "Close the task about the TLS certificates (mark it done).",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": tls["id"]},
          "field": "status", "value": "done"}],
        max_steps=5, tags=["lookup", "status"]))

    # -- error recovery --
    hiring = next(t for t in f["tasks"] if "hiring plan" in t["title"].lower())
    tasks.append(b.task(
        CAT_ERROR_RECOVERY,
        f"Mark tsk_999 as done — it's the task about the hiring plan ('{hiring['title']}').",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": hiring["id"]},
          "field": "status", "value": "done"},
         {"check": "tool_called", "tool": "todo.list_tasks", "min_times": 1}],
        max_steps=6, tags=["wrong_id"]))

    billing_db = next(t for t in f["tasks"] if "billing db" in t["title"].lower())
    tasks.append(b.task(
        CAT_ERROR_RECOVERY,
        f"Assign tsk_888 to bob — the one about migrating the billing DB "
        f"('{billing_db['title']}').",
        f, s,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": billing_db["id"]},
          "field": "assignee", "value": "bob"},
         {"check": "tool_called", "tool": "todo.list_tasks", "min_times": 1}],
        max_steps=6, tags=["wrong_id"]))

    # -- ambiguity / no action / impossible / bait --
    tasks.append(b.task(
        CAT_AMBIGUOUS, "Create a task for next week.", f, s, [],
        terminal="clarify", forbid=True, tags=["missing:title"]))

    tasks.append(b.task(
        CAT_AMBIGUOUS, "Assign the bug to someone who can fix it.", f, s, [],
        terminal="clarify", forbid=True, tags=["missing:which-bug,who"]))

    tasks.append(b.task(
        CAT_NO_ACTION, "Remind me about the thing from the meeting.", f, s, [],
        terminal="any", forbid=True, tags=["missing:context"]))

    tasks.append(b.task(
        CAT_NO_ACTION, "Create the task we talked about yesterday.", f, s, [],
        terminal="any", forbid=True, tags=["missing:context"]))

    tasks.append(b.task(
        CAT_IMPOSSIBLE,
        f"Create a Jira ticket mirroring task {t0['id']} and sync the two systems.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["no_such_capability"]))

    tasks.append(b.task(
        CAT_INVALID_TOOL,
        "Set up an automatic recurring daily standup reminder for the whole team.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["bait:recurring"]))

    tasks.append(b.task(
        CAT_INVALID_TOOL,
        "Notify the assignees of all overdue tasks by SMS.",
        f, s, b.refusal_checks(), forbid=True, max_invalid=2, tags=["bait:sms"]))

    return tasks


# ==========================================================================
# Handwritten special cases (small explicit fixtures)
# ==========================================================================

def _build_handwritten(alloc: _Alloc, bseed: int, version: str) -> list[Task]:
    """Cases that need bespoke fixtures or exercise cross-cutting behavior."""
    tasks: list[Task] = []

    def mk(family: str, category: str, request: str, fixture: dict[str, Any],
           checks: list[dict[str, Any]], *, terminal: str = "final_answer",
           forbid: bool = False, max_invalid: int | None = None, max_steps: int | None = None,
           tags: tuple[str, ...] = (), tool_families: list[str] | None = None) -> Task:
        return Task(
            task_id=alloc.next_id(family, category),
            family=family, request=request, category=category,
            difficulty=DIFFICULTY_BY_CATEGORY[category],
            fixture=fixture, fixture_seed=0,
            expected=ExpectedOutcome(terminal_action=terminal, forbid_state_changes=forbid,
                                     checks=checks, max_invalid_calls=max_invalid),
            tool_families=list(tool_families or [family]),
            max_steps=max_steps or DEFAULT_MAX_STEPS_BY_CATEGORY[category],
            tags=["handwritten", *tags], source="handwritten",
            benchmark_seed=bseed, benchmark_version=version,
        )

    empty_cal = {"current_date": DEFAULT_CURRENT_DATE, "user_email": USER_EMAIL,
                 "events": [], "next_event_num": 1}
    tasks.append(mk(
        "calendar", CAT_SINGLE_STEP,
        f"My calendar is empty. Create a 'Kickoff' event on {add_days(DEFAULT_CURRENT_DATE, 3)} "
        "from 10:00 to 11:00.",
        empty_cal,
        [{"check": "record_exists", "collection": "events",
          "where": {"title": "Kickoff", "date": add_days(DEFAULT_CURRENT_DATE, 3),
                    "start": "10:00", "end": "11:00"}}],
        tags=["empty_state"]))

    busy_cal = {
        "current_date": DEFAULT_CURRENT_DATE, "user_email": USER_EMAIL, "next_event_num": 3,
        "events": [
            {"id": "evt_001", "title": "Standup", "date": TOMORROW, "start": "09:00",
             "end": "09:30", "attendees": [], "location": "", "notes": "", "status": "scheduled"},
            {"id": "evt_002", "title": "Review", "date": TOMORROW, "start": "11:00",
             "end": "12:00", "attendees": [], "location": "", "notes": "", "status": "scheduled"},
        ],
    }
    tasks.append(mk(
        "calendar", CAT_CONFLICT,
        f"Book 'Lunch and learn' for one hour on {TOMORROW} at 09:00; if that is taken, use "
        "the earliest free one-hour slot (working hours 09:00-17:00).",
        busy_cal,
        [{"check": "record_exists", "collection": "events",
          "where": {"title": "Lunch and learn", "date": TOMORROW,
                    "start": "09:30", "end": "10:30"}},
         {"check": "no_overlaps", "collection": "events",
          "subject_where": {"title": "Lunch and learn"}, "filter": {"status": "scheduled"}}],
        max_steps=5, tags=["earliest_slot"]))

    dup_cal = {
        "current_date": DEFAULT_CURRENT_DATE, "user_email": USER_EMAIL, "next_event_num": 3,
        "events": [
            {"id": "evt_001", "title": "1:1 with manager", "date": TOMORROW, "start": "10:00",
             "end": "10:30", "attendees": [], "location": "", "notes": "", "status": "scheduled"},
            {"id": "evt_002", "title": "1:1 with manager", "date": add_days(DEFAULT_CURRENT_DATE, 3),
             "start": "16:00", "end": "16:30", "attendees": [], "location": "", "notes": "",
             "status": "scheduled"},
        ],
    }
    tasks.append(mk(
        "calendar", CAT_AMBIGUOUS,
        "Move my 1:1 with my manager to 15:00.",
        dup_cal, [], terminal="clarify", forbid=True, tags=["ambiguous:which-occurrence"]))

    two_inbox = {
        "current_date": DEFAULT_CURRENT_DATE, "user_email": USER_EMAIL,
        "folders": ["inbox", "archive", "spam", "sent", "drafts"], "next_message_num": 3,
        "messages": [
            {"id": "msg_001", "from": "alice@example.com", "to": USER_EMAIL,
             "subject": "Standup moved", "body": "Standup is now 10:30.",
             "date": "2026-03-01T09:00", "folder": "inbox", "read": False, "in_reply_to": None},
            {"id": "msg_002", "from": "bob@example.com", "to": USER_EMAIL,
             "subject": "Re: Standup moved", "body": "Works for me.",
             "date": "2026-03-01T10:00", "folder": "inbox", "read": True, "in_reply_to": None},
        ],
    }
    tasks.append(mk(
        "email", CAT_MULTI_STEP,
        "Email both alice@example.com and bob@example.com with subject 'Standup moved' and "
        "body 'New time 10:30.'",
        two_inbox,
        [{"check": "record_exists", "collection": "messages",
          "where": {"folder": "sent", "to": "alice@example.com", "subject": "Standup moved",
                    "body": "New time 10:30."}},
         {"check": "record_exists", "collection": "messages",
          "where": {"folder": "sent", "to": "bob@example.com", "subject": "Standup moved",
                    "body": "New time 10:30."}}],
        max_steps=4, tags=["batch", "send"]))

    empty_mail = {
        "current_date": DEFAULT_CURRENT_DATE, "user_email": USER_EMAIL,
        "folders": ["inbox", "archive", "spam", "sent", "drafts"], "next_message_num": 1,
        "messages": [],
    }
    tasks.append(mk(
        "email", CAT_NO_ACTION,
        "Archive the latest newsletter email.",
        empty_mail,
        # Empty mailbox: correct behavior is a final answer or clarification that
        # communicates "there is nothing to archive" — accept either phrasing family.
        [{"check": "answer_contains_any", "strings": MISSING_INFO_PHRASES}],
        terminal="any", forbid=True, tags=["empty_state"]))

    tiny_fs = {
        "current_date": DEFAULT_CURRENT_DATE,
        "files": [
            {"path": "/notes/todo.md", "content": "- [ ] Ship v2\n", "modified": "2026-02-27T10:00",
             "tags": []},
        ],
    }
    tasks.append(mk(
        "files", CAT_DEPENDENT,
        "Rename /notes/todo.md to /notes/tasks.md, keeping the content identical.",
        tiny_fs,
        [{"check": "record_exists", "collection": "files",
          "where": {"path": "/notes/tasks.md", "content": "- [ ] Ship v2\n"}},
         {"check": "record_absent", "collection": "files", "where": {"path": "/notes/todo.md"}}],
        max_steps=5, tags=["read", "create", "delete"]))

    tiny_todo = {
        "current_date": DEFAULT_CURRENT_DATE, "users": ["user", "alice", "bob"],
        "next_task_num": 3,
        "tasks": [
            {"id": "tsk_001", "title": "Patch edge servers", "description": "", "status": "open",
             "owner": "user", "assignee": "alice", "deadline": "2026-03-05", "priority": "high",
             "project": "infra", "created": "2026-02-25"},
            {"id": "tsk_002", "title": "Review infra alerts", "description": "",
             "status": "in_progress", "owner": "user", "assignee": "bob",
             "deadline": "2026-03-04", "priority": "medium", "project": "infra",
             "created": "2026-02-26"},
        ],
    }
    tasks.append(mk(
        "todo", CAT_MULTI_STEP,
        "Close every task in the 'infra' project (mark them done).",
        tiny_todo,
        [{"check": "field_equals", "collection": "tasks", "where": {"id": "tsk_001"},
          "field": "status", "value": "done"},
         {"check": "field_equals", "collection": "tasks", "where": {"id": "tsk_002"},
          "field": "status", "value": "done"}],
        max_steps=5, tags=["batch", "status"]))

    tiny_db = {
        "current_date": DEFAULT_CURRENT_DATE,
        "tables": {
            "customers": [
                {"customer_id": "cus_001", "name": "Test Corp", "region": "north",
                 "tier": "pro", "signup_date": "2025-12-01"},
            ],
            "products": [
                {"product_id": "prd_001", "name": "Bolt", "category": "hardware",
                 "price": 10.0, "stock": 100},
            ],
            "orders": [
                {"order_id": "ord_001", "customer_id": "cus_001", "product_id": "prd_001",
                 "quantity": 3, "unit_price": 10.0, "status": "completed",
                 "order_date": "2026-02-01"},
                {"order_id": "ord_002", "customer_id": "cus_001", "product_id": "prd_001",
                 "quantity": 4, "unit_price": 10.0, "status": "pending",
                 "order_date": "2026-02-15"},
            ],
        },
    }
    tasks.append(mk(
        "sql", CAT_RETRIEVE_ACT,
        "What is the total ordered quantity of product prd_001 across all orders, regardless "
        "of status? Answer with the exact number.",
        tiny_db,
        [{"check": "answer_contains_all", "strings": ["7"]}],
        forbid=True, tags=["aggregate"]))

    tiny_crm = {
        "current_date": DEFAULT_CURRENT_DATE,
        "companies": [{"id": "cmp_001", "name": "Solo Inc", "industry": "software",
                       "employee_count": 10, "website": "https://solo.example.com",
                       "city": "Oslo"}],
        "contacts": [{"id": "ctc_001", "name": "Only Person", "email": "only@solo.example.com",
                      "phone": "+47-555-000", "title": "Founder", "company_id": "cmp_001",
                      "tags": [], "owner": "user"}],
        "notes": [], "interactions": [],
        "next_num": {"contact": 2, "note": 1, "interaction": 1},
    }
    tasks.append(mk(
        "crm", CAT_SINGLE_STEP,
        "Log an email interaction with the founder of Solo Inc summarized as 'Sent pricing'.",
        tiny_crm,
        [{"check": "record_exists", "collection": "interactions",
          "where": {"contact_id": "ctc_001", "type": "email", "summary": "Sent pricing"}}],
        max_steps=4, tags=["lookup", "interaction"]))

    return tasks


# ==========================================================================
# Top-level build
# ==========================================================================

_FAMILY_BUILDERS = {
    "calendar": _build_calendar,
    "email": _build_email,
    "files": _build_files,
    "crm": _build_crm,
    "sql": _build_sql,
    "todo": _build_todo,
}


def build_benchmark(seed: int = 20260301, *, held_out_families: tuple[str, ...] = ("crm",),
                    train_fraction: float = 0.7, val_fraction: float = 0.1,
                    version: str = "v1") -> list[Task]:
    """Build the full benchmark deterministically and assign splits.

    The same ``seed`` always produces byte-identical tasks (fixtures included).
    """
    rng = random.Random(seed)
    alloc = _Alloc()
    tasks: list[Task] = []
    for family in ("calendar", "email", "files", "crm", "sql", "todo"):
        tasks.extend(_FAMILY_BUILDERS[family](rng, alloc, seed, version))
    tasks.extend(_build_handwritten(alloc, seed, version))

    unknown_cats = {t.category for t in tasks} - set(CATEGORIES)
    if unknown_cats:
        raise TaskError(f"Builders produced unknown categories: {unknown_cats}")

    salt = f"{version}:{seed}"
    assign_splits(tasks, held_out_families, train_fraction, val_fraction, salt=salt)
    balance_category_coverage(tasks, held_out_families, salt=salt)
    check_no_leakage(tasks, held_out_families)
    return tasks
