"""CRM environment: companies, contacts, notes, interaction log.

State shape::

    {
      "current_date": "2026-03-02",
      "companies": [{"id","name","industry","employee_count","website","city"}],
      "contacts":  [{"id","name","email","phone","title","company_id","tags","owner"}],
      "notes":     [{"id","contact_id","text","date"}],
      "interactions": [{"id","contact_id","type","date","summary"}],
      "next_num": {"contact": int, "note": int, "interaction": int}
    }

This family is the default held-out family for generalization experiments
(see configs/base.yaml): none of its tasks may appear in train/val.
"""

from __future__ import annotations

import random
import re
from typing import Any, ClassVar

from toolforge.environments.base import (
    ERR_ALREADY_EXISTS,
    ERR_AMBIGUOUS,
    ERR_INVALID_ARGUMENT,
    ERR_NOT_FOUND,
    Environment,
    ToolResult,
    require,
)
from toolforge.environments.pools import PEOPLE, pick, sample
from toolforge.schemas import ToolSpec
from toolforge.timeutils import DEFAULT_CURRENT_DATE, is_valid_date, parse_date

INTERACTION_TYPES = ["call", "email", "meeting", "demo"]

_COMPANIES = [
    ("Northwind Traders", "logistics", 240, "https://northwind.example.com", "Seattle"),
    ("Lumen Analytics", "data", 85, "https://lumen.example.com", "Berlin"),
    ("Kite Robotics", "hardware", 310, "https://kite.example.com", "Austin"),
    ("Solara Energy", "energy", 1200, "https://solara.example.com", "Lisbon"),
    ("BrightPath Learning", "education", 60, "https://brightpath.example.com", "Toronto"),
    ("Helio Health", "healthcare", 540, "https://helio.example.com", "Boston"),
]

_TITLES = ["CTO", "VP Engineering", "Head of Data", "Procurement Manager",
           "Founder", "IT Director", "Product Lead"]

_TAGS = ["vip", "renewal", "conference-lead", "decision-maker", "trial"]


def _valid_email(value: Any) -> bool:
    return isinstance(value, str) and re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value) is not None


class CrmEnvironment(Environment):
    family: ClassVar[str] = "crm"
    volatile_keys: ClassVar[frozenset[str]] = frozenset()

    # ------------------------------------------------------------------ state

    def default_state(self, seed: int) -> dict[str, Any]:
        rng = random.Random(seed)
        companies = []
        for i, (name, industry, size, website, city) in enumerate(_COMPANIES):
            companies.append({
                "id": f"cmp_{i + 1:03d}", "name": name, "industry": industry,
                "employee_count": size, "website": website, "city": city,
            })
        contacts = []
        for i in range(12):
            person_name, person_email = PEOPLE[i % len(PEOPLE)]
            # Vary names slightly so multiple contacts can share a first name
            # (needed for ambiguity tasks): suffix every second rotation.
            if i >= len(PEOPLE):
                person_name = f"{person_name.split()[0]} Santos"
                person_email = person_email.replace("@", f"{i}@")
            company = companies[(i * 2 + rng.randrange(2)) % len(companies)]
            domain = company["website"].split("//", 1)[1]
            contacts.append({
                "id": f"ctc_{i + 1:03d}",
                "name": person_name,
                "email": f"{person_email.split('@')[0]}@{domain}",
                "phone": f"+1-555-{1000 + i * 37:04d}",
                "title": _TITLES[i % len(_TITLES)],
                "company_id": company["id"],
                "tags": sample(rng, _TAGS, rng.randrange(0, 3)),
                "owner": pick(rng, ["user", "alice", "bob"]),
            })
        notes = [
            {"id": f"nte_{i + 1:03d}", "contact_id": contacts[i]["id"],
             "text": pick(rng, ["Prefers email over calls.", "Met at DataConf 2025.",
                                "Renewal decision expected in Q2.", "Asked for a security whitepaper."]),
             "date": "2026-02-25"}
            for i in range(5)
        ]
        interactions = [
            {"id": f"act_{i + 1:03d}", "contact_id": contacts[i + 3]["id"],
             "type": pick(rng, INTERACTION_TYPES), "date": f"2026-02-{20 + i:02d}",
             "summary": pick(rng, ["Discussed pricing tiers.", "Demoed the reporting module.",
                                   "Follow-up on open support ticket.", "Intro call with their CTO."])}
            for i in range(4)
        ]
        return {
            "current_date": DEFAULT_CURRENT_DATE,
            "companies": companies,
            "contacts": contacts,
            "notes": notes,
            "interactions": interactions,
            "next_num": {"contact": len(contacts) + 1, "note": len(notes) + 1,
                         "interaction": len(interactions) + 1},
        }

    def normalize_state(self, fixture: dict[str, Any]) -> dict[str, Any]:
        require(isinstance(fixture, dict), "crm fixture must be an object")
        state: dict[str, Any] = {"current_date": str(fixture.get("current_date", DEFAULT_CURRENT_DATE))}

        companies = fixture.get("companies", [])
        require(isinstance(companies, list), "companies must be a list")
        state["companies"] = []
        for i, c in enumerate(companies):
            require(isinstance(c, dict), f"companies[{i}] must be an object")
            for key in ("id", "name"):
                require(isinstance(c.get(key), str) and c[key], f"companies[{i}] missing {key!r}")
            state["companies"].append({
                "id": c["id"], "name": c["name"],
                "industry": str(c.get("industry", "")),
                "employee_count": int(c.get("employee_count", 0)),
                "website": str(c.get("website", "")),
                "city": str(c.get("city", "")),
            })
        company_ids = {c["id"] for c in state["companies"]}

        contacts = fixture.get("contacts", [])
        require(isinstance(contacts, list), "contacts must be a list")
        state["contacts"] = []
        seen_emails: set[str] = set()
        for i, c in enumerate(contacts):
            require(isinstance(c, dict), f"contacts[{i}] must be an object")
            for key in ("id", "name", "email"):
                require(isinstance(c.get(key), str) and c[key], f"contacts[{i}] missing {key!r}")
            require(_valid_email(c["email"]), f"contacts[{i}].email invalid: {c['email']!r}")
            require(c["email"] not in seen_emails, f"duplicate contact email {c['email']!r}")
            seen_emails.add(c["email"])
            if c.get("company_id") is not None:
                require(c["company_id"] in company_ids,
                        f"contacts[{i}].company_id {c['company_id']!r} not in companies")
            tags = c.get("tags", [])
            require(isinstance(tags, list) and all(isinstance(t, str) for t in tags),
                    f"contacts[{i}].tags must be a list of strings")
            state["contacts"].append({
                "id": c["id"], "name": c["name"], "email": c["email"],
                "phone": str(c.get("phone", "")), "title": str(c.get("title", "")),
                "company_id": c.get("company_id"), "tags": list(tags),
                "owner": str(c.get("owner", "user")),
            })
        contact_ids = {c["id"] for c in state["contacts"]}

        for coll, id_field, extra_check in (
            ("notes", "text", None),
            ("interactions", "summary", "type"),
        ):
            raw = fixture.get(coll, [])
            require(isinstance(raw, list), f"{coll} must be a list")
            state[coll] = []
            for i, rec in enumerate(raw):
                require(isinstance(rec, dict), f"{coll}[{i}] must be an object")
                for key in ("id", "contact_id", id_field, "date"):
                    require(isinstance(rec.get(key), str) and rec.get(key),
                            f"{coll}[{i}] missing {key!r}")
                require(rec["contact_id"] in contact_ids,
                        f"{coll}[{i}].contact_id {rec['contact_id']!r} not in contacts")
                require(is_valid_date(rec["date"]), f"{coll}[{i}].date must be YYYY-MM-DD")
                if extra_check == "type":
                    require(rec.get("type") in INTERACTION_TYPES,
                            f"{coll}[{i}].type must be one of {INTERACTION_TYPES}")
                entry = {"id": rec["id"], "contact_id": rec["contact_id"],
                         "date": rec["date"], id_field: rec[id_field]}
                if extra_check == "type":
                    entry["type"] = rec["type"]
                state[coll].append(entry)

        next_num = fixture.get("next_num", {})
        require(isinstance(next_num, dict), "next_num must be an object")
        state["next_num"] = {
            "contact": int(next_num.get("contact", len(state["contacts"]) + 1)),
            "note": int(next_num.get("note", len(state["notes"]) + 1)),
            "interaction": int(next_num.get("interaction", len(state["interactions"]) + 1)),
        }
        return state

    # ------------------------------------------------------------------ tools

    tool_specs: ClassVar[list[ToolSpec]] = [
        ToolSpec(
            "crm.search_contacts",
            "Search contacts by free-text query (matches name, email, title), "
            "company id, tag, or owner. Filters combine with AND.",
            {"type": "object", "properties": {
                "query": {"type": "string"},
                "company_id": {"type": "string"},
                "tag": {"type": "string"},
                "owner": {"type": "string"},
            }, "additionalProperties": False},
        ),
        ToolSpec(
            "crm.get_company",
            "Look up one company by id, or by (substring of) name. Fails with "
            "'ambiguous' and a candidate list when a name matches several companies.",
            {"type": "object", "properties": {
                "company_id": {"type": "string"},
                "name": {"type": "string"},
            }, "additionalProperties": False},
        ),
        ToolSpec(
            "crm.create_contact",
            "Create a contact. Fails if the email already exists.",
            {"type": "object", "properties": {
                "name": {"type": "string", "minLength": 1},
                "email": {"type": "string", "minLength": 3},
                "company_id": {"type": "string"},
                "title": {"type": "string", "maxLength": 80},
                "phone": {"type": "string", "maxLength": 30},
                "owner": {"type": "string", "maxLength": 40},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
            }, "required": ["name", "email"], "additionalProperties": False},
        ),
        ToolSpec(
            "crm.update_contact",
            "Update fields of a contact (at least one field required). 'tags' "
            "replaces the whole tag list.",
            {"type": "object", "properties": {
                "contact_id": {"type": "string", "minLength": 1},
                "email": {"type": "string", "minLength": 3},
                "phone": {"type": "string", "maxLength": 30},
                "title": {"type": "string", "maxLength": 80},
                "owner": {"type": "string", "maxLength": 40},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
            }, "required": ["contact_id"], "additionalProperties": False},
        ),
        ToolSpec(
            "crm.add_note",
            "Attach a note to a contact.",
            {"type": "object", "properties": {
                "contact_id": {"type": "string", "minLength": 1},
                "text": {"type": "string", "minLength": 1, "maxLength": 2000},
            }, "required": ["contact_id", "text"], "additionalProperties": False},
        ),
        ToolSpec(
            "crm.log_interaction",
            "Record an interaction (call/email/meeting/demo) with a contact.",
            {"type": "object", "properties": {
                "contact_id": {"type": "string", "minLength": 1},
                "type": {"type": "string", "enum": INTERACTION_TYPES},
                "summary": {"type": "string", "minLength": 1, "maxLength": 500},
                "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$",
                         "description": "Defaults to the current date."},
            }, "required": ["contact_id", "type", "summary"], "additionalProperties": False},
        ),
    ]

    # ------------------------------------------------------------ tool bodies

    def _alloc(self, kind: str, prefix: str) -> str:
        n = self._state["next_num"][kind]
        self._state["next_num"][kind] = n + 1
        return f"{prefix}_{n:03d}"

    def _find_contact(self, contact_id: Any) -> dict[str, Any] | None:
        if not isinstance(contact_id, str):
            return None
        for c in self._state["contacts"]:
            if c["id"] == contact_id:
                return c
        return None

    def _tool_search_contacts(self, args: dict[str, Any]) -> ToolResult:
        results = self._state["contacts"]
        if args.get("query"):
            q = args["query"].lower()
            results = [c for c in results
                       if q in c["name"].lower() or q in c["email"].lower() or q in c["title"].lower()]
        if args.get("company_id"):
            results = [c for c in results if c["company_id"] == args["company_id"]]
        if args.get("tag"):
            results = [c for c in results if args["tag"] in c["tags"]]
        if args.get("owner"):
            results = [c for c in results if c["owner"] == args["owner"]]
        return self._ok({"contacts": results, "count": len(results)})

    def _tool_get_company(self, args: dict[str, Any]) -> ToolResult:
        if not args.get("company_id") and not args.get("name"):
            return self._fail(ERR_INVALID_ARGUMENT, "provide company_id or name")
        if args.get("company_id"):
            for c in self._state["companies"]:
                if c["id"] == args["company_id"]:
                    return self._ok({"company": c})
            return self._fail(ERR_NOT_FOUND, f"No company with id {args['company_id']!r}")
        name = args["name"].lower()
        matches = [c for c in self._state["companies"] if name in c["name"].lower()]
        if not matches:
            return self._fail(ERR_NOT_FOUND, f"No company matching name {args['name']!r}")
        if len(matches) > 1:
            return self._fail(ERR_AMBIGUOUS,
                              f"Name {args['name']!r} matches {len(matches)} companies: "
                              + ", ".join(f"{c['id']} ({c['name']})" for c in matches))
        return self._ok({"company": matches[0]})

    def _tool_create_contact(self, args: dict[str, Any]) -> ToolResult:
        if not _valid_email(args["email"]):
            return self._fail(ERR_INVALID_ARGUMENT, f"Invalid email {args['email']!r}")
        if any(c["email"].lower() == args["email"].lower() for c in self._state["contacts"]):
            return self._fail(ERR_ALREADY_EXISTS, f"A contact with email {args['email']!r} already exists")
        if args.get("company_id") is not None:
            if not any(c["id"] == args["company_id"] for c in self._state["companies"]):
                return self._fail(ERR_NOT_FOUND, f"No company with id {args['company_id']!r}")
        contact = {
            "id": self._alloc("contact", "ctc"),
            "name": args["name"], "email": args["email"],
            "phone": args.get("phone", ""), "title": args.get("title", ""),
            "company_id": args.get("company_id"), "tags": list(args.get("tags", [])),
            "owner": args.get("owner", "user"),
        }
        self._state["contacts"].append(contact)
        return self._ok({"contact": contact})

    def _tool_update_contact(self, args: dict[str, Any]) -> ToolResult:
        updatable = ("email", "phone", "title", "owner", "tags")
        changes = {k: v for k, v in args.items() if k in updatable and v is not None}
        if not changes:
            return self._fail(ERR_INVALID_ARGUMENT,
                              f"update_contact requires at least one field among {list(updatable)}")
        contact = self._find_contact(args["contact_id"])
        if contact is None:
            return self._fail(ERR_NOT_FOUND, f"No contact with id {args['contact_id']!r}")
        if "email" in changes:
            if not _valid_email(changes["email"]):
                return self._fail(ERR_INVALID_ARGUMENT, f"Invalid email {changes['email']!r}")
            if any(c["id"] != contact["id"] and c["email"].lower() == changes["email"].lower()
                   for c in self._state["contacts"]):
                return self._fail(ERR_ALREADY_EXISTS, f"Email {changes['email']!r} is used by another contact")
        contact.update(changes)
        if "tags" in changes:
            contact["tags"] = list(changes["tags"])
        return self._ok({"contact": contact})

    def _tool_add_note(self, args: dict[str, Any]) -> ToolResult:
        if self._find_contact(args["contact_id"]) is None:
            return self._fail(ERR_NOT_FOUND, f"No contact with id {args['contact_id']!r}")
        note = {"id": self._alloc("note", "nte"), "contact_id": args["contact_id"],
                "text": args["text"], "date": self._state["current_date"]}
        self._state["notes"].append(note)
        return self._ok({"note": note})

    def _tool_log_interaction(self, args: dict[str, Any]) -> ToolResult:
        if self._find_contact(args["contact_id"]) is None:
            return self._fail(ERR_NOT_FOUND, f"No contact with id {args['contact_id']!r}")
        when = args.get("date") or self._state["current_date"]
        try:
            parse_date(when)
        except Exception as exc:
            return self._fail(ERR_INVALID_ARGUMENT, str(exc))
        interaction = {"id": self._alloc("interaction", "act"), "contact_id": args["contact_id"],
                       "type": args["type"], "date": when, "summary": args["summary"]}
        self._state["interactions"].append(interaction)
        return self._ok({"interaction": interaction})
