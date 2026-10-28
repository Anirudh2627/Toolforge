"""Email environment: a simulated mailbox. Nothing leaves the process.

State shape::

    {
      "current_date": "2026-03-02",
      "user_email": "user@example.com",
      "folders": ["inbox", "archive", "spam", "sent", "drafts"],
      "messages": [{"id","from","to","subject","body","date","folder","read","in_reply_to"}],
      "next_message_num": int
    }

``read`` is a volatile key: :meth:`read_message` marks messages as read, but
that mutation is not treated as a task-relevant side effect by the verifier.
``send_email``/``send_draft`` only append to the simulated ``sent`` folder.
"""

from __future__ import annotations

import random
import re
from typing import Any, ClassVar

from toolforge.environments.base import (
    ERR_INVALID_ARGUMENT,
    ERR_NOT_FOUND,
    Environment,
    ToolResult,
    require,
)
from toolforge.environments.pools import EMAILS, USER_EMAIL, pick
from toolforge.schemas import ToolSpec
from toolforge.timeutils import DEFAULT_CURRENT_DATE, add_days, is_valid_datetime

DEFAULT_FOLDERS = ["inbox", "archive", "spam", "sent", "drafts"]

_SUBJECTS = [
    ("Q1 budget numbers", "Hi,\n\nThe Q1 budget is 45000 USD. Please review before Friday.\n\nThanks"),
    ("Invoice INV-201 overdue", "Hello,\n\nInvoice INV-201 for 4200 USD is overdue by 12 days.\n\nRegards,\nBilling"),
    ("Design review notes", "Team,\n\nNotes from the design review are in /notes/design-review.md.\n\nCheers"),
    ("Lunch tomorrow?", "Hey! Want to grab lunch tomorrow at noon?"),
    ("Contract draft attached", "Please find the contract draft below.\n\nSection 4 needs legal review."),
    ("Server migration plan", "The migration window is Saturday 02:00-06:00 UTC.\n\nOps team"),
    ("Weekly metrics report", "Weekly active users: 12480, up 4 percent.\n\nAnalytics"),
    ("Your order has shipped", "Your order #ORD-7741 has shipped and arrives Tuesday."),
    ("Interview feedback needed", "Please submit interview feedback for the backend candidate by EOD."),
    ("Re: Project timeline", "The timeline slip is approved. New target is March 20."),
    ("URGENT: reset your password", "Click here to claim your prize!!! You have WON a free cruise!"),
    ("Team offsite voting", "Vote for the offsite location: mountains or beach? Poll closes Friday."),
]


def _looks_like_email(value: Any) -> bool:
    return isinstance(value, str) and re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value) is not None


class EmailEnvironment(Environment):
    family: ClassVar[str] = "email"
    volatile_keys: ClassVar[frozenset[str]] = frozenset({"read"})

    # ------------------------------------------------------------------ state

    def default_state(self, seed: int) -> dict[str, Any]:
        rng = random.Random(seed)
        messages: list[dict[str, Any]] = []
        # 8 inbox (mixed read), 2 archive, 1 spam, 1 sent.
        layout = [("inbox", False)] * 5 + [("inbox", True)] * 3 + \
                 [("archive", True)] * 2 + [("spam", False)] + [("sent", True)]
        for i, (folder, read) in enumerate(layout):
            subject, body = _SUBJECTS[i % len(_SUBJECTS)]
            sender = pick(rng, EMAILS) if folder != "sent" else USER_EMAIL
            recipient = USER_EMAIL if folder != "sent" else pick(rng, EMAILS)
            # Spread deterministically over the week before current_date.
            day_offset = -(i % 7)
            date = f"{add_days(DEFAULT_CURRENT_DATE, day_offset)}T{9 + (i * 3) % 9:02d}:{(i * 17) % 60:02d}"
            messages.append({
                "id": f"msg_{i + 1:03d}",
                "from": sender,
                "to": recipient,
                "subject": subject,
                "body": body,
                "date": date,
                "folder": folder,
                "read": read,
                "in_reply_to": None,
            })
        messages.sort(key=lambda m: (m["date"], m["id"]))
        return {
            "current_date": DEFAULT_CURRENT_DATE,
            "user_email": USER_EMAIL,
            "folders": list(DEFAULT_FOLDERS),
            "messages": messages,
            "next_message_num": len(messages) + 1,
        }

    def normalize_state(self, fixture: dict[str, Any]) -> dict[str, Any]:
        require(isinstance(fixture, dict), "email fixture must be an object")
        state: dict[str, Any] = {
            "current_date": str(fixture.get("current_date", DEFAULT_CURRENT_DATE)),
            "user_email": str(fixture.get("user_email", USER_EMAIL)),
            "folders": list(fixture.get("folders", DEFAULT_FOLDERS)),
            "messages": [],
            "next_message_num": 1,
        }
        require(all(isinstance(f, str) for f in state["folders"]), "folders must be strings")
        for required_folder in ("sent", "drafts"):
            require(required_folder in state["folders"], f"folders must include {required_folder!r}")
        raw = fixture.get("messages", [])
        require(isinstance(raw, list), "messages must be a list")
        seen_ids: set[str] = set()
        for i, msg in enumerate(raw):
            require(isinstance(msg, dict), f"messages[{i}] must be an object")
            for key in ("id", "from", "to", "subject", "body", "date", "folder"):
                require(isinstance(msg.get(key), str) and msg.get(key) != "",
                        f"messages[{i}] missing/invalid string field {key!r}")
            require(msg["id"] not in seen_ids, f"duplicate message id {msg['id']!r}")
            seen_ids.add(msg["id"])
            require(is_valid_datetime(msg["date"]),
                    f"messages[{i}].date must be YYYY-MM-DDTHH:MM, got {msg['date']!r}")
            require(msg["folder"] in state["folders"],
                    f"messages[{i}].folder {msg['folder']!r} not in folders")
            state["messages"].append({
                "id": msg["id"],
                "from": msg["from"],
                "to": msg["to"],
                "subject": msg["subject"],
                "body": msg["body"],
                "date": msg["date"],
                "folder": msg["folder"],
                "read": bool(msg.get("read", False)),
                "in_reply_to": msg.get("in_reply_to"),
            })
        next_num = fixture.get("next_message_num")
        if next_num is None:
            next_num = len(state["messages"]) + 1
        require(isinstance(next_num, int) and next_num >= 1, "next_message_num must be a positive int")
        state["next_message_num"] = next_num
        return state

    # ------------------------------------------------------------------ tools

    tool_specs: ClassVar[list[ToolSpec]] = [
        ToolSpec(
            "email.search_messages",
            "Search messages. Matches query as a case-insensitive substring of subject "
            "and body. Returns headers (no body), newest first.",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "Substring to look for in subject/body."},
                "folder": {"type": "string", "description": "Restrict to one folder."},
                "from": {"type": "string", "description": "Restrict to a sender (substring match)."},
                "unread_only": {"type": "boolean", "description": "Only unread messages."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            }, "additionalProperties": False},
        ),
        ToolSpec(
            "email.read_message",
            "Read one message by id; returns the full message and marks it as read.",
            {"type": "object", "properties": {"message_id": {"type": "string", "minLength": 1}},
             "required": ["message_id"], "additionalProperties": False},
        ),
        ToolSpec(
            "email.move_message",
            "Move a message to another folder (e.g. inbox -> archive).",
            {"type": "object", "properties": {
                "message_id": {"type": "string", "minLength": 1},
                "folder": {"type": "string", "minLength": 1,
                           "description": "Destination folder; must already exist."},
            }, "required": ["message_id", "folder"], "additionalProperties": False},
        ),
        ToolSpec(
            "email.draft_reply",
            "Create a draft reply to an existing message. The draft is stored in the "
            "drafts folder with subject 'Re: <original>' and can be sent later.",
            {"type": "object", "properties": {
                "message_id": {"type": "string", "minLength": 1},
                "body": {"type": "string", "minLength": 1, "maxLength": 4000},
            }, "required": ["message_id", "body"], "additionalProperties": False},
        ),
        ToolSpec(
            "email.send_email",
            "Send a new email (simulated: it is stored in the 'sent' folder; nothing "
            "leaves this environment).",
            {"type": "object", "properties": {
                "to": {"type": "string", "minLength": 3, "description": "Recipient address."},
                "subject": {"type": "string", "minLength": 1, "maxLength": 200},
                "body": {"type": "string", "minLength": 1, "maxLength": 8000},
            }, "required": ["to", "subject", "body"], "additionalProperties": False},
        ),
        ToolSpec(
            "email.send_draft",
            "Send an existing draft; it moves from 'drafts' to 'sent'.",
            {"type": "object", "properties": {"draft_id": {"type": "string", "minLength": 1}},
             "required": ["draft_id"], "additionalProperties": False},
        ),
    ]

    # ------------------------------------------------------------ tool bodies

    def _find_message(self, message_id: Any) -> dict[str, Any] | None:
        if not isinstance(message_id, str):
            return None
        for msg in self._state["messages"]:
            if msg["id"] == message_id:
                return msg
        return None

    def _new_id(self) -> str:
        msg_id = f"msg_{self._state['next_message_num']:03d}"
        self._state["next_message_num"] += 1
        return msg_id

    @staticmethod
    def _header(msg: dict[str, Any]) -> dict[str, Any]:
        return {k: msg[k] for k in ("id", "from", "to", "subject", "date", "folder", "read")}

    def _tool_search_messages(self, args: dict[str, Any]) -> ToolResult:
        limit = args.get("limit", 20)
        results = []
        for msg in self._state["messages"]:
            if args.get("folder") and msg["folder"] != args["folder"]:
                continue
            if args.get("from") and args["from"].lower() not in msg["from"].lower():
                continue
            if args.get("unread_only") and msg["read"]:
                continue
            if args.get("query"):
                q = args["query"].lower()
                if q not in msg["subject"].lower() and q not in msg["body"].lower():
                    continue
            results.append(msg)
        results.sort(key=lambda m: (m["date"], m["id"]), reverse=True)
        truncated = len(results) > limit
        return self._ok({
            "messages": [self._header(m) for m in results[:limit]],
            "count": min(len(results), limit),
            "truncated": truncated,
        })

    def _tool_read_message(self, args: dict[str, Any]) -> ToolResult:
        msg = self._find_message(args["message_id"])
        if msg is None:
            return self._fail(ERR_NOT_FOUND, f"No message with id {args['message_id']!r}")
        msg["read"] = True
        return self._ok(dict(msg))

    def _tool_move_message(self, args: dict[str, Any]) -> ToolResult:
        msg = self._find_message(args["message_id"])
        if msg is None:
            return self._fail(ERR_NOT_FOUND, f"No message with id {args['message_id']!r}")
        folder = args["folder"]
        if folder not in self._state["folders"]:
            return self._fail(ERR_NOT_FOUND,
                              f"Folder {folder!r} does not exist; known folders: {self._state['folders']}")
        msg["folder"] = folder
        return self._ok({"message_id": msg["id"], "folder": folder})

    def _tool_draft_reply(self, args: dict[str, Any]) -> ToolResult:
        original = self._find_message(args["message_id"])
        if original is None:
            return self._fail(ERR_NOT_FOUND, f"No message with id {args['message_id']!r}")
        subject = re.sub(r"^(Re:\s*)+", "", original["subject"], flags=re.IGNORECASE)
        draft = {
            "id": self._new_id(),
            "from": self._state["user_email"],
            "to": original["from"],
            "subject": f"Re: {subject}",
            "body": args["body"],
            "date": f"{self._state['current_date']}T12:00",
            "folder": "drafts",
            "read": True,
            "in_reply_to": original["id"],
        }
        self._state["messages"].append(draft)
        return self._ok({"draft": draft})

    def _tool_send_email(self, args: dict[str, Any]) -> ToolResult:
        if not _looks_like_email(args["to"]):
            return self._fail(ERR_INVALID_ARGUMENT,
                              f"Recipient {args['to']!r} is not a valid email address")
        message = {
            "id": self._new_id(),
            "from": self._state["user_email"],
            "to": args["to"],
            "subject": args["subject"],
            "body": args["body"],
            "date": f"{self._state['current_date']}T12:00",
            "folder": "sent",
            "read": True,
            "in_reply_to": None,
        }
        self._state["messages"].append(message)
        return self._ok({"message": message,
                         "note": "Simulated send: stored in 'sent'; no real email was delivered."})

    def _tool_send_draft(self, args: dict[str, Any]) -> ToolResult:
        draft = self._find_message(args["draft_id"])
        if draft is None:
            return self._fail(ERR_NOT_FOUND, f"No message with id {args['draft_id']!r}")
        if draft["folder"] != "drafts":
            return self._fail(ERR_INVALID_ARGUMENT,
                              f"Message {draft['id']} is in '{draft['folder']}', not 'drafts'")
        draft["folder"] = "sent"
        draft["date"] = f"{self._state['current_date']}T12:00"
        return self._ok({"message": draft,
                         "note": "Simulated send: stored in 'sent'; no real email was delivered."})
