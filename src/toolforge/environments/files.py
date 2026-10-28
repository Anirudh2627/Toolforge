"""File-system environment: a flat, in-memory tree of text files.

State shape::

    {
      "current_date": "2026-03-02",
      "files": [{"path","content","modified","tags"}]
    }

Files are stored as a list of records (not a dict) so the generic verifier
checks (record_exists / field_equals / ...) work uniformly across families.
Paths are absolute-looking strings ("/notes/todo.md") but nothing touches the
real filesystem. Reading has no side effects, so ``volatile_keys`` is empty.
"""

from __future__ import annotations

import fnmatch
import random
from typing import Any, ClassVar

from toolforge.environments.base import (
    ERR_ALREADY_EXISTS,
    ERR_INVALID_ARGUMENT,
    ERR_NOT_FOUND,
    Environment,
    ToolResult,
    require,
)
from toolforge.environments.pools import pick
from toolforge.schemas import ToolSpec
from toolforge.timeutils import DEFAULT_CURRENT_DATE, is_valid_datetime

_SEED_FILES: list[tuple[str, str, list[str]]] = [
    ("/notes/meeting-2026-02-27.md",
     "# Meeting notes 2026-02-27\n\nAttendees: Alice, Bob, user\n\n"
     "## Decisions\n- Freeze the API on March 10\n- Q1 budget: 45000 USD\n\n"
     "## Follow-ups\n- Bob: draft migration plan\n",
     ["meeting"]),
    ("/notes/todo.md",
     "# Todo\n\n- [ ] Review contract draft\n- [ ] Book offsite venue\n"
     "- [x] Submit expense report\n",
     ["personal"]),
    ("/notes/design-review.md",
     "# Design review\n\nThe new onboarding flow cuts steps from 5 to 3.\n"
     "Open question: do we keep the skip option?\n",
     ["meeting", "design"]),
    ("/reports/q1-budget.md",
     "# Q1 Budget Report\n\nTotal Q1 budget: 45000 USD.\nHeadcount: 12.\n"
     "Largest line item: infrastructure (18000 USD).\n",
     ["finance"]),
    ("/reports/weekly-metrics.md",
     "# Weekly metrics\n\nWAU 12480 (+4%).\nChurn 1.2%.\nDeploy failures: 0.\n",
     ["metrics"]),
    ("/specs/api.md",
     "# API spec\n\n## Section 1: Auth\nAll endpoints require a bearer token.\n\n"
     "## Section 2: Rate limits\n60 requests per minute per key.\n\n"
     "## Section 3: Errors\nErrors use RFC 7807 problem+json.\n",
     ["spec"]),
    ("/specs/deployment.md",
     "# Deployment runbook\n\nDeployment window: Saturday 02:00-06:00 UTC.\n"
     "Rollback: redeploy the previous tag.\n",
     ["spec", "ops"]),
    ("/reports/jan-summary.md",
     "# January summary\n\nShipped: search revamp, billing export.\n"
     "Incidents: 1 (database failover, 22 minutes).\n",
     ["finance"]),
    ("/notes/ideas.md",
     "# Ideas\n\n- Offline mode for the mobile app\n- Public status page\n"
     "- Self-serve onboarding\n",
     ["personal"]),
    ("/specs/security.md",
     "# Security notes\n\nSecrets live in the vault, never in git.\n"
     "Annual penetration test scheduled for June.\n",
     ["spec", "ops"]),
]


class FilesEnvironment(Environment):
    family: ClassVar[str] = "files"
    volatile_keys: ClassVar[frozenset[str]] = frozenset()

    # ------------------------------------------------------------------ state

    def default_state(self, seed: int) -> dict[str, Any]:
        rng = random.Random(seed)
        files = []
        for i, (path, content, tags) in enumerate(_SEED_FILES):
            # Deterministic but seed-varying modification timestamps.
            day = 14 + (i + rng.randrange(3)) % 14
            files.append({
                "path": path,
                "content": content,
                "modified": f"2026-02-{day:02d}T{9 + (i * 5) % 9:02d}:{(i * 23) % 60:02d}",
                "tags": list(tags),
            })
        # A seed-dependent extra note so different seeds are not identical.
        extra_topic = pick(rng, ["caching strategy", "search ranking", "mobile navigation"])
        files.append({
            "path": f"/notes/draft-{extra_topic.replace(' ', '-')}.md",
            "content": f"# Draft: {extra_topic}\n\nWork in progress.\nOwner: {pick(rng, ['alice', 'bob', 'carol'])}\n",
            "modified": "2026-02-28T15:04",
            "tags": ["draft"],
        })
        files.sort(key=lambda f: f["path"])
        return {"current_date": DEFAULT_CURRENT_DATE, "files": files}

    def normalize_state(self, fixture: dict[str, Any]) -> dict[str, Any]:
        require(isinstance(fixture, dict), "files fixture must be an object")
        state: dict[str, Any] = {
            "current_date": str(fixture.get("current_date", DEFAULT_CURRENT_DATE)),
            "files": [],
        }
        raw = fixture.get("files", [])
        require(isinstance(raw, list), "files must be a list")
        seen: set[str] = set()
        for i, f in enumerate(raw):
            require(isinstance(f, dict), f"files[{i}] must be an object")
            path, content = f.get("path"), f.get("content")
            require(isinstance(path, str) and path.startswith("/") and len(path) > 1,
                    f"files[{i}].path must be an absolute-looking string like '/notes/x.md'")
            require(path not in seen, f"duplicate file path {path!r}")
            seen.add(path)
            require(isinstance(content, str), f"files[{i}].content must be a string")
            modified = f.get("modified", f"{DEFAULT_CURRENT_DATE}T12:00")
            require(is_valid_datetime(modified), f"files[{i}].modified must be YYYY-MM-DDTHH:MM")
            tags = f.get("tags", [])
            require(isinstance(tags, list) and all(isinstance(t, str) for t in tags),
                    f"files[{i}].tags must be a list of strings")
            state["files"].append({"path": path, "content": content,
                                   "modified": modified, "tags": list(tags)})
        state["files"].sort(key=lambda f: f["path"])
        return state

    # ------------------------------------------------------------------ tools

    tool_specs: ClassVar[list[ToolSpec]] = [
        ToolSpec(
            "files.list_files",
            "List files with metadata, optionally restricted to a directory prefix "
            "(e.g. '/notes') and/or an fnmatch pattern on the path (e.g. '*.md').",
            {"type": "object", "properties": {
                "directory": {"type": "string", "description": "Path prefix filter."},
                "pattern": {"type": "string", "description": "fnmatch glob on the full path."},
            }, "additionalProperties": False},
        ),
        ToolSpec(
            "files.read_file",
            "Read the full content of one file.",
            {"type": "object", "properties": {"path": {"type": "string", "minLength": 2}},
             "required": ["path"], "additionalProperties": False},
        ),
        ToolSpec(
            "files.search_files",
            "Search file contents (and/or names) for a case-insensitive substring. "
            "Returns matching files with the matching lines and line numbers.",
            {"type": "object", "properties": {
                "query": {"type": "string", "minLength": 1},
                "directory": {"type": "string", "description": "Restrict to a path prefix."},
                "match_names": {"type": "boolean", "description": "Also match the query against file paths."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            }, "required": ["query"], "additionalProperties": False},
        ),
        ToolSpec(
            "files.create_file",
            "Create a new file. Fails if the path already exists (use update_file instead).",
            {"type": "object", "properties": {
                "path": {"type": "string", "minLength": 2, "pattern": "^/"},
                "content": {"type": "string", "maxLength": 20000},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
            }, "required": ["path", "content"], "additionalProperties": False},
        ),
        ToolSpec(
            "files.update_file",
            "Update an existing file. Provide exactly one of: content (overwrite) "
            "or append (text appended verbatim to the end).",
            {"type": "object", "properties": {
                "path": {"type": "string", "minLength": 2},
                "content": {"type": "string", "maxLength": 20000},
                "append": {"type": "string", "maxLength": 20000},
            }, "required": ["path"], "additionalProperties": False},
        ),
        ToolSpec(
            "files.delete_file",
            "Delete a file permanently.",
            {"type": "object", "properties": {"path": {"type": "string", "minLength": 2}},
             "required": ["path"], "additionalProperties": False},
        ),
    ]

    # ------------------------------------------------------------ tool bodies

    def _find(self, path: Any) -> dict[str, Any] | None:
        if not isinstance(path, str):
            return None
        for f in self._state["files"]:
            if f["path"] == path:
                return f
        return None

    def _stamp(self, record: dict[str, Any]) -> None:
        record["modified"] = f"{self._state['current_date']}T12:00"

    def _tool_list_files(self, args: dict[str, Any]) -> ToolResult:
        files = self._state["files"]
        if args.get("directory"):
            prefix = args["directory"].rstrip("/") + "/"
            files = [f for f in files if f["path"].startswith(prefix)]
        if args.get("pattern"):
            files = [f for f in files if fnmatch.fnmatch(f["path"], args["pattern"])]
        return self._ok({
            "files": [{"path": f["path"], "modified": f["modified"],
                       "size": len(f["content"]), "tags": f["tags"]} for f in files],
            "count": len(files),
        })

    def _tool_read_file(self, args: dict[str, Any]) -> ToolResult:
        f = self._find(args["path"])
        if f is None:
            return self._fail(ERR_NOT_FOUND, f"No file at {args['path']!r}")
        return self._ok({"path": f["path"], "content": f["content"], "modified": f["modified"]})

    def _tool_search_files(self, args: dict[str, Any]) -> ToolResult:
        query = args["query"].lower()
        limit = args.get("limit", 20)
        matches = []
        candidates = self._state["files"]
        if args.get("directory"):
            prefix = args["directory"].rstrip("/") + "/"
            candidates = [f for f in candidates if f["path"].startswith(prefix)]
        for f in candidates:
            hits = []
            if args.get("match_names") and query in f["path"].lower():
                hits.append({"line_number": 0, "line": f"(path match) {f['path']}"})
            for n, line in enumerate(f["content"].splitlines(), start=1):
                if query in line.lower():
                    hits.append({"line_number": n, "line": line.strip()[:200]})
            if hits:
                matches.append({"path": f["path"], "matches": hits[:10]})
        return self._ok({"results": matches[:limit], "count": min(len(matches), limit),
                         "truncated": len(matches) > limit})

    def _tool_create_file(self, args: dict[str, Any]) -> ToolResult:
        if self._find(args["path"]) is not None:
            return self._fail(ERR_ALREADY_EXISTS,
                              f"{args['path']!r} already exists; use files.update_file to modify it")
        record = {"path": args["path"], "content": args["content"],
                  "modified": "", "tags": list(args.get("tags", []))}
        self._stamp(record)
        self._state["files"].append(record)
        self._state["files"].sort(key=lambda f: f["path"])
        return self._ok({"path": record["path"], "size": len(record["content"]),
                         "modified": record["modified"]})

    def _tool_update_file(self, args: dict[str, Any]) -> ToolResult:
        f = self._find(args["path"])
        if f is None:
            return self._fail(ERR_NOT_FOUND, f"No file at {args['path']!r}")
        has_content = args.get("content") is not None
        has_append = args.get("append") is not None
        if has_content == has_append:
            return self._fail(ERR_INVALID_ARGUMENT,
                              "provide exactly one of 'content' (overwrite) or 'append'")
        if has_content:
            f["content"] = args["content"]
        else:
            f["content"] = f["content"] + args["append"]
        self._stamp(f)
        return self._ok({"path": f["path"], "size": len(f["content"]), "modified": f["modified"]})

    def _tool_delete_file(self, args: dict[str, Any]) -> ToolResult:
        f = self._find(args["path"])
        if f is None:
            return self._fail(ERR_NOT_FOUND, f"No file at {args['path']!r}")
        self._state["files"].remove(f)
        return self._ok({"deleted": args["path"]})
