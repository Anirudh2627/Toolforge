"""Action grammar, tool specifications, and JSON-Schema-subset validation.

This module is the single source of truth for the serialization contract that
training, inference, and evaluation must all share:

Grammar — the model emits exactly one JSON object per turn::

    {"action": "tool_call", "tool": "<family>.<tool>", "arguments": {...}}
    {"action": "clarify", "question": "<text>"}
    {"action": "final_answer", "content": "<text>"}

Extra top-level keys are rejected so drift is visible in metrics rather than
silently tolerated. Parsing is tolerant of prose around the JSON and of
```json fences (small models emit these often), but strict about structure.

Argument schemas are validated with a deliberately small JSON-Schema subset
(type, properties, required, additionalProperties, enum, const, items,
minItems, maxItems, minimum, maximum, minLength, maxLength, pattern). This
avoids a jsonschema dependency, keeps behavior fully deterministic, and
produces structured :class:`ValidationIssue` records that feed the
"argument validation failure rate" metric. :class:`ToolSpec` rejects schemas
using keywords outside the subset at construction time, so an unsupported
schema can never enter the benchmark unnoticed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ToolForgeError(Exception):
    """Base class for all project errors."""


#: Bumped whenever the action grammar, system-prompt template, or tool-catalog
#: rendering changes. Evaluation runs record it so checkpoints trained under a
#: different serialization are never silently compared (see evaluation.py).
GRAMMAR_VERSION = "tf-gram-v1"


class InvalidToolSpecError(ToolForgeError):
    """Raised when a ToolSpec uses unsupported schema features."""


# --------------------------------------------------------------------------
# JSON-Schema subset validation
# --------------------------------------------------------------------------

SUPPORTED_SCHEMA_KEYWORDS = frozenset({
    "type", "description", "properties", "required", "additionalProperties",
    "enum", "const", "items", "minItems", "maxItems",
    "minimum", "maximum", "minLength", "maxLength", "pattern",
    "default",  # documentation only; never applied during validation
})

_TYPE_NAMES = ("string", "integer", "number", "boolean", "array", "object", "null")


@dataclass(frozen=True)
class ValidationIssue:
    """A single argument-schema violation, structured for metrics."""

    path: str          # JSON-pointer-like path, e.g. "arguments.attendees[2]"
    kind: str          # missing_required | unexpected_property | type_error |
                       # enum_violation | const_violation | range_violation |
                       # length_violation | pattern_violation | items_violation
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "kind": self.kind, "message": self.message}


def _type_matches(value: Any, type_name: str) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "null":
        return value is None
    raise InvalidToolSpecError(f"Unknown type name in schema: {type_name!r}")


def check_schema_supported(schema: Any, path: str = "$") -> None:
    """Raise :class:`InvalidToolSpecError` if schema uses unsupported features."""
    if not isinstance(schema, dict):
        raise InvalidToolSpecError(f"{path}: schema must be an object")
    unknown = set(schema) - SUPPORTED_SCHEMA_KEYWORDS
    if unknown:
        raise InvalidToolSpecError(
            f"{path}: unsupported schema keyword(s) {sorted(unknown)}; "
            f"the validator implements only {sorted(SUPPORTED_SCHEMA_KEYWORDS)}"
        )
    if "type" in schema:
        types = schema["type"]
        for t in types if isinstance(types, list) else [types]:
            if t not in _TYPE_NAMES:
                raise InvalidToolSpecError(f"{path}.type: unknown type {t!r}")
    if "properties" in schema:
        if not isinstance(schema["properties"], dict):
            raise InvalidToolSpecError(f"{path}.properties must be an object")
        for key, sub in schema["properties"].items():
            check_schema_supported(sub, f"{path}.properties.{key}")
    if "items" in schema:
        if not isinstance(schema["items"], dict):
            raise InvalidToolSpecError(f"{path}.items: only a single schema object is supported")
        check_schema_supported(schema["items"], f"{path}.items")
    if "required" in schema and not isinstance(schema["required"], list):
        raise InvalidToolSpecError(f"{path}.required must be a list")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        raise InvalidToolSpecError(f"{path}.additionalProperties: only true/false is supported")
    if "pattern" in schema:
        re.compile(schema["pattern"])  # raises re.error on bad patterns


def validate_against_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[ValidationIssue]:
    """Validate ``value`` against the supported JSON-Schema subset.

    Returns a (possibly empty) list of issues; never raises for bad input
    values — only for invalid schemas themselves.
    """
    issues: list[ValidationIssue] = []

    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_matches(value, t) for t in types):
            actual = "null" if value is None else type(value).__name__
            issues.append(ValidationIssue(path, "type_error",
                                          f"expected type {'|'.join(types)}, got {actual}"))
            return issues  # further checks on a wrong type are noise

    if "const" in schema and value != schema["const"]:
        issues.append(ValidationIssue(path, "const_violation",
                                      f"value must be {schema['const']!r}, got {value!r}"))
    if "enum" in schema and value not in schema["enum"]:
        issues.append(ValidationIssue(path, "enum_violation",
                                      f"value {value!r} not in allowed set {schema['enum']}"))

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            issues.append(ValidationIssue(path, "length_violation",
                                          f"string shorter than minLength={schema['minLength']}"))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            issues.append(ValidationIssue(path, "length_violation",
                                          f"string longer than maxLength={schema['maxLength']}"))
        if "pattern" in schema and not re.search(schema["pattern"], value):
            issues.append(ValidationIssue(path, "pattern_violation",
                                          f"string {value!r} does not match pattern {schema['pattern']!r}"))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            issues.append(ValidationIssue(path, "range_violation",
                                          f"value {value} < minimum {schema['minimum']}"))
        if "maximum" in schema and value > schema["maximum"]:
            issues.append(ValidationIssue(path, "range_violation",
                                          f"value {value} > maximum {schema['maximum']}"))

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            issues.append(ValidationIssue(path, "length_violation",
                                          f"array has {len(value)} items, minItems={schema['minItems']}"))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            issues.append(ValidationIssue(path, "length_violation",
                                          f"array has {len(value)} items, maxItems={schema['maxItems']}"))
        if "items" in schema:
            for i, item in enumerate(value):
                issues.extend(validate_against_schema(item, schema["items"], f"{path}[{i}]"))

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                issues.append(ValidationIssue(f"{path}.{key}", "missing_required",
                                              f"required property {key!r} is missing"))
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    issues.append(ValidationIssue(f"{path}.{key}", "unexpected_property",
                                                  f"property {key!r} is not allowed"))
        for key, sub_schema in properties.items():
            if key in value:
                issues.extend(validate_against_schema(value[key], sub_schema, f"{path}.{key}"))

    return issues


# --------------------------------------------------------------------------
# Tool specifications
# --------------------------------------------------------------------------

_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class ToolSpec:
    """A tool exposed to the model: qualified name, description, arg schema.

    ``name`` is ``"<family>.<tool>"`` (e.g. ``"calendar.create_event"``). The
    description must be neutral: no hints about tasks, rewards, or expected
    outcomes (verified by benchmark tests).
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _TOOL_NAME_RE.match(self.name):
            raise InvalidToolSpecError(
                f"Tool name {self.name!r} must look like 'family.tool' "
                "(lowercase letters, digits, underscore)"
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise InvalidToolSpecError(f"Tool {self.name}: description must be a non-empty string")
        if not isinstance(self.parameters, dict):
            raise InvalidToolSpecError(f"Tool {self.name}: parameters must be a JSON schema object")
        check_schema_supported(self.parameters, f"tool {self.name} parameters")

    @property
    def family(self) -> str:
        return self.name.split(".", 1)[0]

    @property
    def local_name(self) -> str:
        return self.name.split(".", 1)[1]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolSpec":
        if not isinstance(data, dict):
            raise InvalidToolSpecError(f"ToolSpec.from_dict expects a mapping, got {type(data).__name__}")
        missing = {"name", "description"} - set(data)
        if missing:
            raise InvalidToolSpecError(f"ToolSpec.from_dict missing keys: {sorted(missing)}")
        return cls(name=data["name"], description=data["description"],
                   parameters=data.get("parameters", {}))


def render_tool_catalog(specs: list[ToolSpec]) -> str:
    """Render tool specs as the text catalog shown to the model.

    Identical text is used by the executor's system prompt (Phase 1) and the
    training serialization (Phase 3), so the model sees one consistent format.
    """
    return json.dumps([s.to_dict() for s in specs], indent=1, ensure_ascii=False)


# --------------------------------------------------------------------------
# Action grammar
# --------------------------------------------------------------------------

class ActionKind(str, Enum):
    TOOL_CALL = "tool_call"
    CLARIFY = "clarify"
    FINAL_ANSWER = "final_answer"


@dataclass(frozen=True)
class Action:
    """A successfully parsed model action."""

    kind: ActionKind
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    question: str | None = None
    content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"action": self.kind.value}
        if self.kind is ActionKind.TOOL_CALL:
            data["tool"] = self.tool
            data["arguments"] = self.arguments or {}
        elif self.kind is ActionKind.CLARIFY:
            data["question"] = self.question
        else:
            data["content"] = self.content
        return data


@dataclass(frozen=True)
class ParseFailure:
    """A structured parse failure; ``category`` feeds evaluation metrics."""

    category: str  # invalid_json | not_an_object | missing_field | field_type_error
                   # | unknown_action | unexpected_field | tool_name_format
    message: str
    raw_text: str

    def to_dict(self) -> dict[str, str]:
        return {"category": self.category, "message": self.message,
                "raw_text": self.raw_text[:500]}


def extract_json_object(text: str) -> str | None:
    """Extract the first balanced top-level ``{...}`` block from ``text``.

    Handles ```json fences and surrounding prose. Brace counting respects
    string literals and escapes, so braces inside JSON strings are safe.
    Returns ``None`` when no ``{`` is present.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None  # unbalanced


def parse_action(text: str) -> Action | ParseFailure:
    """Parse one model turn into an :class:`Action` or :class:`ParseFailure`."""
    if not isinstance(text, str):
        return ParseFailure("field_type_error", f"model output must be str, got {type(text).__name__}", str(text))
    stripped = text.strip()
    if not stripped:
        return ParseFailure("invalid_json", "empty output", text)

    candidate = extract_json_object(stripped)
    if candidate is None:
        return ParseFailure("invalid_json", "no balanced top-level JSON object found", text)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ParseFailure("invalid_json", f"JSON decode error: {exc}", text)
    if not isinstance(data, dict):
        return ParseFailure("not_an_object", f"expected a JSON object, got {type(data).__name__}", text)

    allowed_by_kind = {
        ActionKind.TOOL_CALL: {"action", "tool", "arguments"},
        ActionKind.CLARIFY: {"action", "question"},
        ActionKind.FINAL_ANSWER: {"action", "content"},
    }

    raw_kind = data.get("action")
    if "action" not in data:
        return ParseFailure("missing_field", "missing required field 'action'", text)
    if not isinstance(raw_kind, str):
        return ParseFailure("field_type_error", f"'action' must be a string, got {type(raw_kind).__name__}", text)
    try:
        kind = ActionKind(raw_kind)
    except ValueError:
        return ParseFailure(
            "unknown_action",
            f"unknown action {raw_kind!r}; allowed: {[k.value for k in ActionKind]}",
            text,
        )

    extra = set(data) - allowed_by_kind[kind]
    if extra:
        return ParseFailure("unexpected_field",
                            f"unexpected field(s) {sorted(extra)} for action {kind.value!r}", text)

    if kind is ActionKind.TOOL_CALL:
        tool = data.get("tool")
        if tool is None:
            return ParseFailure("missing_field", "tool_call requires field 'tool'", text)
        if not isinstance(tool, str):
            return ParseFailure("field_type_error", f"'tool' must be a string, got {type(tool).__name__}", text)
        if not _TOOL_NAME_RE.match(tool):
            return ParseFailure("tool_name_format",
                                f"tool name {tool!r} must look like 'family.tool'", text)
        arguments = data.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return ParseFailure("field_type_error",
                                f"'arguments' must be an object, got {type(arguments).__name__}", text)
        return Action(kind=kind, tool=tool, arguments=arguments)

    if kind is ActionKind.CLARIFY:
        question = data.get("question")
        if question is None:
            return ParseFailure("missing_field", "clarify requires field 'question'", text)
        if not isinstance(question, str) or not question.strip():
            return ParseFailure("field_type_error", "'question' must be a non-empty string", text)
        return Action(kind=kind, question=question.strip())

    # FINAL_ANSWER
    content = data.get("content")
    if content is None:
        return ParseFailure("missing_field", "final_answer requires field 'content'", text)
    if not isinstance(content, str):
        return ParseFailure("field_type_error", f"'content' must be a string, got {type(content).__name__}", text)
    return Action(kind=kind, content=content)
