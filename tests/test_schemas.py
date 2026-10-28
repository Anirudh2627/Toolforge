"""Tests for the JSON-schema subset validator, ToolSpec, and action grammar."""

from __future__ import annotations

import json

import pytest

from toolforge.schemas import (
    Action,
    ActionKind,
    InvalidToolSpecError,
    ParseFailure,
    ToolSpec,
    check_schema_supported,
    extract_json_object,
    parse_action,
    render_tool_catalog,
    validate_against_schema,
)

# --------------------------------------------------------------------------
# Schema-subset validator
# --------------------------------------------------------------------------

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "role": {"type": "string", "enum": ["admin", "user"]},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "address": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    "required": ["name"],
    "additionalProperties": False,
}


class TestSchemaValidator:
    def test_valid_object(self):
        issues = validate_against_schema(
            {"name": "Ana", "age": 30, "role": "admin", "tags": ["x"],
             "address": {"city": "Berlin"}}, PERSON_SCHEMA)
        assert issues == []

    def test_missing_required(self):
        issues = validate_against_schema({"age": 3}, PERSON_SCHEMA)
        kinds = {(i.path, i.kind) for i in issues}
        assert ("$.name", "missing_required") in kinds

    def test_unexpected_property(self):
        issues = validate_against_schema({"name": "Ana", "salary": 1}, PERSON_SCHEMA)
        assert any(i.kind == "unexpected_property" and i.path == "$.salary" for i in issues)

    def test_type_errors(self):
        issues = validate_against_schema({"name": 42}, PERSON_SCHEMA)
        assert issues[0].kind == "type_error"

    def test_bool_is_not_integer(self):
        # Python bool is a subclass of int; the validator must not accept it as integer.
        issues = validate_against_schema({"name": "a", "age": True}, PERSON_SCHEMA)
        assert any(i.kind == "type_error" and i.path == "$.age" for i in issues)

    def test_enum_violation(self):
        issues = validate_against_schema({"name": "a", "role": "root"}, PERSON_SCHEMA)
        assert any(i.kind == "enum_violation" for i in issues)

    def test_range_and_length(self):
        issues = validate_against_schema({"name": "", "age": 200}, PERSON_SCHEMA)
        kinds = {i.kind for i in issues}
        assert "length_violation" in kinds and "range_violation" in kinds

    def test_items_and_max_items(self):
        issues = validate_against_schema({"name": "a", "tags": ["x", "y", "z", "w"]}, PERSON_SCHEMA)
        assert any(i.kind == "length_violation" and i.path == "$.tags" for i in issues)
        issues = validate_against_schema({"name": "a", "tags": ["x", 2]}, PERSON_SCHEMA)
        assert any(i.kind == "type_error" and i.path == "$.tags[1]" for i in issues)

    def test_nested_required_path(self):
        issues = validate_against_schema({"name": "a", "address": {}}, PERSON_SCHEMA)
        assert any(i.path == "$.address.city" and i.kind == "missing_required" for i in issues)

    def test_pattern(self):
        schema = {"type": "object",
                  "properties": {"date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}}}
        assert validate_against_schema({"date": "2026-03-02"}, schema) == []
        issues = validate_against_schema({"date": "03/02/2026"}, schema)
        assert issues[0].kind == "pattern_violation"

    def test_union_type(self):
        schema = {"type": ["string", "null"]}
        assert validate_against_schema(None, schema) == []
        assert validate_against_schema("x", schema) == []
        assert validate_against_schema(3, schema)[0].kind == "type_error"


class TestToolSpec:
    def test_roundtrip(self):
        spec = ToolSpec("calendar.create_event", "Create an event.", PERSON_SCHEMA)
        assert spec.family == "calendar"
        assert spec.local_name == "create_event"
        assert ToolSpec.from_dict(spec.to_dict()) == spec

    @pytest.mark.parametrize("bad_name", ["create_event", "Calendar.create_event",
                                          "cal..x", "cal. space", ""])
    def test_bad_names_rejected(self, bad_name):
        with pytest.raises(InvalidToolSpecError):
            ToolSpec(bad_name, "desc", {"type": "object"})

    def test_unsupported_keyword_rejected(self):
        with pytest.raises(InvalidToolSpecError, match="anyOf"):
            ToolSpec("cal.x", "desc", {"type": "object", "anyOf": []})

    def test_empty_description_rejected(self):
        with pytest.raises(InvalidToolSpecError):
            ToolSpec("cal.x", "   ", {"type": "object"})

    def test_check_schema_supported_rejects_items_list_form(self):
        with pytest.raises(InvalidToolSpecError):
            check_schema_supported({"type": "array", "items": [{"type": "string"}]})

    def test_render_catalog_is_valid_json(self):
        specs = [ToolSpec("a.b", "Does b.", {"type": "object"})]
        parsed = json.loads(render_tool_catalog(specs))
        assert parsed[0]["name"] == "a.b"


# --------------------------------------------------------------------------
# Action grammar
# --------------------------------------------------------------------------

class TestExtractJsonObject:
    def test_plain(self):
        assert extract_json_object('{"a": 1}') == '{"a": 1}'

    def test_fenced_and_prose(self):
        text = 'Sure! Here you go:\n```json\n{"action": "clarify", "question": "when?"}\n```\nDone.'
        assert json.loads(extract_json_object(text))["action"] == "clarify"

    def test_braces_inside_strings(self):
        text = '{"content": "use {curly} braces"} trailing'
        assert json.loads(extract_json_object(text))["content"] == "use {curly} braces"

    def test_escaped_quotes(self):
        text = r'{"content": "he said \"hi\""}'
        assert json.loads(extract_json_object(text))["content"] == 'he said "hi"'

    def test_unbalanced_returns_none(self):
        assert extract_json_object('{"a": 1') is None

    def test_no_brace_returns_none(self):
        assert extract_json_object("just prose") is None


class TestParseAction:
    def test_tool_call(self):
        action = parse_action('{"action": "tool_call", "tool": "calendar.list_events", '
                              '"arguments": {"date": "2026-03-02"}}')
        assert isinstance(action, Action)
        assert action.kind is ActionKind.TOOL_CALL
        assert action.tool == "calendar.list_events"
        assert action.arguments == {"date": "2026-03-02"}

    def test_tool_call_arguments_default_to_empty(self):
        action = parse_action('{"action": "tool_call", "tool": "sql.list_tables"}')
        assert isinstance(action, Action) and action.arguments == {}

    def test_clarify_and_final(self):
        c = parse_action('{"action": "clarify", "question": "Which day?"}')
        assert isinstance(c, Action) and c.kind is ActionKind.CLARIFY
        f = parse_action('{"action": "final_answer", "content": "done"}')
        assert isinstance(f, Action) and f.kind is ActionKind.FINAL_ANSWER

    def test_empty_content_allowed_for_final_answer(self):
        f = parse_action('{"action": "final_answer", "content": ""}')
        assert isinstance(f, Action)

    @pytest.mark.parametrize("text,category", [
        ("", "invalid_json"),
        ("I will now call a tool", "invalid_json"),
        ("{action: tool_call}", "invalid_json"),
        ('["action", "tool_call"]', "invalid_json"),
        ('{"action": 3}', "field_type_error"),
        ('{"tool": "cal.x", "arguments": {}}', "missing_field"),
        ('{"action": "teleport", "where": "home"}', "unknown_action"),
        ('{"action": "tool_call", "tool": "cal.x", "arguments": {}, "reason": "because"}',
         "unexpected_field"),
        ('{"action": "tool_call", "arguments": {}}', "missing_field"),
        ('{"action": "tool_call", "tool": "no_family_prefix", "arguments": {}}',
         "tool_name_format"),
        ('{"action": "tool_call", "tool": "cal.x", "arguments": "not-an-object"}',
         "field_type_error"),
        ('{"action": "clarify"}', "missing_field"),
        ('{"action": "clarify", "question": "   "}', "field_type_error"),
        ('{"action": "final_answer"}', "missing_field"),
    ])
    def test_failures_are_categorized(self, text, category):
        failure = parse_action(text)
        assert isinstance(failure, ParseFailure)
        assert failure.category == category
        assert failure.to_dict()["category"] == category

    def test_action_roundtrip_dict(self):
        action = parse_action('{"action": "tool_call", "tool": "a.b", "arguments": {"x": 1}}')
        assert isinstance(action, Action)
        assert action.to_dict() == {"action": "tool_call", "tool": "a.b", "arguments": {"x": 1}}
