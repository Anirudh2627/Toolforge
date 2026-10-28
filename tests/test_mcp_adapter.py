"""Tests for the optional MCP adapter.

Skipped when the `mcp` SDK is not installed — the core benchmark never needs
it. Includes a subprocess check proving the core package does not import MCP.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

mcp = pytest.importorskip("mcp", reason="optional MCP SDK not installed")

from toolforge.mcp_adapter import build_toolforge_server  # noqa: E402


def _roundtrip(family: str, calls: list[tuple[str, dict]]):
    """Run an in-process MCP client against the adapter; return (tools, results)."""
    from mcp.client.session import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    server, env = build_toolforge_server(family, seed=11)
    results = []

    async def main():
        async with create_client_server_memory_streams() as (client_stream, server_stream):
            task = asyncio.create_task(
                server.run(server_stream[0], server_stream[1],
                           server.create_initialization_options()))
            try:
                async with ClientSession(client_stream[0], client_stream[1]) as client:
                    await client.initialize()
                    listed = await client.list_tools()
                    for name, args in calls:
                        results.append(await client.call_tool(name, args))
                    return listed.tools
            finally:
                task.cancel()

    tools = asyncio.run(main())
    return tools, results, env


class TestMCPAdapter:
    def test_list_tools_matches_environment(self):
        tools, _, env = _roundtrip("calendar", [])
        names = {t.name for t in tools}
        assert names == set(env.tool_names())
        assert "calendar.create_event" in names
        by_name = {t.name: t for t in tools}
        schema = getattr(by_name["calendar.create_event"], "input_schema", None) or \
                 by_name["calendar.create_event"].inputSchema
        assert schema["required"] == ["title", "date", "start", "end"]

    def test_call_executes_and_mutates_state(self):
        _, results, env = _roundtrip("calendar", [
            ("calendar.create_event",
             {"title": "MCP event", "date": "2026-03-05", "start": "10:00", "end": "11:00"}),
        ])
        payload = json.loads(results[0].content[0].text)
        assert payload["ok"] is True
        assert not results[0].is_error
        assert any(e["title"] == "MCP event" for e in env.state["events"])

    def test_schema_violation_is_structured_error(self):
        _, results, env = _roundtrip("calendar", [
            ("calendar.create_event", {"title": "No dates"}),
        ])
        payload = json.loads(results[0].content[0].text)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "schema_violation"
        assert results[0].is_error
        assert env.state["events"] == env.default_state(11)["events"]  # unchanged

    def test_unknown_tool_is_structured_error(self):
        _, results, _ = _roundtrip("calendar", [("calendar.launch_rocket", {})])
        payload = json.loads(results[0].content[0].text)
        assert payload["ok"] is False and payload["error"]["code"] == "unknown_tool"

    def test_sql_adapter_blocks_writes(self):
        _, results, _ = _roundtrip("sql", [
            ("sql.run_query", {"sql": "DELETE FROM orders"}),
            ("sql.run_query", {"sql": "SELECT COUNT(*) FROM orders"}),
        ])
        denied = json.loads(results[0].content[0].text)
        assert denied["ok"] is False and denied["error"]["code"] == "unsafe_operation"
        allowed = json.loads(results[1].content[0].text)
        assert allowed["ok"] is True and allowed["output"]["rows"][0][0] == 20


class TestCoreIndependence:
    def test_core_never_imports_mcp(self):
        """Importing the full core (executor/evaluation/benchmarking) must not
        pull in the mcp SDK or the adapter module."""
        import os
        from pathlib import Path
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[1] / "src")
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        code = (
            "import sys, toolforge, toolforge.executor, toolforge.evaluation, "
            "toolforge.benchmarking, toolforge.tasks_builders, toolforge.inference;"
            "assert 'mcp' not in sys.modules, 'core imported mcp';"
            "assert 'toolforge.mcp_adapter' not in sys.modules, 'core imported the adapter';"
            "print('core-independent')"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              timeout=120, env=env)
        assert proc.returncode == 0, proc.stderr
        assert "core-independent" in proc.stdout
