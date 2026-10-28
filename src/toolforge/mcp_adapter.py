"""Optional MCP adapter: expose ToolForge mock tools over Model Context Protocol.

Scope distinction (important):

- The **internal executor** (:mod:`toolforge.executor`) routes tool calls
  in-process and is what the benchmark, training, and evaluation use. It has
  zero MCP dependency.
- This **adapter** wraps the exact same :class:`Environment` instances and
  ``ToolSpec`` schemas behind an MCP server so external MCP clients (Claude
  Desktop, inspectors, other agents) can call the mock tools. It is a
  compatibility/demo surface, never part of evaluation scoring.

Requires ``mcp>=2.0`` (developed and tested against mcp 2.3.0; a v1-style
decorator fallback is attempted when the v2 low-level API is absent). The
core package never imports this module; importing it without the SDK raises
:class:`MCPAdapterUnavailable` with install instructions.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from toolforge import get_logger
from toolforge.environments.base import Environment
from toolforge.registry import EnvironmentRegistry, default_registry
from toolforge.schemas import ToolForgeError

logger = get_logger("mcp_adapter")


class MCPAdapterUnavailable(ToolForgeError):
    """Raised when the optional ``mcp`` SDK is not installed."""


def _import_mcp() -> tuple[Any, Any]:
    try:
        from mcp.server.lowlevel import Server
        import mcp.types as types
        return Server, types
    except ImportError as exc:
        raise MCPAdapterUnavailable(
            "The MCP adapter requires the optional 'mcp' package: "
            "pip install 'mcp>=2.0' (core ToolForge never needs it)."
        ) from exc


def build_toolforge_server(family: str, *, fixture: dict[str, Any] | None = None,
                           seed: int = 0, registry: EnvironmentRegistry | None = None,
                           server_name: str | None = None) -> tuple[Any, Environment]:
    """Build an MCP server exposing one environment family's tools.

    Returns ``(server, environment)`` — keep a reference to the environment to
    inspect its state after calls (state lives in-process; each server owns an
    isolated instance, exactly like an episode).
    """
    Server, types = _import_mcp()
    registry = registry or default_registry()
    env = registry.create(family, fixture=fixture, seed=seed)
    server = Server(server_name or f"toolforge-{family}")

    tool_list = [
        types.Tool(name=spec.name, description=spec.description, inputSchema=spec.parameters)
        for spec in env.tools
    ]

    def _execute(name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        if "." not in name or name.split(".", 1)[0] != family:
            payload = {"ok": False,
                       "error": {"code": "unknown_tool",
                                 "message": f"Tool {name!r} is not served by this adapter; "
                                            f"available: {env.tool_names()}"}}
            return json.dumps(payload), True
        local = name.split(".", 1)[1]
        # Schema validation mirrors the executor so MCP clients get the same
        # structured errors as the internal loop.
        from toolforge.schemas import validate_against_schema
        spec = next((s for s in env.tools if s.name == name), None)
        if spec is not None:
            issues = validate_against_schema(arguments or {}, spec.parameters, "arguments")
            if issues:
                payload = {"ok": False,
                           "error": {"code": "schema_violation",
                                     "message": "; ".join(f"{i.path}: {i.message}"
                                                          for i in issues)}}
                return json.dumps(payload), True
        result = env.execute(local, arguments or {})
        return json.dumps(result.to_dict()), not result.ok

    if hasattr(types, "CallToolRequestParams"):
        # mcp 2.x low-level API (verified against mcp 2.3.0).
        async def on_list_tools(ctx: Any, params: Any) -> Any:
            return types.ListToolsResult(tools=tool_list)

        async def on_call_tool(ctx: Any, params: Any) -> Any:
            text, is_error = _execute(params.name, dict(params.arguments or {}))
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)], isError=is_error)

        server.add_request_handler("tools/list", types.PaginatedRequestParams, on_list_tools)
        server.add_request_handler("tools/call", types.CallToolRequestParams, on_call_tool)
    else:  # pragma: no cover - v1 fallback path (not exercised on mcp>=2)
        @server.list_tools()
        async def _list_tools() -> list:  # type: ignore[no-redef]
            return tool_list

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict) -> list:  # type: ignore[no-redef]
            text, is_error = _execute(name, arguments or {})
            return [types.TextContent(type="text", text=text)]

    logger.info("Built MCP server 'toolforge-%s' with %d tools", family, len(tool_list))
    return server, env


def run_stdio_server(family: str, **kwargs: Any) -> None:
    """Run one family's adapter over stdio until interrupted (blocking)."""
    from mcp.server.stdio import stdio_server

    server, _env = build_toolforge_server(family, **kwargs)

    async def _main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(_main())


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import argparse

    parser = argparse.ArgumentParser(description="Serve ToolForge mock tools over MCP (stdio).")
    parser.add_argument("family", choices=["calendar", "email", "files", "crm", "sql", "todo"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_stdio_server(args.family, seed=args.seed)
