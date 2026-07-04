"""MCP stdio client: launch the engine and talk to it."""

import os
import shlex
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def engine_session(engine_cmd: str, data_dir: str):
    """Launch the engine over stdio and yield a ready ClientSession."""
    command, *args = shlex.split(engine_cmd)
    env = {**os.environ, "CONTEXT1_DATA_DIR": data_dir}
    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def anthropic_tools(session: ClientSession) -> list[dict]:
    """Convert the engine's MCP tools to Anthropic tool schemas.

    The `exclude_chunk_ids` property is stripped: the harness injects it itself,
    the model never sees or controls it.
    """
    result = await session.list_tools()
    tools = []
    for tool in result.tools:
        schema = dict(tool.inputSchema)
        properties = dict(schema.get("properties", {}))
        properties.pop("exclude_chunk_ids", None)
        schema["properties"] = properties
        required = schema.get("required")
        if required:
            schema["required"] = [r for r in required if r != "exclude_chunk_ids"]
        tools.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": schema,
            }
        )
    return tools


async def call(session: ClientSession, name: str, args: dict) -> str:
    """Call an MCP tool and return its text result (concatenated text blocks)."""
    result = await session.call_tool(name, args)
    text = "".join(block.text for block in result.content if getattr(block, "type", None) == "text")
    if result.isError:
        return f"Tool error: {text}"
    return text
