"""Optional client for calling tools exposed by a local MCP stdio server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MCPServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


class MCPClient:
    """Use the official MCP Python SDK only when a server is configured."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config

    async def list_tools(self) -> list[str]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.list_tools()
                return [tool.name for tool in response.tools]

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> list[str]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.call_tool(tool_name, arguments)
                return [
                    getattr(item, "text", str(item)) for item in response.content
                ]

