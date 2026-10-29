"""Client for tools exposed by a local MCP stdio server, using the official MCP Python SDK."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .models import ToolError, TransientToolError


@dataclass(frozen=True)
class MCPServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    timeout_seconds: float = 10.0


class MCPClient:
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config

    async def list_tools(self) -> list[str]:
        async with self._session() as session:
            response = await session.list_tools()
            return [tool.name for tool in response.tools]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> list[str]:
        # Errors are raised after the session closes, so they are not wrapped in an ExceptionGroup.
        response = None
        async with self._session() as session:
            available = {tool.name for tool in (await session.list_tools()).tools}
            if tool_name in available:
                response = await session.call_tool(tool_name, arguments)
        if response is None:
            raise ToolError(f"MCP 服务没有工具 {tool_name}，可用：{sorted(available)}")
        texts = [getattr(item, "text", str(item)) for item in response.content]
        if response.isError:
            raise ToolError(f"MCP 工具 {tool_name} 返回错误：{' '.join(texts)[:200]}")
        return texts

    def call_tool_sync(self, tool_name: str, arguments: dict[str, Any]) -> list[str]:
        try:
            return asyncio.run(
                asyncio.wait_for(self.call_tool(tool_name, arguments), self.config.timeout_seconds)
            )
        except asyncio.TimeoutError as exc:
            raise TransientToolError(f"MCP 调用超过 {self.config.timeout_seconds} 秒") from exc
        except (ConnectionError, OSError) as exc:
            raise TransientToolError(f"MCP 服务连接失败：{exc}") from exc

    def _session(self):
        from contextlib import asynccontextmanager

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.config.command, args=self.config.args, env=self.config.env
        )

        @asynccontextmanager
        async def session():
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as client_session:
                    await client_session.initialize()
                    yield client_session

        return session()
