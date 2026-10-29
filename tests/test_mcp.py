import sys
from pathlib import Path

import pytest

from research_agent.mcp_client import MCPClient, MCPServerConfig
from research_agent.models import ToolError
from research_agent.tools import MCPResearchTool

SERVER = Path(__file__).resolve().parents[1] / "examples" / "demo_mcp_server.py"


@pytest.fixture()
def client() -> MCPClient:
    return MCPClient(MCPServerConfig(command=sys.executable, args=[str(SERVER)], timeout_seconds=30))


def test_stdio_round_trip_returns_sourced_evidence(client: MCPClient) -> None:
    results = MCPResearchTool(client=client).search("MCP 权限 Schema 校验", limit=2)
    assert results
    assert results[0].source == "mcp-demo://security/permissions"
    assert results[0].tool == "mcp"


def test_unknown_mcp_tool_is_a_permanent_error(client: MCPClient) -> None:
    with pytest.raises(ToolError, match="没有工具"):
        MCPResearchTool(client=client, tool_name="delete_everything").search("x")


def test_server_side_error_is_reported(client: MCPClient) -> None:
    with pytest.raises(ToolError, match="返回错误"):
        MCPResearchTool(client=client).search("   ")
