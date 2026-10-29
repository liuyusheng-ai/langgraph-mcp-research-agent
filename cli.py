"""Command-line demo for the research Agent."""

from __future__ import annotations

import argparse
import json
import sys

from research_agent import ResearchAgent
from research_agent.mcp_client import MCPClient, MCPServerConfig
from research_agent.tools import ResearchToolbox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="需要研究的问题")
    parser.add_argument("--run-id", help="复用的运行 ID")
    parser.add_argument("--resume", action="store_true", help="从检查点继续未完成的任务")
    parser.add_argument("--demo-mcp", action="store_true", help="连接仓库自带的本地 stdio MCP 服务")
    args = parser.parse_args()

    client = None
    if args.demo_mcp:
        client = MCPClient(MCPServerConfig(command=sys.executable, args=["examples/demo_mcp_server.py"]))
    agent = ResearchAgent(toolbox=ResearchToolbox.default(mcp_client=client))
    result = agent.run(args.query, args.run_id, args.resume)
    print(result["report"])
    print("\n--- Tool calls ---")
    for call in result["tool_calls"]:
        print(f"{call['tool']:<11} {call['status']:<16} {call['result_count']} 条  {call['query']}")
    print("\n--- Evaluation ---")
    print(json.dumps(result["evaluation"], ensure_ascii=False, indent=2))
    print(f"\nrun_id={result['run_id']}")


if __name__ == "__main__":
    main()
