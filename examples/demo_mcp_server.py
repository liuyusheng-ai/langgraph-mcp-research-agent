"""A small stdio MCP server used by the demo, the tests and the evaluation.

Run it through the agent with `python cli.py "问题" --demo-mcp`.
"""

from __future__ import annotations

import json
import re

from mcp.server.fastmcp import FastMCP

DOCUMENTS = [
    {
        "title": "MCP 服务的权限边界",
        "source": "mcp-demo://security/permissions",
        "content": "MCP 服务应按工具声明最小权限，对参数做 Schema 校验，并对返回内容脱敏后再交给模型。",
    },
    {
        "title": "Agent 评测指标",
        "source": "mcp-demo://eval/metrics",
        "content": "Agent 评测通常关注任务完成率、工具调用成功率、引用覆盖率与无依据回答率。",
    },
    {
        "title": "LangGraph 断点恢复",
        "source": "mcp-demo://langgraph/checkpoint",
        "content": "LangGraph 可以把每个节点结束后的状态写入检查点，任务中断后从最后完成的节点继续。",
    },
]

server = FastMCP("research-demo")


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-z][a-z0-9_-]+", lowered))
    for run in re.findall(r"[一-鿿]+", lowered):
        words.update(run[i:i + 2] for i in range(len(run) - 1))
    return words


@server.tool()
def search(query: str, limit: int = 3) -> str:
    """Search the demo documents; returns a JSON list of {title, source, content}."""
    if not query.strip():
        raise ValueError("query 不能为空")
    limit = max(1, min(int(limit), 5))
    wanted = _tokens(query)
    scored = [
        (len(wanted & _tokens(doc["title"] + doc["content"])), doc) for doc in DOCUMENTS
    ]
    hits = [doc for score, doc in sorted(scored, key=lambda pair: -pair[0]) if score > 0]
    return json.dumps(hits[:limit], ensure_ascii=False)


if __name__ == "__main__":
    server.run()
