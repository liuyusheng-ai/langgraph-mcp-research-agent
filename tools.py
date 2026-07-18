"""Web, local RAG and MCP research tools with one normalized interface."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Callable, Protocol

import requests

from mcp_client import MCPClient
from models import Evidence


class ResearchTool(Protocol):
    name: str

    def search(self, query: str, limit: int = 3) -> list[Evidence]:
        ...


def _tokens(text: str) -> set[str]:
    return set(
        re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
    )


DEFAULT_KNOWLEDGE = [
    {
        "title": "LangGraph 工作流",
        "source": "local://langgraph-workflow",
        "content": (
            "LangGraph 使用共享状态和有向图组织 Agent 节点，适合包含规划、"
            "工具调用、条件路由和持久化的长流程任务。"
        ),
    },
    {
        "title": "RAG 证据链",
        "source": "local://rag-evidence",
        "content": (
            "RAG 应保存来源、切片和检索分数，并在生成阶段要求答案引用证据；"
            "当资料不足时应明确提示，而不是补写未经验证的事实。"
        ),
    },
    {
        "title": "MCP 工具协议",
        "source": "local://mcp-tools",
        "content": (
            "MCP 为模型上下文、资源与工具调用提供统一协议。客户端在调用工具前"
            "应获取工具 Schema，并对参数、超时和返回内容进行验证。"
        ),
    },
]


class LocalRAGTool:
    name = "local_rag"

    def __init__(self, documents: list[dict[str, str]] | None = None) -> None:
        self.documents = documents or DEFAULT_KNOWLEDGE

    def search(self, query: str, limit: int = 3) -> list[Evidence]:
        query_tokens = _tokens(query)
        ranked: list[Evidence] = []
        for document in self.documents:
            haystack = f"{document['title']} {document['content']}"
            overlap = len(query_tokens & _tokens(haystack))
            if overlap == 0:
                continue
            ranked.append(
                Evidence(
                    title=document["title"],
                    content=document["content"],
                    source=document["source"],
                    tool=self.name,
                    score=float(overlap),
                )
            )
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]


class WebSearchTool:
    """Tavily-backed web search with an explicit offline demo fallback."""

    name = "web_search"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.offline = LocalRAGTool(
            [
                {
                    "title": "公开资料：Agent 工程实践",
                    "source": "demo://web/agent-engineering",
                    "content": (
                        "企业 Agent 项目通常需要规划、检索、工具调用、来源追踪、"
                        "评测与安全控制，并通过回归测试保证迭代稳定性。"
                    ),
                },
                {
                    "title": "公开资料：多智能体协作",
                    "source": "demo://web/multi-agent",
                    "content": (
                        "多智能体系统可按 Planner、Researcher、Writer 划分职责，"
                        "使用共享状态传递任务、证据和最终报告。"
                    ),
                },
            ]
        )

    def search(self, query: str, limit: int = 3) -> list[Evidence]:
        if not self.api_key:
            return [
                Evidence(
                    **{
                        **item.to_dict(),
                        "tool": self.name,
                    }
                )
                for item in self.offline.search(query, limit)
            ]

        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": limit,
                "search_depth": "basic",
            },
            timeout=10,
        )
        response.raise_for_status()
        return [
            Evidence(
                title=item.get("title", "网页资料"),
                content=item.get("content", ""),
                source=item.get("url", ""),
                tool=self.name,
                score=float(item.get("score", 0)),
            )
            for item in response.json().get("results", [])
            if item.get("url")
        ]


class MCPResearchTool:
    name = "mcp"

    def __init__(
        self,
        client: MCPClient | None = None,
        tool_name: str = "search",
        argument_builder: Callable[[str, int], dict] | None = None,
    ) -> None:
        self.client = client
        self.tool_name = tool_name
        self.argument_builder = argument_builder or (
            lambda query, limit: {"query": query, "limit": limit}
        )

    def search(self, query: str, limit: int = 3) -> list[Evidence]:
        if self.client is None:
            return []
        contents = asyncio.run(
            self.client.call_tool(
                self.tool_name, self.argument_builder(query, limit)
            )
        )
        return [
            Evidence(
                title=f"MCP result {index}",
                content=content,
                source=f"mcp://{self.tool_name}/{index}",
                tool=self.name,
            )
            for index, content in enumerate(contents[:limit], start=1)
        ]


@dataclass
class ResearchToolbox:
    tools: list[ResearchTool]

    @classmethod
    def default(cls) -> "ResearchToolbox":
        return cls([WebSearchTool(), LocalRAGTool(), MCPResearchTool()])

