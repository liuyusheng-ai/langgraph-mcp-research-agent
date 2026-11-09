"""Web, local RAG and MCP research tools with one normalized interface."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Protocol

import requests

from .mcp_client import MCPClient
from .models import Evidence, ToolError, TransientToolError


class ResearchTool(Protocol):
    name: str
    available: bool

    def search(self, query: str, limit: int = 3) -> list[Evidence]:
        ...


def tokens(text: str) -> set[str]:
    """English words plus Chinese character bigrams, so partial Chinese phrases still match."""
    lowered = text.lower()
    words = set(re.findall(r"[a-z][a-z0-9_-]+", lowered))
    for run in re.findall(r"[一-鿿]+", lowered):
        words.update(run[i:i + 2] for i in range(len(run) - 1))
    return words


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
    available = True

    def __init__(self, documents: list[dict[str, str]] | None = None) -> None:
        self.documents = documents or DEFAULT_KNOWLEDGE

    def search(self, query: str, limit: int = 3) -> list[Evidence]:
        query_tokens = tokens(query)
        ranked: list[Evidence] = []
        for document in self.documents:
            overlap = len(query_tokens & tokens(f"{document['title']} {document['content']}"))
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
    available = True

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
                Evidence(**{**item.to_dict(), "tool": self.name})
                for item in self.offline.search(query, limit)
            ]
        try:
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
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise TransientToolError(f"网页搜索暂时不可用：{exc}") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientToolError(f"网页搜索返回 {response.status_code}")
        if response.status_code >= 400:
            raise ToolError(f"网页搜索请求被拒绝：{response.status_code}")
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


class MetaSearchTool:
    """Key-free web search through the `ddgs` metasearch package (Bing, DuckDuckGo and others).

    Optional dependency: `pip install ddgs`. Snippets are short, so evidence is a search-result
    summary rather than the full page; that is the same shape Tavily's basic search returns.
    """

    name = "web_search"

    def __init__(
        self, backends: tuple[str, ...] = ("bing", "auto"), rounds: int = 3, backoff_seconds: float = 2.0
    ) -> None:
        self.backends = backends
        # Free backends sometimes answer "no results" for queries that do have results (soft rate
        # limiting); in a 90-query eval run a quarter of queries came back empty without retries.
        self.rounds = rounds
        self.backoff_seconds = backoff_seconds
        try:
            from ddgs import DDGS  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def search(self, query: str, limit: int = 3) -> list[Evidence]:
        import time

        from ddgs import DDGS
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

        last_error: Exception | None = None
        for round_index in range(self.rounds):
            if round_index:
                time.sleep(self.backoff_seconds * round_index)
            for backend in self.backends:
                try:
                    results = DDGS(timeout=8).text(query, max_results=limit, backend=backend)
                except (RatelimitException, TimeoutException) as exc:
                    last_error = exc
                    continue
                except DDGSException:
                    # "No results found" from one backend: try the next one before calling it empty.
                    continue
                evidence = [
                    Evidence(
                        title=item.get("title") or "网页资料",
                        content=item.get("body") or "",
                        source=item["href"],
                        tool=self.name,
                    )
                    for item in results
                    if item.get("href") and item.get("body")
                ]
                if evidence:
                    return evidence[:limit]
        if last_error is not None:
            raise TransientToolError(f"网页搜索暂时不可用：{last_error}")
        return []


class MCPResearchTool:
    """Calls one tool on an MCP server; without a client it reports itself as not configured."""

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

    @property
    def available(self) -> bool:
        return self.client is not None

    def search(self, query: str, limit: int = 3) -> list[Evidence]:
        if self.client is None:
            raise ToolError("MCP 服务未配置")
        contents = self.client.call_tool_sync(
            self.tool_name, self.argument_builder(query, limit)
        )
        items: list = []
        for content in contents:
            try:
                parsed = json.loads(content)
            except (TypeError, ValueError):
                parsed = content
            items.extend(parsed if isinstance(parsed, list) else [parsed])
        return [self._to_evidence(item) for item in items[:limit]]

    def _to_evidence(self, item) -> Evidence:
        # Servers that return {"title", "source", "content"} keep their own source;
        # plain text gets a stable source from its content so duplicates collapse.
        if isinstance(item, dict) and item.get("content"):
            return Evidence(
                title=str(item.get("title") or "MCP 结果"),
                content=str(item["content"]),
                source=str(item.get("source") or self._content_source(item["content"])),
                tool=self.name,
            )
        content = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return Evidence(
            title="MCP 结果",
            content=content,
            source=self._content_source(content),
            tool=self.name,
        )

    def _content_source(self, content: str) -> str:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        return f"mcp://{self.tool_name}/{digest}"


@dataclass
class ResearchToolbox:
    tools: list[ResearchTool]

    @classmethod
    def default(cls, mcp_client: MCPClient | None = None) -> "ResearchToolbox":
        return cls([WebSearchTool(), LocalRAGTool(), MCPResearchTool(client=mcp_client)])
