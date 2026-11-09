"""Offline planner and writer that can be replaced by a real chat model."""

from __future__ import annotations

import re

# Terms the offline planner turns into focused sub-queries.
FOCUS_TERMS = ("LangGraph", "RAG", "MCP", "多智能体", "评测", "安全")


class DemoResearchModel:
    def plan(self, query: str) -> list[str]:
        """Return the sub-queries the researcher runs, the full question first."""
        sub_queries = [query]
        for term in FOCUS_TERMS:
            if re.search(re.escape(term), query, flags=re.I):
                sub_queries.append(term)
        return sub_queries[:3]

    def write_report(self, query: str, evidence: list[dict]) -> str:
        lines = [f"# {query}", "", "## 研究结论", ""]
        for index, item in enumerate(evidence, start=1):
            lines.append(f"- {item['content']} [{index}]")
        lines.extend(["", "## 参考来源", ""])
        for index, item in enumerate(evidence, start=1):
            lines.append(f"[{index}] {item['title']} - {item['source']}")
        return "\n".join(lines)


INSUFFICIENT_PREFIX = "资料不足"


class OllamaResearchModel(DemoResearchModel):
    """Planner and writer backed by a local model served by Ollama (standard library only).

    `plan_queries=False` turns the planner off (the original question is the only search),
    which is the ablation baseline for the Planner role. Token counts and latency of every
    call are kept in `self.calls`.
    """

    PLAN_PROMPT = (
        "你是研究助理中的规划者。把用户的问题拆成 2 个互补的网页搜索关键词，"
        "覆盖问题中不同的方面，每个不超过 20 个字，不要重复原问题。"
        '只输出 JSON 数组，例如 ["关键词一", "关键词二"]。'
    )
    WRITE_PROMPT = (
        "你是研究助理中的写作者。只根据给定的编号资料回答问题，用中文写一份简洁的研究报告（不超过 300 字）。\n"
        "规则：\n"
        "1. 每一句事实陈述后面都用 [编号] 标注它依据的资料，只能使用给出的编号；\n"
        "2. 资料里没有的内容不要写，不要用常识补充；\n"
        f"3. 如果资料无法回答这个问题（例如问的是未来、私人或资料里不存在的信息），只输出一行：{INSUFFICIENT_PREFIX}：<一句话说明资料缺少什么>。"
    )

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        host: str = "http://localhost:11434",
        plan_queries: bool = True,
        temperature: float = 0.0,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.plan_queries = plan_queries
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.calls: list[dict] = []

    def plan(self, query: str) -> list[str]:
        if not self.plan_queries:
            return [query]
        content = self.chat(self.PLAN_PROMPT, query, role="planner")
        sub_queries = [query]
        for item in _json_list(content):
            text = str(item).strip()
            if text and text.lower() != query.lower() and text not in sub_queries:
                sub_queries.append(text)
        return sub_queries[:3]

    def write_report(self, query: str, evidence: list[dict]) -> str:
        sources = "\n".join(
            f"[{index}] {item['title']}：{item['content']}" for index, item in enumerate(evidence, start=1)
        )
        body = self.chat(self.WRITE_PROMPT, f"问题：{query}\n\n资料：\n{sources}", role="writer").strip()
        if body.startswith(INSUFFICIENT_PREFIX):
            return body.splitlines()[0]
        lines = [f"# {query}", "", body, "", "## 参考来源", ""]
        lines += [f"[{index}] {item['title']} - {item['source']}" for index, item in enumerate(evidence, start=1)]
        return "\n".join(lines)

    def chat(self, system: str, user: str, role: str = "chat") -> str:
        import json
        import time
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {"temperature": self.temperature, "num_ctx": 8192},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read())
        self.calls.append({
            "role": role,
            "prompt_tokens": payload.get("prompt_eval_count", 0),
            "completion_tokens": payload.get("eval_count", 0),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        })
        return payload["message"]["content"]


def _json_list(text: str) -> list:
    """First JSON array in the model's answer; an unparsable answer yields no sub-queries."""
    import json

    match = re.search(r"\[.*?\]", text, flags=re.S)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return []
    return parsed if isinstance(parsed, list) else []
