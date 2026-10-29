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
