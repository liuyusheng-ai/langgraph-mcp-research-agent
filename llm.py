"""Offline planner and writer that can be replaced by a real chat model."""

from __future__ import annotations


class DemoResearchModel:
    def plan(self, query: str) -> list[str]:
        return [
            f"明确研究范围与关键概念：{query}",
            f"检索公开资料并记录来源：{query}",
            f"检索本地知识库与 MCP 工具：{query}",
        ]

    def write_report(self, query: str, evidence: list[dict]) -> str:
        if not evidence:
            return "资料不足：当前没有可核验来源，系统停止生成结论性报告。"

        lines = [f"# {query}", "", "## 研究结论", ""]
        for index, item in enumerate(evidence, start=1):
            lines.append(f"- {item['content']} [{index}]")
        lines.extend(["", "## 参考来源", ""])
        for index, item in enumerate(evidence, start=1):
            lines.append(f"[{index}] {item['title']} - {item['source']}")
        return "\n".join(lines)

