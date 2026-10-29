"""Offline, reproducible evaluation: research tasks and harness fault scenarios.

    python eval/run_eval.py            # writes eval/results.json and eval/RESULTS.md

Research tasks run the default toolbox (offline web fallback, local RAG) plus the
bundled stdio MCP server, so every tool call is real but no network or model is used.
Latency therefore covers the graph, the tools and the MCP subprocess, not a language model.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_agent import ResearchAgent  # noqa: E402
from research_agent.checkpoint import SQLiteCheckpointStore  # noqa: E402
from research_agent.llm import DemoResearchModel  # noqa: E402
from research_agent.mcp_client import MCPClient, MCPServerConfig  # noqa: E402
from research_agent.models import Evidence, ToolError, TransientToolError  # noqa: E402
from research_agent.tools import LocalRAGTool, MCPResearchTool, ResearchToolbox  # noqa: E402


class FaultTool:
    def __init__(self, name: str, script: list) -> None:
        self.name, self.available, self.script, self.calls = name, True, list(script), 0

    def search(self, query: str, limit: int = 3):
        self.calls += 1
        step = self.script.pop(0) if self.script else []
        if isinstance(step, BaseException):
            raise step
        if step == "sleep":
            time.sleep(0.5)
            return []
        return step


def ev(source: str) -> Evidence:
    return Evidence(title=source, content=f"{source} 的资料", source=source, tool="fault")


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return round(ordered[index], 2)


def run_research_tasks(tasks: list[dict], workdir: Path) -> list[dict]:
    client = MCPClient(MCPServerConfig(command=sys.executable, args=[str(ROOT / "examples" / "demo_mcp_server.py")]))
    agent = ResearchAgent(
        toolbox=ResearchToolbox.default(mcp_client=client),
        checkpoints=SQLiteCheckpointStore(str(workdir / "research.db")),
    )
    rows = []
    for task in tasks:
        started = time.perf_counter()
        result = agent.run(task["query"])
        elapsed = (time.perf_counter() - started) * 1000
        metrics = result["evaluation"]
        if task["answerable"]:
            passed = metrics["task_completed"] and metrics["evidence_count"] >= task["min_evidence"]
        else:
            passed = result["status"] == "insufficient_evidence" and "资料不足" in result["report"]
        rows.append({
            "id": task["id"],
            "query": task["query"],
            "answerable": task["answerable"],
            "status": result["status"],
            "passed": bool(passed),
            "elapsed_ms": round(elapsed, 2),
            **{key: metrics[key] for key in (
                "task_completed", "citation_coverage", "tool_success_rate",
                "effective_result_rate", "evidence_count", "tool_call_count", "harness_blocked_count",
            )},
        })
    return rows


def run_harness(kind: str, workdir: Path) -> tuple[bool, str]:
    store = SQLiteCheckpointStore(str(workdir / f"{kind}.db"))

    def agent(tools, **kwargs):
        return ResearchAgent(toolbox=ResearchToolbox(tools), checkpoints=store, **kwargs)

    if kind == "budget":
        tool = FaultTool("t", [[ev("a")], [ev("b")], [ev("c")]])
        result = agent([tool], call_budget=2).run("RAG 与 MCP")
        return tool.calls == 2 and result["stopped_reason"] == "budget_exhausted", f"calls={tool.calls}"
    if kind == "loop":
        class Repeat(DemoResearchModel):
            def plan(self, query):
                return [query, query.upper()]
        tool = FaultTool("t", [[ev("a")], [ev("b")]])
        result = agent([tool], model=Repeat()).run("mcp")
        statuses = [call["status"] for call in result["tool_calls"]]
        return statuses == ["success", "loop_blocked"] and result["remaining_budget"] == 7, str(statuses)
    if kind == "timeout":
        tool = FaultTool("t", ["sleep", "sleep"])
        result = agent([tool], timeout_seconds=0.05, max_retries=1).run("x")
        call = result["tool_calls"][0]
        ok = call["status"] == "timeout" and call["attempts"] == 2 and result["status"] == "insufficient_evidence"
        return ok, f"{call['status']} attempts={call['attempts']}"
    if kind == "transient_retry":
        tool = FaultTool("t", [TransientToolError("503"), [ev("a")]])
        call = agent([tool], max_retries=1).run("x")["tool_calls"][0]
        return call["status"] == "success" and call["attempts"] == 2, f"attempts={call['attempts']}"
    if kind == "permanent_error":
        tool = FaultTool("t", [ToolError("bad args"), [ev("a")]])
        call = agent([tool], max_retries=3).run("x")["tool_calls"][0]
        return call["status"] == "failed" and call["attempts"] == 1, f"attempts={call['attempts']}"
    if kind == "mcp_not_configured":
        result = agent([LocalRAGTool(), MCPResearchTool()]).run("RAG 证据")
        statuses = {call["tool"]: call["status"] for call in result["tool_calls"]}
        ok = statuses.get("mcp") == "not_configured" and result["evaluation"]["tool_success_rate"] == 1.0
        return ok, str(statuses)
    if kind == "no_evidence":
        result = agent([]).run("完全未知主题")
        return result["status"] == "insufficient_evidence" and "资料不足" in result["report"], result["status"]
    if kind == "resume":
        try:
            agent([FaultTool("a", [[ev("a")]]), FaultTool("b", [KeyboardInterrupt()])]).run("x", run_id="r")
        except KeyboardInterrupt:
            pass
        tool_a, tool_b = FaultTool("a", [[ev("again")]]), FaultTool("b", [[ev("b")]])
        result = agent([tool_a, tool_b]).run("x", run_id="r", resume=True)
        ok = tool_a.calls == 0 and tool_b.calls == 1 and result["status"] == "completed"
        return ok, f"a={tool_a.calls} b={tool_b.calls}"
    raise ValueError(kind)


def main() -> int:
    spec = json.loads((ROOT / "eval" / "scenarios.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        research = run_research_tasks(spec["research_tasks"], workdir)
        harness = []
        for scenario in spec["harness_scenarios"]:
            passed, detail = run_harness(scenario["kind"], workdir)
            harness.append({**scenario, "passed": passed, "detail": detail})

    answerable = [row for row in research if row["answerable"]]
    latencies = [row["elapsed_ms"] for row in research]
    summary = {
        "research_tasks": len(research),
        "research_passed": sum(row["passed"] for row in research),
        "task_completion_rate": round(sum(row["task_completed"] for row in answerable) / len(answerable), 4),
        "citation_coverage": round(statistics.mean(row["citation_coverage"] for row in answerable), 4),
        "tool_success_rate": round(statistics.mean(row["tool_success_rate"] for row in answerable), 4),
        "effective_result_rate": round(statistics.mean(row["effective_result_rate"] for row in answerable), 4),
        "unanswerable_refused": sum(row["passed"] for row in research if not row["answerable"]),
        "unanswerable_total": sum(not row["answerable"] for row in research),
        "harness_passed": sum(row["passed"] for row in harness),
        "harness_total": len(harness),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
    }
    out = {"summary": summary, "research": research, "harness": harness}
    (ROOT / "eval" / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "eval" / "RESULTS.md").write_text(render_markdown(out), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    all_passed = summary["research_passed"] == summary["research_tasks"] and summary["harness_passed"] == summary["harness_total"]
    return 0 if all_passed else 1


def render_markdown(out: dict) -> str:
    s = out["summary"]
    lines = [
        "# 评测结果",
        "",
        "由 `python eval/run_eval.py` 生成。离线运行：网页搜索使用演示资料，MCP 为仓库自带的 stdio 服务，不调用大模型。",
        "",
        "| 指标 | 结果 |",
        "| --- | --- |",
        f"| 研究任务通过 | {s['research_passed']}/{s['research_tasks']} |",
        f"| 可回答任务完成率 | {s['task_completion_rate']:.0%} |",
        f"| 引用覆盖率 | {s['citation_coverage']:.0%} |",
        f"| 工具调用成功率 | {s['tool_success_rate']:.0%} |",
        f"| 有效工具结果率 | {s['effective_result_rate']:.0%} |",
        f"| 无资料问题正确停止 | {s['unanswerable_refused']}/{s['unanswerable_total']} |",
        f"| Harness 场景通过 | {s['harness_passed']}/{s['harness_total']} |",
        f"| 单任务延迟 P50 / P95 | {s['latency_p50_ms']} ms / {s['latency_p95_ms']} ms |",
        "",
        "## 研究任务",
        "",
        "| ID | 问题 | 状态 | 证据数 | 引用覆盖 | 耗时 ms | 通过 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in out["research"]:
        lines.append(
            f"| {row['id']} | {row['query']} | {row['status']} | {row['evidence_count']} | "
            f"{row['citation_coverage']:.0%} | {row['elapsed_ms']} | {'✅' if row['passed'] else '❌'} |"
        )
    lines += ["", "## Harness 场景", "", "| ID | 场景 | 预期 | 实际 | 通过 |", "| --- | --- | --- | --- | --- |"]
    for row in out["harness"]:
        lines.append(f"| {row['id']} | {row['kind']} | {row['expect']} | {row['detail']} | {'✅' if row['passed'] else '❌'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
