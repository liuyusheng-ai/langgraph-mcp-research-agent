from pathlib import Path

import pytest

from research_agent import ResearchAgent
from research_agent.checkpoint import SQLiteCheckpointStore
from research_agent.evaluation import evaluate_run
from research_agent.llm import DemoResearchModel
from research_agent.models import Evidence, ToolError, TransientToolError
from research_agent.tools import LocalRAGTool, MCPResearchTool, ResearchToolbox, tokens


class ScriptedTool:
    """A tool whose behaviour per call is scripted: a list of results or exceptions."""

    def __init__(self, name: str, script: list, available: bool = True) -> None:
        self.name = name
        self.available = available
        self.script = list(script)
        self.calls = 0

    def search(self, query: str, limit: int = 3):
        self.calls += 1
        step = self.script.pop(0) if self.script else []
        if isinstance(step, BaseException):
            raise step
        if step == "sleep":
            import time
            time.sleep(1)
            return []
        return step


def evidence(source: str) -> Evidence:
    return Evidence(title=source, content=f"关于 {source} 的资料", source=source, tool="scripted")


def make_agent(tmp_path: Path, tools=None, **kwargs) -> ResearchAgent:
    toolbox = ResearchToolbox(tools) if tools is not None else ResearchToolbox.default()
    return ResearchAgent(
        toolbox=toolbox,
        checkpoints=SQLiteCheckpointStore(str(tmp_path / "checkpoints.db")),
        **kwargs,
    )


def test_generates_report_with_sources_and_evaluation(tmp_path: Path) -> None:
    result = make_agent(tmp_path).run("LangGraph RAG MCP Agent 工程实践")
    assert result["status"] == "completed"
    assert len(result["evidence"]) >= 2
    assert "## 参考来源" in result["report"]
    assert result["evaluation"]["task_completed"] is True
    assert result["evaluation"]["citation_coverage"] == 1.0


def test_researcher_runs_every_planned_sub_query(tmp_path: Path) -> None:
    tool = ScriptedTool("scripted", [[evidence("a")], [evidence("b")], [evidence("c")]])
    agent = make_agent(tmp_path, [tool], model=DemoResearchModel())
    result = agent.run("RAG 与 MCP 如何协作")
    assert result["plan"] == ["RAG 与 MCP 如何协作", "RAG", "MCP"]
    assert tool.calls == 3
    assert [item["source"] for item in result["evidence"]] == ["a", "b", "c"]


def test_call_budget_limits_tool_execution(tmp_path: Path) -> None:
    result = make_agent(tmp_path, call_budget=1).run("多智能体 Agent")
    executed = [call for call in result["tool_calls"] if call["status"] in {"success", "empty"}]
    assert len(executed) == 1
    assert result["remaining_budget"] == 0
    assert result["stopped_reason"] == "budget_exhausted"


def test_repeated_sub_query_is_loop_blocked(tmp_path: Path) -> None:
    class RepeatingPlanner(DemoResearchModel):
        def plan(self, query: str) -> list[str]:
            return [query, f"  {query.upper()} "]

    tool = ScriptedTool("scripted", [[evidence("a")], [evidence("b")]])
    result = make_agent(tmp_path, [tool], model=RepeatingPlanner()).run("mcp agent")
    assert tool.calls == 1
    assert [call["status"] for call in result["tool_calls"]] == ["success", "loop_blocked"]


def test_transient_error_is_retried_then_succeeds(tmp_path: Path) -> None:
    tool = ScriptedTool("flaky", [TransientToolError("503"), [evidence("a")]])
    result = make_agent(tmp_path, [tool], max_retries=1).run("未知主题")
    call = result["tool_calls"][0]
    assert (call["status"], call["attempts"]) == ("success", 2)


def test_permanent_error_is_not_retried(tmp_path: Path) -> None:
    tool = ScriptedTool("broken", [ToolError("参数错误"), [evidence("a")]])
    result = make_agent(tmp_path, [tool], max_retries=3).run("未知主题")
    call = result["tool_calls"][0]
    assert (call["status"], call["attempts"]) == ("failed", 1)
    assert result["status"] == "insufficient_evidence"


def test_slow_tool_times_out_and_is_retried(tmp_path: Path) -> None:
    tool = ScriptedTool("slow", ["sleep", "sleep"])
    result = make_agent(tmp_path, [tool], timeout_seconds=0.05, max_retries=1).run("未知主题")
    call = result["tool_calls"][0]
    assert (call["status"], call["attempts"]) == ("timeout", 2)


def test_unconfigured_mcp_is_not_counted_as_success(tmp_path: Path) -> None:
    result = make_agent(tmp_path, [LocalRAGTool(), MCPResearchTool()]).run("RAG 证据")
    statuses = {call["tool"]: call["status"] for call in result["tool_calls"]}
    assert statuses["mcp"] == "not_configured"
    assert result["evaluation"]["tool_call_count"] == sum(
        call["status"] == "success" for call in result["tool_calls"]
    )


def test_empty_results_are_not_effective_results(tmp_path: Path) -> None:
    tool = ScriptedTool("empty", [[], [], []])
    result = make_agent(tmp_path, [tool]).run("未知主题")
    assert result["tool_calls"][0]["status"] == "empty"
    assert result["evaluation"]["tool_success_rate"] == 1.0
    assert result["evaluation"]["effective_result_rate"] == 0.0


def test_no_source_gate_blocks_conclusion(tmp_path: Path) -> None:
    result = make_agent(tmp_path, []).run("完全未知主题")
    assert result["status"] == "insufficient_evidence"
    assert result["stopped_reason"] == "no_evidence"
    assert "资料不足" in result["report"]
    assert result["evaluation"]["task_completed"] is False


def test_checkpoint_can_restore_completed_run(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    first = agent.run("MCP 工具安全", run_id="demo-run")
    restored = agent.run("不会覆盖原问题", run_id="demo-run", resume=True)
    assert restored["report"] == first["report"]
    assert restored["query"] == first["query"]


def test_interrupted_run_resumes_without_repeating_calls(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(str(tmp_path / "checkpoints.db"))
    first_tool = ScriptedTool("a", [[evidence("a")]])
    crashing = ScriptedTool("b", [KeyboardInterrupt()])
    agent = ResearchAgent(toolbox=ResearchToolbox([first_tool, crashing]), checkpoints=store)
    with pytest.raises(KeyboardInterrupt):
        agent.run("未知主题", run_id="crash")
    assert store.load("crash")["status"] == "researching"

    resumed_tool_a = ScriptedTool("a", [[evidence("again")]])
    resumed_tool_b = ScriptedTool("b", [[evidence("b")]])
    agent = ResearchAgent(toolbox=ResearchToolbox([resumed_tool_a, resumed_tool_b]), checkpoints=store)
    result = agent.run("未知主题", run_id="crash", resume=True)
    assert resumed_tool_a.calls == 0
    assert resumed_tool_b.calls == 1
    assert [item["source"] for item in result["evidence"]] == ["a", "b"]
    assert result["status"] == "completed"


def test_evaluation_handles_failed_and_blocked_calls() -> None:
    metrics = evaluate_run(
        {
            "status": "completed",
            "evidence": [{"source": "a"}],
            "tool_calls": [
                {"status": "success"},
                {"status": "failed"},
                {"status": "loop_blocked"},
                {"status": "not_configured"},
            ],
            "report": "结论 [1]",
        }
    )
    assert metrics["tool_success_rate"] == 0.5
    assert metrics["harness_blocked_count"] == 2
    assert metrics["citation_coverage"] == 1.0


def test_report_citing_unknown_source_is_not_completed() -> None:
    metrics = evaluate_run(
        {"status": "completed", "evidence": [{"source": "a"}], "tool_calls": [], "report": "结论 [1][3]"}
    )
    assert metrics["unknown_citations"] == [3]
    assert metrics["task_completed"] is False


def test_chinese_tokens_match_partial_phrases() -> None:
    assert "证据" in tokens("RAG 证据链")
    assert tokens("完全未知主题").isdisjoint(tokens("RAG 证据链"))
