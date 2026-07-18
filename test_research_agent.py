from pathlib import Path

from agent import ResearchAgent
from checkpoint import SQLiteCheckpointStore
from evaluation import evaluate_run
from tools import ResearchToolbox


def make_agent(tmp_path: Path, budget: int = 6) -> ResearchAgent:
    return ResearchAgent(
        checkpoints=SQLiteCheckpointStore(str(tmp_path / "checkpoints.db")),
        call_budget=budget,
    )


def test_generates_report_with_sources_and_evaluation(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    result = agent.run("LangGraph RAG MCP Agent 工程实践")
    assert result["status"] == "completed"
    assert len(result["evidence"]) >= 2
    assert "## 参考来源" in result["report"]
    assert result["evaluation"]["task_completed"] is True
    assert result["evaluation"]["citation_coverage"] == 1.0


def test_call_budget_limits_tool_execution(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, budget=1)
    result = agent.run("多智能体 Agent")
    assert len(result["tool_calls"]) == 1
    assert result["remaining_budget"] == 0


def test_no_source_gate_blocks_conclusion(tmp_path: Path) -> None:
    empty_toolbox = ResearchToolbox([])
    agent = ResearchAgent(
        toolbox=empty_toolbox,
        checkpoints=SQLiteCheckpointStore(str(tmp_path / "empty.db")),
    )
    result = agent.run("完全未知主题")
    assert result["status"] == "insufficient_evidence"
    assert "资料不足" in result["report"]
    assert result["evaluation"]["task_completed"] is False


def test_checkpoint_can_restore_completed_run(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    first = agent.run("MCP 工具安全", run_id="demo-run")
    restored = agent.run("不会覆盖原问题", run_id="demo-run", resume=True)
    assert restored["report"] == first["report"]
    assert restored["query"] == first["query"]


def test_evaluation_handles_failed_tools() -> None:
    metrics = evaluate_run(
        {
            "status": "completed",
            "evidence": [{"source": "a"}],
            "tool_calls": [
                {"status": "success"},
                {"status": "failed"},
            ],
            "report": "结论 [1]",
        }
    )
    assert metrics["tool_success_rate"] == 0.5
    assert metrics["citation_coverage"] == 1.0

