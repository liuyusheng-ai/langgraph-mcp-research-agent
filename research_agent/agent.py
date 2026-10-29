"""Planner-Researcher-Writer LangGraph with a bounded tool harness."""

from __future__ import annotations

import hashlib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .checkpoint import SQLiteCheckpointStore
from .evaluation import evaluate_run
from .llm import DemoResearchModel
from .models import ToolCallRecord, ToolError, TransientToolError
from .tools import ResearchToolbox

INSUFFICIENT_REPORT = "资料不足：当前没有可核验来源，系统停止生成结论性报告。"


class ResearchState(TypedDict, total=False):
    run_id: str
    query: str
    plan: list[str]
    evidence: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    seen_calls: list[str]
    remaining_budget: int
    report: str
    status: str
    stopped_reason: str | None
    evaluation: dict[str, Any]


class ResearchAgent:
    def __init__(
        self,
        toolbox: ResearchToolbox | None = None,
        model: DemoResearchModel | None = None,
        checkpoints: SQLiteCheckpointStore | None = None,
        call_budget: int = 8,
        timeout_seconds: float = 5.0,
        max_retries: int = 1,
        results_per_call: int = 3,
    ) -> None:
        self.toolbox = toolbox or ResearchToolbox.default()
        self.model = model or DemoResearchModel()
        self.checkpoints = checkpoints or SQLiteCheckpointStore()
        self.call_budget = call_budget
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.results_per_call = results_per_call
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("planner", self._planner)
        graph.add_node("researcher", self._researcher)
        graph.add_node("writer", self._writer)
        graph.add_node("stop_insufficient", self._stop_insufficient)
        # A resumed run re-enters at the first node it had not finished.
        graph.add_conditional_edges(
            START,
            self._route_entry,
            {"planner": "planner", "researcher": "researcher", "writer": "writer"},
        )
        graph.add_edge("planner", "researcher")
        graph.add_conditional_edges(
            "researcher",
            self._route_after_research,
            {"writer": "writer", "stop": "stop_insufficient"},
        )
        graph.add_edge("writer", END)
        graph.add_edge("stop_insufficient", END)
        return graph.compile()

    def run(self, query: str, run_id: str | None = None, resume: bool = False) -> ResearchState:
        actual_run_id = run_id or str(uuid.uuid4())
        saved = self.checkpoints.load(actual_run_id) if resume else None
        if saved and saved.get("status") in {"completed", "insufficient_evidence"}:
            return saved
        state: ResearchState = saved or {
            "run_id": actual_run_id,
            "query": query.strip(),
            "evidence": [],
            "tool_calls": [],
            "seen_calls": [],
            "remaining_budget": self.call_budget,
            "status": "started",
            "stopped_reason": None,
        }
        result = self.graph.invoke(state)
        result["evaluation"] = evaluate_run(result)
        self.checkpoints.save(actual_run_id, result)
        return result

    def _route_entry(self, state: ResearchState) -> str:
        if state.get("status") == "researched":
            return "writer"
        if state.get("plan"):
            return "researcher"
        return "planner"

    def _route_after_research(self, state: ResearchState) -> str:
        return "writer" if state.get("evidence") else "stop"

    def _planner(self, state: ResearchState) -> ResearchState:
        update = {"plan": self.model.plan(state["query"]), "status": "planned"}
        self._save_partial(state, update)
        return update

    def _researcher(self, state: ResearchState) -> ResearchState:
        evidence = list(state.get("evidence", []))
        records = list(state.get("tool_calls", []))
        seen = set(state.get("seen_calls", []))
        budget = state.get("remaining_budget", self.call_budget)
        stopped_reason = state.get("stopped_reason")

        for sub_query in state.get("plan") or [state["query"]]:
            for tool in self.toolbox.tools:
                fingerprint = self._fingerprint(tool.name, sub_query)
                if fingerprint in seen:
                    # Same tool, same normalized query: a repeat cannot add evidence, so it is not run.
                    records.append(self._record(tool.name, sub_query, "loop_blocked", error="重复的工具调用"))
                    continue
                if not tool.available:
                    seen.add(fingerprint)
                    records.append(self._record(tool.name, sub_query, "not_configured", error="工具未配置"))
                    continue
                if budget <= 0:
                    records.append(self._record(tool.name, sub_query, "budget_exhausted", error="调用预算已用完"))
                    stopped_reason = "budget_exhausted"
                    break

                seen.add(fingerprint)
                budget -= 1
                started = time.perf_counter()
                results, status, attempts, error = self._call_with_retry(tool, sub_query)
                evidence.extend(item.to_dict() for item in results)
                records.append(
                    self._record(
                        tool.name,
                        sub_query,
                        status,
                        result_count=len(results),
                        elapsed_ms=int((time.perf_counter() - started) * 1000),
                        attempts=attempts,
                        error=error,
                    )
                )
                self._save_partial(
                    state,
                    {
                        "evidence": evidence,
                        "tool_calls": records,
                        "seen_calls": sorted(seen),
                        "remaining_budget": budget,
                        "status": "researching",
                    },
                )
            if stopped_reason == "budget_exhausted":
                break

        update = {
            "evidence": self._deduplicate_evidence(evidence),
            "tool_calls": records,
            "seen_calls": sorted(seen),
            "remaining_budget": budget,
            "status": "researched",
            "stopped_reason": stopped_reason,
        }
        self._save_partial(state, update)
        return update

    def _call_with_retry(self, tool, query: str) -> tuple[list, str, int, str | None]:
        """Retry only transient failures (timeouts, network); a permanent error is returned at once."""
        last_status, last_error = "failed", None
        for attempt in range(1, self.max_retries + 2):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(tool.search, query, self.results_per_call)
            try:
                results = future.result(timeout=self.timeout_seconds)
                return results, "success" if results else "empty", attempt, None
            except FutureTimeout:
                future.cancel()
                last_status, last_error = "timeout", f"工具调用超过 {self.timeout_seconds} 秒"
            except TransientToolError as exc:
                last_status, last_error = "failed", str(exc)
            except ToolError as exc:
                return [], "failed", attempt, str(exc)
            except Exception as exc:  # unknown errors are treated as permanent
                return [], "failed", attempt, f"{type(exc).__name__}: {exc}"
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        return [], last_status, self.max_retries + 1, last_error

    def _writer(self, state: ResearchState) -> ResearchState:
        update = {
            "report": self.model.write_report(state["query"], state.get("evidence", [])),
            "status": "completed",
        }
        self._save_partial(state, update)
        return update

    def _stop_insufficient(self, state: ResearchState) -> ResearchState:
        update = {
            "report": INSUFFICIENT_REPORT,
            "status": "insufficient_evidence",
            "stopped_reason": state.get("stopped_reason") or "no_evidence",
        }
        self._save_partial(state, update)
        return update

    @staticmethod
    def _fingerprint(tool_name: str, query: str) -> str:
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(f"{tool_name}:{normalized}".encode("utf-8")).hexdigest()

    @staticmethod
    def _record(tool: str, query: str, status: str, **fields: Any) -> dict[str, Any]:
        return ToolCallRecord(
            tool=tool,
            query=query,
            status=status,
            result_count=fields.get("result_count", 0),
            elapsed_ms=fields.get("elapsed_ms", 0),
            attempts=fields.get("attempts", 0),
            error=fields.get("error"),
        ).to_dict()

    @staticmethod
    def _deduplicate_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for item in evidence:
            source = item.get("source", "")
            if not source or source in seen_sources:
                continue
            seen_sources.add(source)
            unique.append(item)
        return unique

    def _save_partial(self, state: ResearchState, update: dict[str, Any]) -> None:
        snapshot = dict(state)
        snapshot.update(update)
        self.checkpoints.save(state["run_id"], snapshot)
