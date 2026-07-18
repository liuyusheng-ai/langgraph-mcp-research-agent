"""Planner-Researcher-Writer LangGraph with bounded tool execution."""

from __future__ import annotations

import hashlib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from checkpoint import SQLiteCheckpointStore
from evaluation import evaluate_run
from llm import DemoResearchModel
from tools import ResearchToolbox


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
    error: str | None
    evaluation: dict[str, Any]


class ResearchAgent:
    def __init__(
        self,
        toolbox: ResearchToolbox | None = None,
        model: DemoResearchModel | None = None,
        checkpoints: SQLiteCheckpointStore | None = None,
        call_budget: int = 6,
        timeout_seconds: float = 5.0,
        max_retries: int = 1,
    ) -> None:
        self.toolbox = toolbox or ResearchToolbox.default()
        self.model = model or DemoResearchModel()
        self.checkpoints = checkpoints or SQLiteCheckpointStore()
        self.call_budget = call_budget
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("planner", self._planner)
        graph.add_node("researcher", self._researcher)
        graph.add_node("writer", self._writer)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "researcher")
        graph.add_edge("researcher", "writer")
        graph.add_edge("writer", END)
        return graph.compile()

    def run(
        self, query: str, run_id: str | None = None, resume: bool = False
    ) -> ResearchState:
        actual_run_id = run_id or str(uuid.uuid4())
        if resume:
            saved = self.checkpoints.load(actual_run_id)
            if saved and saved.get("status") == "completed":
                return saved

        initial: ResearchState = {
            "run_id": actual_run_id,
            "query": query.strip(),
            "evidence": [],
            "tool_calls": [],
            "seen_calls": [],
            "remaining_budget": self.call_budget,
            "status": "started",
        }
        result = self.graph.invoke(initial)
        result["evaluation"] = evaluate_run(result)
        self.checkpoints.save(actual_run_id, result)
        return result

    def _planner(self, state: ResearchState) -> ResearchState:
        plan = self.model.plan(state["query"])
        updated = {"plan": plan, "status": "planned"}
        self._save_partial(state, updated)
        return updated

    def _researcher(self, state: ResearchState) -> ResearchState:
        evidence = list(state.get("evidence", []))
        records = list(state.get("tool_calls", []))
        seen = set(state.get("seen_calls", []))
        budget = state.get("remaining_budget", self.call_budget)

        for tool in self.toolbox.tools:
            if budget <= 0:
                break
            fingerprint = hashlib.sha256(
                f"{tool.name}:{state['query']}".encode("utf-8")
            ).hexdigest()
            if fingerprint in seen:
                records.append(
                    {
                        "tool": tool.name,
                        "query": state["query"],
                        "status": "loop_blocked",
                        "result_count": 0,
                        "elapsed_ms": 0,
                        "error": "检测到重复工具调用",
                    }
                )
                continue

            seen.add(fingerprint)
            budget -= 1
            started = time.perf_counter()
            results, status, error = self._call_with_retry(tool, state["query"])
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            evidence.extend(item.to_dict() for item in results)
            records.append(
                {
                    "tool": tool.name,
                    "query": state["query"],
                    "status": status,
                    "result_count": len(results),
                    "elapsed_ms": elapsed_ms,
                    "error": error,
                }
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

        return {
            "evidence": evidence,
            "tool_calls": records,
            "seen_calls": sorted(seen),
            "remaining_budget": budget,
            "status": "researched",
        }

    def _call_with_retry(self, tool, query: str):
        last_error: str | None = None
        for _ in range(self.max_retries + 1):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(tool.search, query, 3)
            try:
                return future.result(timeout=self.timeout_seconds), "success", None
            except FutureTimeout:
                future.cancel()
                last_error = f"工具调用超过 {self.timeout_seconds} 秒"
            except Exception as exc:
                last_error = str(exc)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        return [], "failed", last_error

    def _writer(self, state: ResearchState) -> ResearchState:
        evidence = self._deduplicate_evidence(state.get("evidence", []))
        report = self.model.write_report(state["query"], evidence)
        status = "completed" if evidence else "insufficient_evidence"
        updated = {
            "evidence": evidence,
            "report": report,
            "status": status,
        }
        self._save_partial(state, updated)
        return updated

    def _deduplicate_evidence(
        self, evidence: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for item in evidence:
            source = item.get("source", "")
            if not source or source in seen_sources:
                continue
            seen_sources.add(source)
            unique.append(item)
        return unique

    def _save_partial(
        self, state: ResearchState, update: dict[str, Any]
    ) -> None:
        snapshot = dict(state)
        snapshot.update(update)
        self.checkpoints.save(state["run_id"], snapshot)

