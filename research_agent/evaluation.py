"""Agentic evaluation across task, tool and citation dimensions."""

from __future__ import annotations

import re
from typing import Any

from .models import EXECUTED_STATUSES


def evaluate_run(state: dict[str, Any]) -> dict[str, Any]:
    evidence = state.get("evidence", [])
    tool_calls = state.get("tool_calls", [])
    report = state.get("report", "")

    # Calls the harness stopped (loop, budget, unconfigured tool) never reached a tool,
    # so they count neither for nor against the tool success rate.
    executed = [call for call in tool_calls if call.get("status") in EXECUTED_STATUSES]
    answered = sum(call.get("status") in {"success", "empty"} for call in executed)
    with_results = sum(call.get("status") == "success" for call in executed)

    cited = {int(match) for match in re.findall(r"\[(\d+)\]", report)}
    expected = set(range(1, len(evidence) + 1))
    citation_coverage = len(cited & expected) / len(expected) if expected else 0.0
    unknown_citations = sorted(cited - expected)

    completed = bool(state.get("status") == "completed" and evidence and not unknown_citations)
    return {
        "task_completed": completed,
        "tool_success_rate": round(answered / len(executed), 4) if executed else 0.0,
        "effective_result_rate": round(with_results / len(executed), 4) if executed else 0.0,
        "citation_coverage": round(citation_coverage, 4),
        "unknown_citations": unknown_citations,
        "evidence_count": len(evidence),
        "tool_call_count": len(executed),
        "harness_blocked_count": len(tool_calls) - len(executed),
        "stopped_reason": state.get("stopped_reason"),
    }
