"""Agentic evaluation across task, tool and citation dimensions."""

from __future__ import annotations

import re
from typing import Any


def evaluate_run(state: dict[str, Any]) -> dict[str, Any]:
    evidence = state.get("evidence", [])
    tool_calls = state.get("tool_calls", [])
    report = state.get("report", "")

    successful = sum(call.get("status") == "success" for call in tool_calls)
    tool_success_rate = successful / len(tool_calls) if tool_calls else 0.0

    cited_numbers = {
        int(match) for match in re.findall(r"\[(\d+)\]", report)
    }
    expected = set(range(1, len(evidence) + 1))
    citation_coverage = (
        len(cited_numbers & expected) / len(expected) if expected else 0.0
    )

    completed = bool(
        state.get("status") == "completed"
        and evidence
        and not report.startswith("资料不足")
    )
    return {
        "task_completed": completed,
        "tool_success_rate": round(tool_success_rate, 4),
        "citation_coverage": round(citation_coverage, 4),
        "evidence_count": len(evidence),
        "tool_call_count": len(tool_calls),
    }
