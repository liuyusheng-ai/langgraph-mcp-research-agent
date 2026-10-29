"""Shared data structures for tools, checkpoints and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class ToolError(Exception):
    """A tool failed in a way that retrying will not fix (bad arguments, server error)."""


class TransientToolError(ToolError):
    """A tool failed in a way that may succeed on retry (network, timeout)."""


# Executed calls reached a tool; the harness statuses were stopped before any call was made.
EXECUTED_STATUSES = {"success", "empty", "failed", "timeout"}
HARNESS_STATUSES = {"loop_blocked", "budget_exhausted", "not_configured"}


@dataclass(frozen=True)
class Evidence:
    title: str
    content: str
    source: str
    tool: str
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCallRecord:
    tool: str
    query: str
    status: str
    result_count: int
    elapsed_ms: int
    attempts: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
