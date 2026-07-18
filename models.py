"""Shared data structures for tools, checkpoints and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

