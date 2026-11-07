from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict[str, Any] = field(default_factory=dict)

    def args_text(self) -> str:
        return json.dumps(self.input, sort_keys=True)


@dataclass(frozen=True)
class AgentResult:
    """What an agent did in one turn: the text it wrote and the tool calls it asked for."""
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()

    @staticmethod
    def from_dict(d: dict) -> "AgentResult":
        return AgentResult(d.get("text", ""), tuple(ToolCall(t["name"], t.get("input", {})) for t in d.get("tool_calls", [])))


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    messages: list[dict]
    graders: list[dict]
    system: str = ""
    tools: list[dict] = field(default_factory=list)
    tags: tuple[str, ...] = ()
    fixtures: dict[str, dict] = field(default_factory=dict)   # {"pass": {...}, "fail": {...}} reference responses


@dataclass(frozen=True)
class Grade:
    grader: str
    passed: bool
    detail: str = ""


class AgentError(Exception):
    """The agent could not produce a result (network failure, API error)."""
