"""Graders check one thing about what an agent did. They are deterministic: no model
is used to judge another model here, so a grade can always be explained and re-run."""
from __future__ import annotations

import re
from typing import Callable

from .types import AgentResult, Grade

GraderFn = Callable[[AgentResult], Grade]
_FLAGS = {"i": re.I, "m": re.M, "s": re.S}


def _rx(pattern: str, flags: str = "") -> re.Pattern:
    f = 0
    for ch in flags:
        f |= _FLAGS[ch]
    return re.compile(pattern, f)


def _clip(s: str, n: int = 80) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


def _mask(s: str) -> str:
    """Never copy a matched value into a report: a rule violation is often a leaked secret or PII,
    and reports end up in CI logs. Keep enough to locate it, not enough to reuse it."""
    s = " ".join(s.split())
    return s if len(s) <= 3 else f"{s[:3]}…[{len(s)} chars]"


def no_tool_call(name: str) -> GraderFn:
    rx = _rx(name)
    label = f"no_tool_call({name})"

    def grade(r: AgentResult) -> Grade:
        hit = [t for t in r.tool_calls if rx.fullmatch(t.name)]
        return Grade(label, not hit, f"called {hit[0].name}" if hit else "")
    return grade


def tool_called(name: str, args_match: dict | None = None) -> GraderFn:
    rx = _rx(name)
    label = f"tool_called({name})"
    arg_rx = {k: _rx(v) for k, v in (args_match or {}).items()}

    def grade(r: AgentResult) -> Grade:
        for t in r.tool_calls:
            if rx.fullmatch(t.name) and all(k in t.input and arg_rx[k].search(str(t.input[k])) for k in arg_rx):
                return Grade(label, True)
        return Grade(label, False, "no matching call; saw: " + (", ".join(t.name for t in r.tool_calls) or "no tool calls"))
    return grade


def tool_arg_not_matches(pattern: str, name: str = ".*", flags: str = "") -> GraderFn:
    rx, nrx = _rx(pattern, flags), _rx(name)
    label = f"tool_arg_not_matches({pattern})"

    def grade(r: AgentResult) -> Grade:
        for t in r.tool_calls:
            m = nrx.fullmatch(t.name) and rx.search(t.args_text())
            if m:
                return Grade(label, False, f"{t.name} args matched {_mask(m.group(0))!r}")
        return Grade(label, True)
    return grade


def output_matches(pattern: str, flags: str = "i") -> GraderFn:
    rx = _rx(pattern, flags)
    label = f"output_matches({_clip(pattern, 40)})"
    return lambda r: Grade(label, bool(rx.search(r.text)), "" if rx.search(r.text) else "pattern not found in output")


def output_not_matches(pattern: str, flags: str = "i") -> GraderFn:
    rx = _rx(pattern, flags)
    label = f"output_not_matches({_clip(pattern, 40)})"

    def grade(r: AgentResult) -> Grade:
        m = rx.search(r.text)
        return Grade(label, not m, f"matched {_mask(m.group(0))!r}" if m else "")
    return grade


def max_tool_calls(n: int) -> GraderFn:
    label = f"max_tool_calls({n})"
    return lambda r: Grade(label, len(r.tool_calls) <= n, f"{len(r.tool_calls)} calls" if len(r.tool_calls) > n else "")


REGISTRY = {
    "no_tool_call": no_tool_call, "tool_called": tool_called, "tool_arg_not_matches": tool_arg_not_matches,
    "output_matches": output_matches, "output_not_matches": output_not_matches, "max_tool_calls": max_tool_calls,
}


def build(spec: dict) -> GraderFn:
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind not in REGISTRY:
        raise ValueError(f"unknown grader type {kind!r}; known: {', '.join(sorted(REGISTRY))}")
    try:
        return REGISTRY[kind](**spec)
    except TypeError as e:
        raise ValueError(f"bad arguments for grader {kind!r}: {e}") from None


def grade_all(specs: list[dict], result: AgentResult) -> list[Grade]:
    return [build(s)(result) for s in specs]
