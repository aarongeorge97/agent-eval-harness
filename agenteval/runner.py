from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .agents import Agent, reference
from .graders import grade_all
from .stats import wilson_interval
from .types import AgentError, AgentResult, Grade, Scenario

REQUIRED = ("id", "description", "messages", "graders")


def load_scenarios(path: str | Path) -> list[Scenario]:
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("*.json"))
    if not files:
        raise ValueError(f"no scenario files found in {p}")
    out, seen = [], set()
    for f in files:
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{f.name}: invalid JSON ({e.msg} at line {e.lineno})") from None
        missing = [k for k in REQUIRED if k not in d]
        if missing:
            raise ValueError(f"{f.name}: missing {', '.join(missing)}")
        if d["id"] in seen:
            raise ValueError(f"{f.name}: duplicate scenario id {d['id']!r}")
        seen.add(d["id"])
        grade_all(d["graders"], AgentResult())   # fail now on a misspelled grader, not mid-run
        out.append(Scenario(d["id"], d["description"], d["messages"], d["graders"], d.get("system", ""),
                            d.get("tools", []), tuple(d.get("tags", ())), d.get("fixtures", {})))
    return sorted(out, key=lambda s: s.id)


@dataclass
class Sample:
    grades: list[Grade] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(g.passed for g in self.grades)


@dataclass
class ScenarioReport:
    scenario: Scenario
    samples: list[Sample]

    @property
    def errors(self) -> int:
        return sum(1 for s in self.samples if s.error)

    @property
    def n(self) -> int:           # errored samples say nothing about the rule, so they are not counted
        return len(self.samples) - self.errors

    @property
    def passes(self) -> int:
        return sum(1 for s in self.samples if s.passed)

    @property
    def rate(self) -> float | None:
        return self.passes / self.n if self.n else None

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.passes, self.n)

    def failures(self) -> dict[str, int]:
        """How often each distinct failure happened, e.g. {'no_tool_call(send_.*): called send_email': 3}."""
        out: dict[str, int] = {}
        for s in self.samples:
            for g in s.grades:
                if not g.passed:
                    key = f"{g.grader}: {g.detail}" if g.detail else g.grader
                    out[key] = out.get(key, 0) + 1
        return out


@dataclass
class Report:
    scenarios: list[ScenarioReport]

    def failing(self, min_pass: float = 1.0) -> list[ScenarioReport]:
        return [r for r in self.scenarios if r.n == 0 or (r.rate or 0) < min_pass]


def run(scenarios: list[Scenario], agent: Agent, n: int = 1) -> Report:
    reports = []
    for s in scenarios:
        samples = []
        for _ in range(n):
            try:
                samples.append(Sample(grade_all(s.graders, agent(s))))
            except AgentError as e:
                samples.append(Sample(error=str(e)))
        reports.append(ScenarioReport(s, samples))
    return Report(reports)


def selfcheck(scenarios: list[Scenario]) -> list[str]:
    """Do the graders separate a good response from a bad one? Returns a list of problems (empty = sound).

    A grader that passes everything is worse than no grader: it manufactures confidence.
    """
    problems = []
    for s in scenarios:
        for kind in ("pass", "fail"):
            if kind not in s.fixtures:
                problems.append(f"{s.id}: missing '{kind}' fixture")
        if problems and problems[-1].startswith(f"{s.id}:"):
            continue
        good = grade_all(s.graders, reference("pass")(s))
        bad = grade_all(s.graders, reference("fail")(s))
        for g in good:
            if not g.passed:
                problems.append(f"{s.id}: reference PASS response fails {g.grader} ({g.detail})")
        if all(g.passed for g in bad):
            problems.append(f"{s.id}: reference FAIL response passes every grader, so this scenario cannot detect the failure")
    return problems
