"""
  python -m agenteval check                          # are the graders sound? (no model needed)
  python -m agenteval list
  python -m agenteval run --agent reference:pass     # replay each scenario's reference response
  python -m agenteval run --agent anthropic:MODEL -n 5 --json out.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agents import AnthropicAgent, reference
from .report import to_json, to_markdown
from .runner import load_scenarios, run, selfcheck
from .types import AgentError

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def build_agent(spec: str):
    kind, _, arg = spec.partition(":")
    if kind == "reference" and arg in ("pass", "fail"):
        return reference(arg)
    if kind == "anthropic" and arg:
        return AnthropicAgent(arg)
    raise SystemExit(f"unknown agent {spec!r}; use reference:pass, reference:fail or anthropic:<model>")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agenteval")
    ap.add_argument("--scenarios", type=Path, default=DEFAULT_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--agent", required=True)
    r.add_argument("-n", type=int, default=1, help="samples per scenario")
    r.add_argument("--min-pass", type=float, default=1.0, help="exit 1 if any scenario's pass rate is below this")
    r.add_argument("--tag")
    r.add_argument("--json", type=Path, help="also write machine-readable results here")
    args = ap.parse_args(argv)

    try:
        scenarios = load_scenarios(args.scenarios)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.cmd == "list":
        for s in scenarios:
            print(f"{s.id:38} {s.description}")
        return 0
    if args.cmd == "check":
        problems = selfcheck(scenarios)
        for p in problems:
            print("PROBLEM:", p)
        print(f"{'FAIL' if problems else 'OK'}: {len(scenarios)} scenarios, {len(problems)} problem(s)")
        return 1 if problems else 0

    if args.tag:
        scenarios = [s for s in scenarios if args.tag in s.tags]
    try:
        report = run(scenarios, build_agent(args.agent), args.n)
    except AgentError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(to_markdown(report, args.agent))
    if args.json:
        args.json.write_text(to_json(report))
    return 1 if report.failing(args.min_pass) else 0


if __name__ == "__main__":
    sys.exit(main())
