from __future__ import annotations

import json

from .runner import Report


def to_markdown(report: Report, label: str = "") -> str:
    lines = [f"## Agent eval{f': {label}' if label else ''}", "",
             "| Scenario | Pass | Rate | 95% CI | Errors | Why it failed |", "|---|---|---|---|---|---|"]
    for r in report.scenarios:
        lo, hi = r.ci
        rate = "n/a" if r.rate is None else f"{r.rate:.0%}"
        why = "; ".join(f"{k} (x{v})" for k, v in r.failures().items()) or ""
        lines.append(f"| `{r.scenario.id}` | {r.passes}/{r.n} | {rate} | {lo:.0%}–{hi:.0%} | {r.errors} | {why} |")
    total_n = sum(r.n for r in report.scenarios)
    total_p = sum(r.passes for r in report.scenarios)
    lines += ["", f"**{total_p}/{total_n}** samples passed across {len(report.scenarios)} scenarios.",
              "", "A small sample cannot show a rule is always followed: 5/5 passes is consistent with a true rate as low as ~57%."]
    return "\n".join(lines)


def to_json(report: Report) -> str:
    out = []
    for r in report.scenarios:
        lo, hi = r.ci
        out.append({"id": r.scenario.id, "tags": list(r.scenario.tags), "passes": r.passes, "n": r.n, "errors": r.errors,
                    "rate": r.rate, "ci95": [round(lo, 4), round(hi, 4)], "failures": r.failures()})
    return json.dumps({"scenarios": out}, indent=2)
