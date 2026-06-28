"""Categorize base↔FT disagreements by *mechanism* (validated by hand-reading).

The qualitative read (see analysis/ft_gains_analysis.md) established what kinds of
base failure FT fixes and what kinds of new error FT introduces. Here we tag every
disagreement by measurable, hand-validated signals so the per-model distribution
can be charted. Tags are mechanism-level (format vs content), which is exactly the
"worin lagen die Verbesserungen" question.

Correction buckets (base wrong → FT right):
  - code-style impl error : base answered by writing/executing Python (its
        pretraining default) and the program was wrong → FT switches to explicit
        arithmetic reasoning. [Qwen2's dominant mode]
  - non-termination / runaway : base solved-then-kept-going (echoed extra
        Problem/Solution blocks, multiple ####, or hit the length cap) so the
        extracted answer was wrong → FT emits one clean #### and stops.
        [Qwen2.5's dominant mode]
  - prose reasoning fix : base used clean prose, no code/runaway, but reasoned or
        computed wrong → FT's terse chain gets it right. [genuine content fix]

Regression buckets (base right → FT wrong) use the mirror signals:
  - non-termination / degenerate loop : FT looped / hit the cap.
  - lost-a-step compression : FT's chain is shorter than base's (fewer reasoning
        lines) → a step/term got dropped or collapsed. [the compression tax]
  - other content error : FT comparable length but wrong.

Writes analysis/ft_gains_categories_<model>.json and a combined
analysis/ft_gains_categories.json (default model = qwen2_1.5b for the chart).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def is_code(t: str) -> bool:
    return bool(re.search(r"```|^\s*def |\bimport \b|sympy|print\(", t))


def is_runaway(t: str) -> bool:
    # solved then echoed more exemplars, or emitted several final answers
    n_problem = len(re.findall(r"\bProblem:", t))
    n_final = len(re.findall(r"Final answer:|The answer:", t, re.I))
    n_marker = t.count("####")
    return n_problem >= 1 or n_final >= 2 or n_marker >= 2


def n_reasoning_lines(t: str) -> int:
    return sum(1 for ln in t.splitlines() if ln.strip())


def categorize(model: str) -> dict:
    src = REPO / "results_headroom" / model / "base_vs_ft_completions.json"
    d = json.loads(src.read_text())
    base = {e["index"]: e for e in d["base"]["examples"]}
    ft = {e["index"]: e for e in d["ft"]["examples"]}

    corrections: dict[str, int] = {
        "code-style impl error (→ explicit reasoning)": 0,
        "non-termination / runaway (→ clean ####)": 0,
        "prose reasoning/arithmetic fix": 0,
    }
    regressions: dict[str, int] = {
        "non-termination / degenerate loop": 0,
        "lost-a-step compression": 0,
        "other content error": 0,
    }
    for i, b in base.items():
        f = ft.get(i)
        if f is None:
            continue
        if not b["correct"] and f["correct"]:
            bt = b["completion"]
            if is_code(bt):
                corrections["code-style impl error (→ explicit reasoning)"] += 1
            elif b.get("truncated") or is_runaway(bt):
                corrections["non-termination / runaway (→ clean ####)"] += 1
            else:
                corrections["prose reasoning/arithmetic fix"] += 1
        elif b["correct"] and not f["correct"]:
            ftc = f["completion"]
            if f.get("truncated") or is_runaway(ftc):
                regressions["non-termination / degenerate loop"] += 1
            elif n_reasoning_lines(ftc) < n_reasoning_lines(b["completion"]):
                regressions["lost-a-step compression"] += 1
            else:
                regressions["other content error"] += 1

    out = {
        "model": model,
        "base_correct": d["base"]["correct"], "ft_correct": d["ft"]["correct"],
        "total": d["base"]["total"],
        "n_corrections": sum(corrections.values()),
        "n_regressions": sum(regressions.values()),
        "corrections": corrections, "regressions": regressions,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["qwen2_1.5b", "qwen25_1.5b", "qwen3_1.7b"])
    ap.add_argument("--chart-model", default="qwen2_1.5b",
                    help="Which model's breakdown the default categories.json holds.")
    args = ap.parse_args()
    outdir = REPO / "analysis"
    allcats = {}
    for m in args.models:
        c = categorize(m)
        allcats[m] = c
        (outdir / f"ft_gains_categories_{m}.json").write_text(json.dumps(c, indent=2))
        net = c["ft_correct"] - c["base_correct"]
        print(f"=== {m} ===  base {c['base_correct']}/{c['total']} → "
              f"ft {c['ft_correct']}/{c['total']}  (net {net:+d})")
        print("  corrections:", c["corrections"])
        print("  regressions:", c["regressions"])
    (outdir / "ft_gains_categories.json").write_text(
        json.dumps(allcats[args.chart_model], indent=2))
    (outdir / "ft_gains_categories_all.json").write_text(json.dumps(allcats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
