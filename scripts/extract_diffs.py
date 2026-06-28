"""Extract base-vs-FT disagreements into readable text for hand categorization.

Reads results_headroom/<model>/base_vs_ft_completions.json (saved with both base
and ft per-example completions) and writes two text dumps under analysis/_work/:
  <model>_corrections.txt  — base wrong, FT correct
  <model>_regressions.txt  — base correct, FT wrong
Each entry shows the question, gold answer, and both completions so the dominant
error/fix category can be read off by hand.

Usage:
    python scripts/extract_diffs.py --model qwen2_1.5b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-chars", type=int, default=1400,
                    help="Truncate each completion in the dump for readability.")
    args = ap.parse_args()

    src = REPO / "results_headroom" / args.model / "base_vs_ft_completions.json"
    d = json.loads(src.read_text())
    base = {e["index"]: e for e in d["base"]["examples"]}
    ft = {e["index"]: e for e in d["ft"]["examples"]}

    corrections, regressions = [], []
    for i in sorted(base):
        b, f = base[i], ft.get(i)
        if f is None:
            continue
        if not b["correct"] and f["correct"]:
            corrections.append(i)
        elif b["correct"] and not f["correct"]:
            regressions.append(i)

    outdir = REPO / "analysis" / "_work"
    outdir.mkdir(parents=True, exist_ok=True)

    def dump(indices: list[int], path: Path, title: str) -> None:
        lines = [f"# {title}: {len(indices)} examples (model={args.model})", ""]
        for i in indices:
            b, f = base[i], ft[i]
            lines += [
                f"================= idx {i}  gold={b['gt']} =================",
                f"[base pred={b['predicted']} trunc={b.get('truncated')}]",
                b["completion"][:args.max_chars].rstrip(),
                "",
                f"[ft   pred={f['predicted']} trunc={f.get('truncated')}]",
                f["completion"][:args.max_chars].rstrip(),
                "", "",
            ]
        path.write_text("\n".join(lines))
        print(f"wrote {path} ({len(indices)} entries)")

    dump(corrections, outdir / f"{args.model}_corrections.txt",
         "Corrections (base wrong -> FT correct)")
    dump(regressions, outdir / f"{args.model}_regressions.txt",
         "Regressions (base correct -> FT wrong)")

    nb = d["base"]["correct"]
    nf = d["ft"]["correct"]
    tot = d["base"]["total"]
    print(f"\n{args.model}: base {nb}/{tot}  ft {nf}/{tot}  net {nf-nb:+d}  "
          f"| {len(corrections)} corrections, {len(regressions)} regressions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
