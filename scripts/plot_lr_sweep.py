"""Plot the Stage-2 LR-sweep dev accuracy (qv r=8, E*=3, cosine)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

LRS = ["5e-5", "1e-4", "2e-4", "4e-4"]
RESULTS = Path("results_v2")


def main() -> int:
    xs, ys, ses = [], [], []
    for lr in LRS:
        d = json.loads((RESULTS / f"eval_lr_{lr}.json").read_text())
        acc = d["ft"]["accuracy"]
        n = d["num_problems"]
        xs.append(float(lr))
        ys.append(acc)
        ses.append((acc * (1 - acc) / n) ** 0.5)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, ys, yerr=ses, marker="o", capsize=4, color="tab:green")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(LRS)
    ax.set_xlabel("learning rate (cosine, 3 epochs)")
    ax.set_ylabel("GSM8K dev accuracy (n=1000)")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    best = max(zip(ys, LRS))
    ax.set_title(f"Stage 2 LR sweep — qv r=8, E*=3 (best: LR={best[1]}, {best[0]:.3f})")
    fig.tight_layout()
    out = RESULTS / "lr_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
