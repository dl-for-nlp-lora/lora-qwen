"""Plot intra-epoch step diagnostic (dev acc vs optimizer step / epoch fraction)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/instruct/diag_steps_v_proj.json"))
    ap.add_argument("--out", type=Path, default=Path("results/instruct/diag_steps_v_proj.png"))
    args = ap.parse_args()

    d = json.loads(args.run.read_text())
    curve = d.get("step_curve") or []
    if not curve:
        raise SystemExit(f"No step_curve in {args.run}")

    steps_done = d["steps_done"]
    xs = [p["step"] for p in curve]
    dev = [p["dev_accuracy"] for p in curve]
    n = curve[0]["dev_total"]
    se = [(_a * (1 - _a) / n) ** 0.5 for _a in dev]
    fracs = [s / steps_done if steps_done else 0 for s in xs]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.errorbar(xs, dev, yerr=se, marker="o", capsize=4, color="tab:blue",
                   label="dev accuracy (+/-1 SE)")
    ax1.set_xlabel("optimizer step (macro, 1 epoch)")
    ax1.set_ylabel("GSM8K dev accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    best_i = max(range(len(dev)), key=lambda i: dev[i])
    ax1.annotate(
        f"peak {dev[best_i]:.3f} @ step {xs[best_i]}",
        (xs[best_i], dev[best_i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
        color="tab:blue",
    )

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    tick_steps = [s for s in xs if s % 100 == 0 or s == 0 or s == steps_done]
    ax2.set_xticks(tick_steps)
    ax2.set_xticklabels([f"{s / steps_done:.0%}" if steps_done else "0" for s in tick_steps])
    ax2.set_xlabel("fraction of 1 epoch")

    cfg = d["train"]
    ax1.set_title(
        f"Instruct step diagnostic — v_proj, constant LR {cfg['learning_rate']:g}, "
        f"eval every {d.get('eval_every_steps', '?')} steps (n={n} dev)"
    )
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
