"""Plot the Stage-1 epochs-diagnostic learning curve (dev acc + train loss)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/instruct/diag_epochs_v_proj.json"))
    ap.add_argument("--out", type=Path, default=Path("results/instruct/diag_epochs_v_proj.png"))
    args = ap.parse_args()

    d = json.loads(args.run.read_text())
    epochs = [e["epoch"] for e in d["dev_curve"]]
    dev = [e["dev_accuracy"] for e in d["dev_curve"]]
    n = d["dev_curve"][0]["dev_total"]
    se = [(_a * (1 - _a) / n) ** 0.5 for _a in dev]

    # macro-step -> epoch axis for the loss curve
    steps_per_epoch = d["steps_done"] / d["train"]["num_epochs"]
    loss_x = [s / steps_per_epoch for s, _ in d["loss_curve"]]
    loss_y = [v for _, v in d["loss_curve"]]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(epochs, dev, yerr=se, marker="o", color="tab:blue",
                 capsize=4, label="dev accuracy (+/-1 SE)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("GSM8K dev accuracy", color="tab:blue")
    ax1.set_xticks(epochs)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    for x, y in zip(epochs, dev):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9, color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(loss_x, loss_y, color="tab:red", alpha=0.5, label="train loss")
    ax2.set_ylabel("train loss", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    cfg = d["train"]
    plt.title(f"Stage 1 epochs diagnostic — qv r=8, {cfg['lr_schedule']} LR "
              f"{cfg['learning_rate']:g} (n={n} dev)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
