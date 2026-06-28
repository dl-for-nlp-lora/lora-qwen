"""Plot instruct-cohort results: E1 target-set bars + E2 rank curve."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R = Path("results/instruct")
SETS = ["q_proj", "v_proj", "qv_proj", "attention", "all_linear"]
RANKS = [1, 2, 4, 8, 16, 32, 64]


def _se(acc: float, n: int) -> float:
    return math.sqrt(acc * (1 - acc) / n) if n else 0.0


def _load_ft_test(path: Path) -> tuple[float, int, int] | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    ft = d.get("ft")
    if not ft:
        return None
    return ft["accuracy"], ft["correct"], d.get("num_problems", ft["total"])


def plot_e1(out: Path) -> None:
    base_path = R / "base_instruct_test.json"
    base_acc = json.loads(base_path.read_text())["base"]["accuracy"]
    n = json.loads(base_path.read_text())["num_problems"]

    accs, ses = [], []
    for s in SETS:
        t = _load_ft_test(R / f"e1_{s}_test.json")
        if t:
            accs.append(t[0])
            ses.append(_se(t[0], t[2]))
        else:
            accs.append(float("nan"))
            ses.append(0.0)

    x = np.arange(len(SETS))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, accs, yerr=ses, capsize=4, color="tab:blue", label="FT instruct (test)")
    ax.axhline(base_acc, color="tab:orange", ls="--", lw=1.5,
               label=f"base instruct ({base_acc:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(SETS)
    ax.set_ylabel(f"GSM8K-test accuracy (n={n})")
    ax.set_ylim(0.45, 0.75)
    ax.set_title("Instruct cohort E1 — target sets @ ~4.4M params "
                 "(train instruct / eval instruct, E*=3, LR=2e-4)")
    for bar, y in zip(bars, accs):
        if not math.isnan(y):
            ax.annotate(
                f"{y:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=9,
            )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def plot_rank(out: Path) -> None:
    base_acc = json.loads((R / "base_instruct_test.json").read_text())["base"]["accuracy"]
    xs, ys, ses = [], [], []
    for r in RANKS:
        t = _load_ft_test(R / f"rank_q_r{r}_test.json")
        if not t:
            continue
        xs.append(r)
        ys.append(t[0])
        ses.append(_se(t[0], t[2]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, ys, yerr=ses, marker="o", capsize=4, color="tab:green")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r) for r in xs])
    ax.axhline(base_acc, color="tab:orange", ls="--", lw=1.5,
               label=f"base instruct ({base_acc:.3f})")
    ax.set_xlabel("LoRA rank (q_proj only)")
    ax.set_ylabel("GSM8K-test accuracy (FT, eval instruct)")
    ax.set_title("Instruct cohort E2 — rank sweep on q_proj "
                 "(train instruct / eval instruct, E*=3, LR=2e-4)")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def plot_vrank(out: Path) -> None:
    base_acc = json.loads((R / "base_instruct_test.json").read_text())["base"]["accuracy"]
    xs, ys, ses = [], [], []
    for r in RANKS:
        t = _load_ft_test(R / f"rank_v_r{r}_test.json")
        if not t:
            continue
        xs.append(r)
        ys.append(t[0])
        ses.append(_se(t[0], t[2]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, ys, yerr=ses, marker="o", capsize=4, color="tab:purple")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r) for r in xs])
    ax.axhline(base_acc, color="tab:orange", ls="--", lw=1.5,
               label=f"base instruct ({base_acc:.3f})")
    ax.set_xlabel("LoRA rank (v_proj only)")
    ax.set_ylabel("GSM8K-test accuracy (FT, eval instruct)")
    ax.set_title("Instruct cohort E2b — rank sweep on v_proj "
                 "(train instruct / eval instruct, E*=3, LR=2e-4)")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def plot_lr(out: Path) -> None:
    lrs = ["5e-5", "1e-4", "2e-4", "4e-4"]
    xs, ys, ses = [], [], []
    for lr in lrs:
        p = R / f"eval_lr_{lr}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        acc = d["ft"]["accuracy"]
        n = d["num_problems"]
        xs.append(float(lr))
        ys.append(acc)
        ses.append(_se(acc, n))

    if not xs:
        print(f"skip {out}: no LR sweep results yet")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, ys, yerr=ses, marker="o", capsize=4, color="tab:green")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(lrs[: len(xs)])
    ax.set_xlabel("learning rate (cosine, 3 epochs)")
    ax.set_ylabel("GSM8K dev accuracy (n=1000)")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    best = max(zip(ys, lrs[: len(xs)]))
    ax.set_title(f"Instruct LR sweep — v_proj, E*=3 (best: LR={best[1]}, {best[0]:.3f})")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def main() -> int:
    R.mkdir(exist_ok=True)
    plot_e1(R / "instruct_e1_targets.png")
    plot_rank(R / "instruct_e2_rank_q_proj.png")
    plot_vrank(R / "instruct_e2_rank_v_proj.png")
    plot_lr(R / "instruct_lr_sweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
