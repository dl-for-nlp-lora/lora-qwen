"""Figures for the 'where do FT gains come from' + timing analysis.

Timing (from results_headroom/<model>/latency.json):
  - mean generated tokens per sample, base vs FT
  - mean wall time per sample (ms), base vs FT, with the per-model speedup
Category breakdown (from analysis/ft_gains_categories.json, hand-filled after
reading the completions): stacked/bar counts of correction & regression types.

Outputs into results_headroom/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results_headroom"
FIG = R / "figures"

MODELS = {
    "qwen2_1.5b": "Qwen2-1.5B",
    "qwen25_1.5b": "Qwen2.5-1.5B",
    "qwen3_1.7b": "Qwen3-1.7B",
}
COL_BASE = "tab:gray"
COL_FT = "tab:green"


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def plot_timing() -> None:
    data = []
    for mk, label in MODELS.items():
        d = _read(R / mk / "latency.json")
        if d:
            data.append((label, d))
    if not data:
        print("skip timing: no latency.json")
        return

    labels = [x[0] for x in data]
    x = np.arange(len(labels))
    w = 0.38

    # --- tokens per sample ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    base_tok = [d["base"]["mean_gen_tokens"] for _, d in data]
    ft_tok = [d["ft"]["mean_gen_tokens"] for _, d in data]
    b1 = ax1.bar(x - w / 2, base_tok, w, color=COL_BASE, label="base (instruct)")
    b2 = ax1.bar(x + w / 2, ft_tok, w, color=COL_FT, label="LoRA-FT")
    for bars in (b1, b2):
        for b in bars:
            ax1.annotate(f"{b.get_height():.0f}",
                         (b.get_x() + b.get_width() / 2, b.get_height()),
                         textcoords="offset points", xytext=(0, 4),
                         ha="center", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("mean generated tokens / sample")
    ax1.set_title("Generation length per sample (GSM8K-test, instruct @512)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # --- wall time per sample (ms) ---
    base_ms = [d["base"]["mean_wall_sec"] * 1000 for _, d in data]
    ft_ms = [d["ft"]["mean_wall_sec"] * 1000 for _, d in data]
    b3 = ax2.bar(x - w / 2, base_ms, w, color=COL_BASE, label="base (instruct)")
    b4 = ax2.bar(x + w / 2, ft_ms, w, color=COL_FT, label="LoRA-FT")
    for bars in (b3, b4):
        for b in bars:
            ax2.annotate(f"{b.get_height():.0f}",
                         (b.get_x() + b.get_width() / 2, b.get_height()),
                         textcoords="offset points", xytext=(0, 4),
                         ha="center", fontsize=10)
    for i, (_, d) in enumerate(data):
        ax2.annotate(f"×{d['speedup_wall']:.2f} faster",
                     (i, max(base_ms[i], ft_ms[i])),
                     textcoords="offset points", xytext=(0, 18),
                     ha="center", fontsize=10, fontweight="bold", color="tab:green")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("mean wall time / sample (ms, batch=1)")
    ax2.set_title(f"Latency per sample on {data[0][1]['device']}")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG / "ft_timing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


CORR_CATS = [
    "code-style impl error (→ explicit reasoning)",
    "non-termination / runaway (→ clean ####)",
    "prose reasoning/arithmetic fix",
]
REGR_CATS = [
    "non-termination / degenerate loop",
    "lost-a-step compression",
    "other content error",
]
CORR_COL = ["tab:blue", "tab:orange", "tab:green"]
REGR_COL = ["tab:purple", "tab:red", "tab:brown"]


def _short(label: str) -> str:
    return {
        "code-style impl error (→ explicit reasoning)": "code → reasoning",
        "non-termination / runaway (→ clean ####)": "fix non-termination",
        "prose reasoning/arithmetic fix": "prose reasoning fix",
        "non-termination / degenerate loop": "non-termination loop",
        "lost-a-step compression": "lost-a-step (compression)",
        "other content error": "other content error",
    }.get(label, label)


def plot_categories() -> None:
    """Stacked correction & regression mechanism mix per model (all 3)."""
    allc = _read(REPO / "analysis" / "ft_gains_categories_all.json")
    if not allc:
        print("skip categories: ft_gains_categories_all.json not present")
        return
    models = [m for m in MODELS if m in allc]
    labels = [MODELS[m] for m in models]
    x = np.arange(len(models))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    # Shared y-scale so the two panels are directly comparable in bar height
    # (corrections dwarf regressions; on independent axes that contrast is lost).
    y_top = max(
        max(allc[m]["n_corrections"] for m in models),
        max(allc[m]["n_regressions"] for m in models),
    ) * 1.12

    # corrections (stacked)
    bottom = np.zeros(len(models))
    for cat, col in zip(CORR_CATS, CORR_COL, strict=True):
        vals = np.array([allc[m]["corrections"].get(cat, 0) for m in models])
        ax1.bar(x, vals, bottom=bottom, color=col, label=_short(cat))
        for xi, (v, b0) in enumerate(zip(vals, bottom, strict=True)):
            if v > 0:
                ax1.annotate(str(int(v)), (xi, b0 + v / 2), ha="center",
                             va="center", fontsize=9, color="white", fontweight="bold")
        bottom += vals
    for xi, m in enumerate(models):
        ax1.annotate(f"Σ {allc[m]['n_corrections']}", (xi, bottom[xi]),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("count (GSM8K-test)")
    ax1.set_title("Corrections — base ✗ → FT ✓ (what FT fixes)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    # regressions (stacked)
    bottom = np.zeros(len(models))
    for cat, col in zip(REGR_CATS, REGR_COL, strict=True):
        vals = np.array([allc[m]["regressions"].get(cat, 0) for m in models])
        ax2.bar(x, vals, bottom=bottom, color=col, label=_short(cat))
        for xi, (v, b0) in enumerate(zip(vals, bottom, strict=True)):
            if v > 0:
                ax2.annotate(str(int(v)), (xi, b0 + v / 2), ha="center",
                             va="center", fontsize=9, color="white", fontweight="bold")
        bottom += vals
    for xi, m in enumerate(models):
        ax2.annotate(f"Σ {allc[m]['n_regressions']}", (xi, bottom[xi]),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("count (GSM8K-test)")
    ax2.set_title("Regressions — base ✓ → FT ✗ (what FT breaks)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, y_top)

    fig.suptitle("Where the fine-tuning effect comes from — by mechanism, per model",
                 fontsize=13)
    fig.tight_layout()
    out = FIG / "ft_gain_categories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def plot_net() -> None:
    """Net correct-count change with corrections up / regressions down per model."""
    allc = _read(REPO / "analysis" / "ft_gains_categories_all.json")
    if not allc:
        return
    models = [m for m in MODELS if m in allc]
    labels = [MODELS[m] for m in models]
    x = np.arange(len(models))
    corr = [allc[m]["n_corrections"] for m in models]
    regr = [-allc[m]["n_regressions"] for m in models]
    net = [allc[m]["ft_correct"] - allc[m]["base_correct"] for m in models]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(x, corr, color="tab:green", label="corrections (base ✗ → FT ✓)")
    ax.bar(x, regr, color="tab:red", label="regressions (base ✓ → FT ✗)")
    ax.axhline(0, color="black", lw=1)
    for xi, n in enumerate(net):
        ax.annotate(f"net {n:+d}", (xi, corr[xi]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("# GSM8K-test problems")
    ax.set_title("Fine-tuning: problems fixed vs. broken (net = accuracy change)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "ft_net_changes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    plot_timing()
    plot_categories()
    plot_net()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
