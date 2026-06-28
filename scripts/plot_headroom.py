"""Figures + slide table for the controlled headroom study.

Controlled pair: **Qwen2-1.5B vs Qwen2.5-1.5B** — identical size / architecture
(28 layers, 12 Q / 2 KV heads, hidden 1536, GQA), only the pretraining math level
differs. The single varying factor is the base model's GSM8K headroom.

Reads the funnel outputs in ``results_headroom/<model>/``:
  decisions.json            stage winners + dev sweep scores
  final_base_test.json      base 0-shot / few-shot / instruct @512 on GSM8K-test
  final_lora_test.json      best LoRA @512 on test
  final_full_ft_test.json   full FT @512 on test
  qwen2_1.5b/diag_dev_test_gap.json   base/FT on dev-500 vs test-500 (contamination)

Writes PNGs into ``results_headroom/figures/`` and a slide-ready markdown table
to ``analysis/headroom_summary.md``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results_headroom"
FIG = R / "figures"

MODELS = {
    "qwen2_1.5b":  "Qwen2-1.5B",
    "qwen25_1.5b": "Qwen2.5-1.5B",
}
COL = {"qwen2_1.5b": "tab:blue", "qwen25_1.5b": "tab:orange"}

# Shared y-axis ceiling for the per-model method bars so the two figures are
# directly comparable in height (max bar is ~0.59).
METHODS_YMAX = 0.65

# Extra reference backbones (not part of the controlled 1.5B pair: different
# size/architecture). Trained with the identical recipe and read from
# results_headroom/ref_<key>/summary.json. They extend the headroom range so the
# trend (gain -> 0 -> negative) is visible across the full spectrum.
REFS = {
    "qwen3_1.7b": "Qwen3-1.7B",
}
REF_COL = "tab:gray"


def _se(acc: float, n: int) -> float:
    return math.sqrt(acc * (1 - acc) / n) if n else 0.0


def _read(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _read_test_sweeps(md: Path, best_target: str | None) -> tuple[dict, dict]:
    """Re-scored E1/E2 sweep accuracies on GSM8K-test (clean numbers).

    The funnel selects target/rank on the held-out dev slice, but that slice is a
    part of GSM8K-train and is inflated by pretraining (see contamination plot).
    To put honest absolute numbers on the slide we re-score every dev-trained
    adapter on test (results_headroom/<model>/test_sweeps/). Selection is still
    dev; this is a robustness check that the dev ranking holds on test.
    """
    ts = md / "test_sweeps"
    e1: dict[str, float] = {}
    for t in ("q", "v", "qv", "attention", "all_linear"):
        r = _read(ts / f"e1_{t}.json")
        if r:
            e1[t] = r["ft"]["accuracy"]
    e2: dict[str, float] = {}
    if best_target:
        for rank in (1, 2, 4, 8, 16, 32, 64):
            r = _read(ts / f"e2_{best_target}_r{rank}.json")
            if r:
                e2[str(rank)] = r["ft"]["accuracy"]
    return e1, e2


def collect(mk: str) -> dict | None:
    md = R / mk
    base = _read(md / "final_base_test.json")
    lora = _read(md / "final_lora_test.json")
    if not base or not lora:
        return None
    dec = _read(md / "decisions.json") or {}
    full = (_read(md / "final_full_ft_test.json") or {}).get("ft", {}).get("accuracy")
    br = base["runs"]
    n = base["runs"]["instruct"]["total"]
    e1_test, e2_test = _read_test_sweeps(md, dec.get("best_target"))
    return {
        "key": mk, "label": MODELS[mk], "n": n,
        "base_zeroshot": br["zeroshot"]["accuracy"],
        "base_fewshot": br["fewshot"]["accuracy"],
        "base_instruct": br["instruct"]["accuracy"],
        "lora": lora["ft"]["accuracy"],
        "full_ft": full,
        "delta": lora["ft"]["accuracy"] - br["instruct"]["accuracy"],
        "best_lr": dec.get("best_lr"), "best_target": dec.get("best_target"),
        "best_rank": dec.get("best_rank"),
        "e1_scores": dec.get("e1_scores", {}), "e2_scores": dec.get("e2_scores", {}),
        "e1_test": e1_test, "e2_test": e2_test,
        "diag_epoch_curve": dec.get("diag_epoch_curve", {}),
    }


def plot_comparison(data: list[dict]) -> None:
    """Grouped bars: per model, base instruct / best LoRA / full FT on test."""
    cats = [("base_instruct", "base\n(instruct)"), ("lora", "best LoRA"),
            ("full_ft", "full FT")]
    x = np.arange(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, d in enumerate(data):
        ys = [d[c] if d[c] is not None else float("nan") for c, _ in cats]
        ses = [_se(y, d["n"]) for y in ys]
        bars = ax.bar(x + (i - 0.5) * w, ys, w, yerr=ses, capsize=4,
                      color=COL[d["key"]], label=d["label"])
        for b, y in zip(bars, ys, strict=True):
            if not math.isnan(y):
                ax.annotate(f"{y:.3f}", (b.get_x() + b.get_width() / 2, y),
                            textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in cats])
    ax.set_ylabel(f"GSM8K-test accuracy (n={data[0]['n']}, @512)")
    ax.set_title("Controlled headroom pair — same size/arch, only pretraining math differs")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "headroom_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def plot_methods_per_model(d: dict) -> None:
    """One figure per model: 0-shot, base instruct, best LoRA, full FT on test."""
    cats = [
        ("base_zeroshot", "base\n(0-shot)", "tab:gray"),
        ("base_instruct", "base\n(instruct)", "tab:red"),
        ("lora", "best LoRA", COL[d["key"]]),
        ("full_ft", "full FT", "tab:green"),
    ]
    ys = [d[c] if d[c] is not None else float("nan") for c, _, _ in cats]
    ses = [_se(y, d["n"]) for y in ys]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.bar(x, ys, 0.62, yerr=ses, capsize=4,
                  color=[c for _, _, c in cats], zorder=3)
    for b, y in zip(bars, ys, strict=True):
        if not math.isnan(y):
            ax.annotate(f"{y:.3f}", (b.get_x() + b.get_width() / 2, y),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in cats])
    ax.set_ylabel(f"GSM8K-test accuracy (n={d['n']}, @512)")
    cfg = f"LoRA: {d['best_target']} r{d['best_rank']}, LR {d['best_lr']}"
    ax.set_title(f"{d['label']} — adaptation methods on GSM8K-test\n({cfg}; "
                 f"error bars = ±1 binomial SE)")
    ax.set_ylim(0, METHODS_YMAX)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / f"{d['key']}_methods.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def collect_ref(key: str) -> dict | None:
    s = _read(R / f"ref_{key}" / "summary.json")
    if not s:
        return None
    base = _read(R / f"ref_{key}" / "final_base_test.json")
    n = base["runs"]["instruct"]["total"] if base else 1319
    bi = s["base_test"]["instruct"]
    return {
        "key": key, "label": REFS[key], "n": n, "is_ref": True,
        "base_zeroshot": s["base_test"]["zeroshot"],
        "base_fewshot": s["base_test"]["fewshot"],
        "base_instruct": bi,
        "lora": s["lora_test"], "full_ft": None,
        "delta": s["lora_test"] - bi,
    }


def plot_delta(data: list[dict], refs: list[dict]) -> None:
    """The headroom message: LoRA gain vs base instruct, per model.

    The controlled 1.5B pair is shown in color; extra reference backbones (gray)
    extend the range so the gain -> 0 -> negative trend is visible.
    """
    allpts = sorted(data + refs, key=lambda d: d["base_instruct"])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    labels = [f"{d['label']}\n(base {d['base_instruct']:.2f})" for d in allpts]
    deltas = [d["delta"] for d in allpts]
    colors = [REF_COL if d.get("is_ref") else COL[d["key"]] for d in allpts]
    bars = ax.bar(labels, deltas, color=colors, width=0.6)
    ax.axhline(0, color="grey", lw=1)
    for b, d in zip(bars, deltas, strict=True):
        ax.annotate(f"{d:+.3f}", (b.get_x() + b.get_width() / 2, d),
                    textcoords="offset points", xytext=(0, 6 if d >= 0 else -14),
                    ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("LoRA gain  Δ = acc(LoRA) − acc(base instruct)")
    ax.set_title("LoRA gain shrinks with pretraining headroom — and turns negative once mastered")
    ax.grid(axis="y", alpha=0.3)
    if refs:
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="tab:blue", label="controlled 1.5B pair"),
            Patch(color=REF_COL, label="reference backbone (other size/arch)"),
        ], loc="upper right", fontsize=9)
    fig.tight_layout()
    out = FIG / "headroom_delta.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def plot_trend(data: list[dict], refs: list[dict]) -> None:
    """LoRA gain as a function of base level — the headroom curve."""
    allpts = sorted(data + refs, key=lambda d: d["base_instruct"])
    xs = [d["base_instruct"] for d in allpts]
    ys = [d["delta"] for d in allpts]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhline(0, color="grey", lw=1)
    ax.plot(xs, ys, "-", color="black", lw=1.2, zorder=1)
    for d in allpts:
        c = REF_COL if d.get("is_ref") else COL[d["key"]]
        ax.scatter([d["base_instruct"]], [d["delta"]], s=120, color=c, zorder=2)
        # Place the label below-right for the top point so it clears the title.
        dy = -28 if d["delta"] == max(ys) else 8
        ax.annotate(f"{d['label']}\nΔ={d['delta']:+.3f}",
                    (d["base_instruct"], d["delta"]),
                    textcoords="offset points", xytext=(10, dy), fontsize=9)
    ax.set_xlabel("base GSM8K-test accuracy (instruct prompt, @512)")
    ax.set_ylabel("LoRA gain  Δ = acc(LoRA) − acc(base instruct)")
    ax.set_title("Headroom curve — LoRA gain vs. how much the base already masters GSM8K")
    ax.margins(x=0.12, y=0.12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "headroom_trend.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def _baseline_lines(ax, d: dict) -> None:
    """Draw base zero-shot and instruct accuracy as horizontal reference lines."""
    ax.axhline(d["base_instruct"], ls="--", lw=1.4, color="black",
               label=f"base instruct ({d['base_instruct']:.3f})")
    ax.axhline(d["base_zeroshot"], ls=":", lw=1.4, color="dimgray",
               label=f"base zero-shot ({d['base_zeroshot']:.3f})")


def plot_sweeps(d: dict) -> None:
    """E1 target bars + E2 rank curve on GSM8K-test, with base reference lines.

    Numbers are re-scored on test (selection stays on dev — see _read_test_sweeps).
    """
    e1 = d.get("e1_test") or d["e1_scores"]
    e2 = d.get("e2_test") or d["e2_scores"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if e1:
        ks = list(e1)
        axes[0].bar(ks, [e1[k] for k in ks], color=COL[d["key"]], zorder=3)
        axes[0].set_title(
            f"{d['label']} — E1 target sweep (test, iso-budget ≈2.5M; dev-selected)")
        axes[0].set_ylabel(f"GSM8K-test accuracy (n={d['n']})")
        for i, k in enumerate(ks):
            axes[0].annotate(f"{e1[k]:.3f}", (i, e1[k]), textcoords="offset points",
                             xytext=(0, 5), ha="center", fontsize=9)
        _baseline_lines(axes[0], d)
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].legend(loc="lower right", fontsize=8)
    if e2:
        xs = sorted(int(r) for r in e2)
        ys = [e2[str(r)] for r in xs]
        ses = [_se(y, d["n"]) for y in ys]
        axes[1].errorbar(xs, ys, yerr=ses, marker="o", color=COL[d["key"]],
                         capsize=4, zorder=3)
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels([str(r) for r in xs])
        axes[1].set_title(
            f"{d['label']} — E2 rank sweep on {d['best_target']} (test; dev-selected)")
        axes[1].set_xlabel("LoRA rank")
        axes[1].set_ylabel(f"GSM8K-test accuracy (n={d['n']})")
        for xx, yy in zip(xs, ys, strict=True):
            axes[1].annotate(f"{yy:.3f}", (xx, yy), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=9)
        _baseline_lines(axes[1], d)
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = FIG / f"{d['key']}_sweeps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def plot_contamination() -> None:
    """Base & FT on dev-500 vs test-500: the dev slice is inflated because the
    base model already saw GSM8K-train (the dev slice's source) in pretraining."""
    g = _read(R / "qwen2_1.5b" / "diag_dev_test_gap.json")
    if not g:
        print("skip contamination plot: no diag_dev_test_gap.json")
        return
    c = g["cells"]
    groups = [("Base\n(no training)", c["base_dev"]["accuracy"], c["base_test"]["accuracy"]),
              ("best LoRA", c["ft_dev"]["accuracy"], c["ft_test"]["accuracy"])]
    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5.5))
    dev = ax.bar(x - w / 2, [g[1] for g in groups], w, color="tab:green",
                 label="dev-500 (held-out slice of GSM8K-train)")
    tst = ax.bar(x + w / 2, [g[2] for g in groups], w, color="tab:gray",
                 label="test-500 (official GSM8K-test)")
    for bars in (dev, tst):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("GSM8K accuracy (n=500, instruct @512)")
    ax.set_title("Qwen2-1.5B: dev≫test even for the UNTRAINED base\n"
                 "→ GSM8K-train contamination in pretraining, not a training/leakage bug")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "contamination_dev_vs_test.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def write_md(data: list[dict], refs: list[dict] | None = None) -> None:
    refs = refs or []
    n = data[0]["n"]
    lines = [
        "# Headroom study — LoRA gain vs. pretraining headroom",
        "",
        "**Controlled experiment:** Qwen2-1.5B vs Qwen2.5-1.5B — identical size and "
        "architecture (28 layers, 12 Q / 2 KV heads, hidden 1536, GQA), trained with "
        "the *same* recipe (GSM8K-train → test, instruct prompt, 1 epoch, LR 2e-4, "
        "512-token budget). The only varying factor is the base model's pretraining "
        "math level, i.e. its GSM8K headroom.",
        "",
        f"All numbers are GSM8K-**test** accuracy (n={n}) at a 512-token budget.",
        "",
        "## Final comparison (GSM8K-test @512)",
        "",
        "| Model | base 0-shot | base few-shot | base instruct | best LoRA | "
        "Δ (LoRA − base instruct) | full FT |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in data:
        ff = f"{d['full_ft']:.3f}" if d["full_ft"] is not None else "—"
        lines.append(
            f"| {d['label']} | {d['base_zeroshot']:.3f} | {d['base_fewshot']:.3f} | "
            f"{d['base_instruct']:.3f} | {d['lora']:.3f} | **{d['delta']:+.3f}** | {ff} |"
        )
    for r in sorted(refs, key=lambda x: x["base_instruct"]):
        lines.append(
            f"| {r['label']} (ref) | {r['base_zeroshot']:.3f} | {r['base_fewshot']:.3f} | "
            f"{r['base_instruct']:.3f} | {r['lora']:.3f} | **{r['delta']:+.3f}** | — |"
        )
    if refs:
        lines += [
            "",
            "*(ref) = reference backbone trained with the identical recipe but "
            "differing in size/architecture from the controlled 1.5B pair; included "
            "only to extend the headroom range, not as a controlled cell.*",
        ]
    lines += [
        "",
        "## Best LoRA configuration per model (from the dev funnels)",
        "",
        "| Model | best LR | best target | best rank |",
        "|---|---|---|---|",
    ]
    for d in data:
        lines.append(f"| {d['label']} | {d['best_lr']} | {d['best_target']} | {d['best_rank']} |")
    lines += [
        "",
        "## Reading",
        "",
        f"- **Headroom effect.** The weak-pretrained Qwen2-1.5B gains "
        f"**{data[0]['delta']:+.3f}** from LoRA, the strong-pretrained Qwen2.5-1.5B only "
        f"**{data[1]['delta']:+.3f}** — LoRA helps in proportion to how much room the base "
        "model still has on the task. Extending the range with a larger reference "
        "backbone (Qwen3-1.7B, base instruct ≈0.76), the gain goes **negative** "
        "(≈−0.09): once a model already masters GSM8K, in-style SFT trades reasoning "
        "for format and *loses* accuracy.",
        "- **Where the gain comes from (Qwen2).** The Qwen2-1.5B base ignores the "
        "`#### N` instruct format and answers in a Python-code style (its pretraining "
        "default), so its base-instruct score is low (0.22) despite real reasoning "
        "ability; LoRA mostly buys format/style compliance. 0% of these were "
        "length-truncated, so this is genuine, not an eval artifact.",
        "- **Stage funnel.** Both models confirm 1 epoch as the operating point and "
        "LR 2e-4 as best. The high-headroom model prefers spreading a *tiny* adapter "
        "across attention (rank 1); the low-headroom model prefers a small q/v adapter "
        "(rank 16) and is actively hurt by larger target sets.",
        "",
        "## Caveat — GSM8K-train contamination in the dev metric",
        "",
        "The sweep dev slice is a held-out part of GSM8K-**train**. The Qwen models "
        "appear to have seen GSM8K-train during pretraining: the **untrained** Qwen2-1.5B "
        "base already scores **0.87 on dev-500 vs 0.25 on test-500** (same prompt, same "
        "budget, ~0% truncation, identical question difficulty). Since no fine-tuning is "
        "involved, this gap is pretraining memorization of the train split, not a "
        "training/leakage bug in our pipeline (train/dev indices are disjoint; 0/1000 "
        "dev items have a ≥0.8-similar train neighbor). **All headline numbers above are "
        "on the clean GSM8K-test split.** The dev metric is only used for *relative* "
        "stage decisions within a model, where the constant offset cancels.",
        "",
        "## Target / rank sweeps (E1 / E2)",
        "",
        "The funnel selects the target set (E1) and rank (E2) on the held-out dev "
        "slice. To report honest absolute numbers, every dev-trained adapter is "
        "re-scored on GSM8K-test below (selection stays on dev — this is a "
        "robustness check, no re-training). On test the within-sweep spread "
        "collapses to ≈1 binomial SE (±0.014 at n=1319): both the target set and "
        "the rank are largely interchangeable, and a tiny rank (1–2) already "
        "matches the best — reproducing the paper's \"LoRA is robust to target "
        "choice / small rank suffices\" finding.",
        "",
        "![qwen2 sweeps](../results_headroom/figures/qwen2_1.5b_sweeps.png)",
        "",
        "![qwen2.5 sweeps](../results_headroom/figures/qwen25_1.5b_sweeps.png)",
        "",
        "## Adaptation methods per model (GSM8K-test @512)",
        "",
        "Per model, the four operating points on the selected best config: base "
        "zero-shot, base instruct, best LoRA, and full fine-tuning.",
        "",
        "![qwen2 methods](../results_headroom/figures/qwen2_1.5b_methods.png)",
        "",
        "![qwen2.5 methods](../results_headroom/figures/qwen25_1.5b_methods.png)",
        "",
        "## Figures",
        "",
        "![comparison](../results_headroom/figures/headroom_comparison.png)",
        "",
        "![delta](../results_headroom/figures/headroom_delta.png)",
        "",
        "![trend](../results_headroom/figures/headroom_trend.png)",
        "",
        "![contamination](../results_headroom/figures/contamination_dev_vs_test.png)",
    ]
    out = REPO / "analysis" / "headroom_summary.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    data = [collect(mk) for mk in MODELS]
    data = [d for d in data if d]
    if not data:
        print("no model data found")
        return 1
    refs = [r for r in (collect_ref(k) for k in REFS) if r]
    plot_comparison(data)
    plot_delta(data, refs)
    plot_trend(data, refs)
    for d in data:
        plot_sweeps(d)
        plot_methods_per_model(d)
    plot_contamination()
    write_md(data, refs)
    print("\n=== summary ===")
    for d in data:
        print(f"{d['label']}: base_instruct={d['base_instruct']:.3f} "
              f"lora={d['lora']:.3f} delta={d['delta']:+.3f} "
              f"(best {d['best_target']}/r{d['best_rank']}/lr{d['best_lr']})")
    for r in refs:
        print(f"{r['label']} (ref): base_instruct={r['base_instruct']:.3f} "
              f"lora={r['lora']:.3f} delta={r['delta']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
