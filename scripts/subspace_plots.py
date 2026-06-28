#!/usr/bin/env python3
"""Plotting helpers for the E3 subspace analyses.

Figure rendering for ``analyze_subspace.py``:

- ``plot_grassmann_heatmap``  : E3a inter-layer similarity heatmap (Fig. 3)
- ``_plot_cross_seed``        : E3b cross-seed φ(i,j) heatmap + top-k curves (Fig. 4)
- ``plot_amplification_table``: E3c Table 7 norms + amplification-vs-k curve
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_grassmann_heatmap(
    similarity_matrix: np.ndarray,
    layer_names: list[str],
    module_type: str,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Create heatmap visualization like paper Fig. 3."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        similarity_matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto"
    )

    ax.set_xticks(range(len(layer_names)))
    ax.set_yticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, fontsize=10)
    ax.set_yticklabels(layer_names, fontsize=10)

    cbar = plt.colorbar(im, ax=ax, label="Grassmann Similarity")
    cbar.ax.tick_params(labelsize=10)

    title = f"{module_type} - Inter-layer Subspace Similarity"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)

    for i in range(len(layer_names)):
        for j in range(len(layer_names)):
            val = similarity_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {output_path}")
    plt.close()


def plot_amplification_table(
    all_data: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot the amplification analysis.

    Left: mean Frobenius norms per module (log scale) — ‖ΔW‖_F and the three
    projections (ΔW-subspace, W-subspace, random). Right: amplification factor
    vs the number of top-k ΔW directions kept.
    """
    modules = list(all_data.keys())
    if not modules:
        print("    No data to plot.")
        return

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(15, 6))

    bar_keys = [
        ("delta_norm", "‖ΔW‖_F"),
        ("proj_onto_delta_subspace", "ΔW-subspace"),
        ("proj_onto_W_subspace", "W-subspace"),
        ("proj_onto_random_subspace", "random"),
    ]
    x = np.arange(len(modules))
    width = 0.8 / len(bar_keys)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#7f7f7f"]
    for i, (key, label) in enumerate(bar_keys):
        vals = [all_data[m]["table7_means"][key] for m in modules]
        ax_bar.bar(x + i * width, vals, width, label=label, color=colors[i])

    ax_bar.set_yscale("log")
    ax_bar.set_xticks(x + width * (len(bar_keys) - 1) / 2)
    ax_bar.set_xticklabels(modules)
    ax_bar.set_ylabel("Mean Frobenius norm (log scale)", fontsize=12)
    ax_bar.set_title("Table 7 norms: ‖ΔW‖_F vs ‖UᵀWVᵀ‖_F", fontsize=13)
    ax_bar.legend(fontsize=10)
    ax_bar.grid(True, axis="y", alpha=0.3)

    for xi, m in zip(x, modules):
        amp = all_data[m]["mean"]
        ax_bar.text(
            xi + width * (len(bar_keys) - 1) / 2,
            ax_bar.get_ylim()[1],
            f"amp≈{amp:.1f}×",
            ha="center", va="top", fontsize=10, fontweight="bold",
        )

    for m in modules:
        topk_means = all_data[m].get("topk_amplification_means", {})
        if not topk_means:
            continue
        ks = sorted(int(k) for k in topk_means)
        vals = [topk_means[str(k)] for k in ks]
        ax_line.plot(ks, vals, marker="o", label=m)

    ax_line.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="no amplification (=1)")
    ax_line.set_xscale("log", base=2)
    ax_line.set_xlabel("top-k ΔW directions kept", fontsize=12)
    ax_line.set_ylabel("Amplification factor  ‖ΔWₖ‖_F / ‖UₖᵀWVₖᵀ‖_F", fontsize=12)
    ax_line.set_title("Amplification vs rank truncation k", fontsize=13)
    ax_line.legend(fontsize=10)
    ax_line.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved Table 7 plot to {output_path}")
    plt.close()


def plot_cross_seed(
    avg_grid: np.ndarray,
    per_layer: dict[str, dict[str, float]],
    top_ks: list[int],
    baseline_per_k: dict[str, float],
    module_type: str,
    output_path: Path,
) -> None:
    """Two-panel E3b figure: autoscaled φ(i,j) heatmap (left) and per-layer
    top-k similarity with its random-chance baseline k/d (right)."""
    fig, (ax_grid, ax_bar) = plt.subplots(1, 2, figsize=(15, 6))

    cap = avg_grid.shape[0]
    vmax = float(max(avg_grid.max(), 1e-6))
    im = ax_grid.imshow(
        avg_grid, cmap="viridis", vmin=0, vmax=vmax, origin="upper", aspect="auto"
    )
    ax_grid.set_xticks(range(cap))
    ax_grid.set_yticks(range(cap))
    ax_grid.set_xticklabels(range(1, cap + 1), fontsize=8)
    ax_grid.set_yticklabels(range(1, cap + 1), fontsize=8)
    ax_grid.set_xlabel("top-j directions (seed 2)", fontsize=12)
    ax_grid.set_ylabel("top-i directions (seed 1)", fontsize=12)
    ax_grid.set_title(
        f"{module_type} - cross-seed φ(i,j), avg over layers\n"
        f"(autoscaled, max={vmax:.3f})",
        fontsize=12,
    )
    plt.colorbar(im, ax=ax_grid, label="normalized similarity φ")

    layers = list(per_layer.keys())
    layer_idx = [int(name[1:]) for name in layers]
    ymax = 0.0
    for k in top_ks:
        key = f"top{k}"
        vals = [per_layer[name].get(key, np.nan) for name in layers]
        ymax = max(ymax, np.nanmax(vals))
        line, = ax_bar.plot(layer_idx, vals, marker="o", label=f"top-{k}")
        baseline = baseline_per_k.get(key)
        if baseline is not None:
            ax_bar.axhline(
                baseline, color=line.get_color(), linestyle=":", alpha=0.6
            )
    ax_bar.set_xlabel("Layer index", fontsize=12)
    ax_bar.set_ylabel("Cross-seed similarity φ(k,k)", fontsize=12)
    ax_bar.set_title(
        f"{module_type} - shared directions vs depth\n"
        "(dotted = random-chance baseline k/d)",
        fontsize=12,
    )
    ax_bar.set_ylim(0, max(0.1, ymax * 1.25))
    ax_bar.legend(fontsize=10)
    ax_bar.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved cross-seed plot to {output_path}")
    plt.close()
