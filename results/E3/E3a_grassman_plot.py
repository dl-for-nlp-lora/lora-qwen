"""E3a plotting: reproduce Fig. 3 layout from Hu et al. 2021.

For each module (q_proj, v_proj) and each of the 4 layers, produces a
4-panel figure matching the paper's Fig. 3:

    Panel 1: full i×j grid  (i=1..8, j=1..64)   ΔW_q
    Panel 2: full i×j grid  (i=1..8, j=1..64)   ΔW_v
    Panel 3: zoom lower-left (i=1..8, j=1..8)   ΔW_q
    Panel 4: zoom lower-left (i=1..8, j=1..8)   ΔW_v

Two output figures are produced:
  E3a_fig3_L{layer}.png  -- one per layer, 4-panel layout
  E3a_fig3_alllayers_{module}.png  -- one row per module across all layers

Usage:
    python results/E3/E3_grassman_plot.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "e3a_output"

MODULES = ["q_proj", "v_proj"]
MODULE_LABELS = {"q_proj": r"$\Delta W_q$", "v_proj": r"$\Delta W_v$"}

# ── Load data ─────────────────────────────────────────────────────────────────
def load(module: str) -> dict:
    path = OUT_DIR / f"E3a_rank_vs_rank_{module}.json"
    return json.loads(path.read_text())


def grid_to_array(data: dict, layer: str) -> np.ndarray:
    """Return full grid as numpy array (i x j)."""
    return np.array(data["per_layer"][layer])  # shape (8, 64)


def mask_full_panel(grid: np.ndarray) -> np.ndarray:
    """Grey out cells where j < i — used on the full (j=1..64) panels.

    When j < i, r=64's j directions cannot fully contain r=8's i-dimensional
    subspace, so phi is structurally constrained. Paper Fig. 3, panels 1 & 2.
    NaN renders as grey via cmap.set_bad().
    """
    masked = grid.astype(float).copy()
    n_i, n_j = masked.shape
    for i in range(n_i):
        for j in range(n_j):
            if j < i:           # 0-indexed: grey where col < row
                masked[i, j] = np.nan
    return masked


def mask_zoom_panel(grid: np.ndarray) -> np.ndarray:
    """Grey out cells where j > i — used on the zoom (j=1..8) panels.

    The zoom panels show exactly the lower-left triangle that was grey in the
    full panels (j < i region). Cells where j > i were already visible in the
    full panels, so they are greyed here to focus attention on the new region.
    Paper Fig. 3, panels 3 & 4. NaN renders as grey via cmap.set_bad().
    """
    masked = grid.astype(float).copy()
    n_i, n_j = masked.shape
    for i in range(n_i):
        for j in range(n_j):
            if j > i:           # 0-indexed: grey where col > row
                masked[i, j] = np.nan
    return masked


# ── Plotting helpers ──────────────────────────────────────────────────────────
CMAP   = "magma"     # high values = bright/warm, matching paper's Fig. 3 convention
VMIN   = 0.0
VMAX   = 1.0

def add_heatmap(ax, data_2d: np.ndarray, title: str,
                xlabel: str, ylabel: str,
                vmin: float = VMIN, vmax: float = VMAX,
                x_tick_step: int = 8,
                grey_nan: bool = False) -> object:
    """Draw one heatmap panel and return the image handle for colorbar.

    grey_nan: if True, NaN cells (masked lower triangle) render as grey,
              matching the paper's Fig. 3 zoom panel style.
    """
    cmap = plt.get_cmap(CMAP).copy()
    if grey_nan:
        cmap.set_bad(color="#AAAAAA")

    im = ax.imshow(
        data_2d,
        aspect="auto",
        origin="upper",       # i=1 at top, matching paper orientation
        vmin=vmin, vmax=vmax,
        cmap=cmap,
    )
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

    # x ticks: label every x_tick_step-th column (1-indexed)
    n_j = data_2d.shape[1]
    x_ticks = list(range(0, n_j, x_tick_step))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(t + 1) for t in x_ticks], fontsize=7)

    # y ticks: every row
    n_i = data_2d.shape[0]
    ax.set_yticks(range(n_i))
    ax.set_yticklabels([str(i + 1) for i in range(n_i)], fontsize=7)

    return im


# ── Figure 1: one 4-panel figure per layer (closest to paper's Fig. 3) ────────
# L9 is the best representative layer: strongest φ(1,1) signal for both modules,
# mid-network position (9/28 ≈ layer 32/96 in GPT-3 terms).
# A dedicated high-res version is also saved as E3a_fig3_PRESENTATION.png.
PRESENTATION_LAYER = "L9"

def _make_four_panel(data_q, data_v, layer, suptitle, vmax=None):
    """Build and return a 4-panel Fig-3-style figure for one layer."""
    raw_q  = grid_to_array(data_q, layer)           # (8, 64) unmasked
    raw_v  = grid_to_array(data_v, layer)           # (8, 64) unmasked
    grid_q = mask_full_panel(raw_q)                 # grey j < i  (full panels)
    grid_v = mask_full_panel(raw_v)
    zoom_q = mask_zoom_panel(raw_q[:, :8])          # grey j > i  (zoom panels)
    zoom_v = mask_zoom_panel(raw_v[:, :8])

    if vmax is None:
        # compute vmax from unmasked data so both panel sets share the same scale
        vmax = 0.5

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.suptitle(suptitle, fontsize=12, y=1.02)

    im = add_heatmap(axes[0], grid_q, r"$\Delta W_q$", "j", "i",
                     vmin=0, vmax=vmax, x_tick_step=8, grey_nan=True)
    add_heatmap(axes[1], grid_v, r"$\Delta W_v$", "j", "i",
                vmin=0, vmax=vmax, x_tick_step=8, grey_nan=True)
    add_heatmap(axes[2], zoom_q, r"$\Delta W_q$ (zoom)", "j", "i",
                vmin=0, vmax=vmax, x_tick_step=1, grey_nan=True)
    add_heatmap(axes[3], zoom_v, r"$\Delta W_v$ (zoom)", "j", "i",
                vmin=0, vmax=vmax, x_tick_step=1, grey_nan=True)

    fig.colorbar(im, ax=axes, label=r"$\phi(i,j)$", shrink=0.85,
                 location="right", pad=0.02)
    return fig


def plot_per_layer(data_q: dict, data_v: dict) -> None:
    """4-panel figure per layer: full q, full v, zoom q, zoom v."""
    layers = data_q["layers"]

    for layer in layers:
        title = (
            rf"$\phi(A_{{r=8}}, A_{{r=64}}, i, j)$ — {layer}  (Qwen3-1.7B)"
            + (" ← paper-comparison layer" if layer == PRESENTATION_LAYER else "")
        )
        fig = _make_four_panel(data_q, data_v, layer, title)
        out_path = OUT_DIR / f"E3a_fig3_{layer}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")

    # Dedicated cleaner presentation version of the best layer
    pres_title = (
        rf"$\phi(A_{{r=8}}, A_{{r=64}}, i, j)$  —  layer {PRESENTATION_LAYER}  "
        r"(Qwen3-1.7B,  $q$+$v$ target,  r=8 vs r=64)"
    )
    fig = _make_four_panel(data_q, data_v, PRESENTATION_LAYER, pres_title)
    out_path = OUT_DIR / "E3a_fig3_PRESENTATION.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  ← use this one in the slides")


# ── Figure 2: all layers side by side, one row per module ────────────────────
def plot_all_layers_grid(data_q: dict, data_v: dict) -> None:
    """2-row × 4-col grid: rows = modules, cols = layers. Shows zoom (8×8) only."""
    layers = data_q["layers"]
    n_layers = len(layers)

    fig, axes = plt.subplots(2, n_layers, figsize=(4 * n_layers, 7),
                             sharex=False, sharey=True)
    fig.suptitle(
        rf"$\phi(A_{{r=8}}, A_{{r=64}}, i, j)$ — zoom j=1..8, all layers",
        fontsize=13, y=1.02,
    )

    module_data = {"q_proj": data_q, "v_proj": data_v}
    module_label = {"q_proj": r"$\Delta W_q$", "v_proj": r"$\Delta W_v$"}

    vmax = max(
        max(np.array(data["per_layer"][l])[:, :8].max()
            for l in layers)
        for data in [data_q, data_v]
    )

    im_handle = None
    for row, module in enumerate(["q_proj", "v_proj"]):
        data = module_data[module]
        for col, layer in enumerate(layers):
            ax = axes[row][col]
            grid = mask_zoom_panel(
                np.array(data["per_layer"][layer])[:, :8]  # zoom: grey j > i
            )

            im_handle = add_heatmap(
                ax, grid,
                title=f"{module_label[module]}  {layer}",
                xlabel="j", ylabel="i" if col == 0 else "",
                vmin=0, vmax=vmax, x_tick_step=1, grey_nan=True,
            )

    fig.colorbar(im_handle, ax=axes, label=r"$\phi(i,j)$",
                 shrink=0.6, location="right", pad=0.02)

    out_path = OUT_DIR / "E3a_fig3_all_layers_zoom.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ── Figure 3: full grid, all layers, one figure per module ───────────────────
def plot_full_grid_per_module(data_q: dict, data_v: dict) -> None:
    """1×4 row of full (8×64) heatmaps per module."""
    for module, data in [("q_proj", data_q), ("v_proj", data_v)]:
        layers = data["layers"]
        fig, axes = plt.subplots(1, len(layers),
                                 figsize=(5 * len(layers), 3.8), sharey=True)
        fig.suptitle(
            rf"$\phi(A_{{r=8}}, A_{{r=64}}, i, j)$ — {MODULE_LABELS[module]}, full grid",
            fontsize=12, y=1.02,
        )

        vmax = max(np.array(data["per_layer"][l]).max() for l in layers)

        im_handle = None
        for ax, layer in zip(axes, layers):
            grid = mask_full_panel(np.array(data["per_layer"][layer]))  # grey j < i
            im_handle = add_heatmap(ax, grid, layer, "j (r=64 directions)",
                                    "i (r=8 directions)",
                                    vmin=0, vmax=vmax, x_tick_step=8,
                                    grey_nan=True)

        fig.colorbar(im_handle, ax=axes, label=r"$\phi(i,j)$",
                     shrink=0.85, location="right", pad=0.02)

        out_path = OUT_DIR / f"E3a_full_grid_{module}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading JSON results...")
    data_q = load("q_proj")
    data_v = load("v_proj")

    # Check that the JSONs have the full 8×64 grid (not the old 8×8)
    sample = np.array(data_q["per_layer"][data_q["layers"][0]])
    if sample.shape[1] < 64:
        raise SystemExit(
            f"JSON grid is {sample.shape} — looks like the old 8×8 output. "
            "Re-run E3_grassman_script.py first to generate the full 8×64 grid."
        )

    print("Generating figures...")
    plot_per_layer(data_q, data_v)           # 4 files: one per layer, 4 panels each
    plot_all_layers_grid(data_q, data_v)     # 1 file:  zoom grid, all layers
    plot_full_grid_per_module(data_q, data_v) # 2 files: full grid per module

    print("\nDone. Outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()