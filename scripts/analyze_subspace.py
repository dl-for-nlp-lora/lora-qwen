#!/usr/bin/env python3
"""E3 subspace analyses for the LoRA reproduction.

- E3a: Grassmann similarity heatmap between layers
- E3b: Cross-seed subspace similarity
- E3c: Amplification factor analysis

Run:
    # E3a: Grassmann similarity heatmap
    python scripts/analyze_subspace.py \\
      --checkpoint-dir checkpoints/E2_r64 \\
      --analysis grassmann \\
      --module-types q_proj v_proj \\
      --output results/E3a_grassmann_heatmap.png

    # E3b: Cross-seed similarity (requires two checkpoints with different seeds)
    python scripts/analyze_subspace.py \\
      --checkpoint-dir checkpoints/E2_r64_seed1 \\
      --checkpoint-dir-2 checkpoints/E2_r64_seed2 \\
      --analysis cross-seed \\
      --module-types q_proj v_proj \\
      --output results/E3b_cross_seed.png

    # E3c: Amplification factor
    python scripts/analyze_subspace.py \\
      --checkpoint-dir checkpoints/E2_r64 \\
      --analysis amplification \\
      --module-types q_proj v_proj \\
      --output results/E3c_amplification.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.linalg import subspace_angles

from subspace_plots import (
    plot_cross_seed,
    plot_amplification_table,
    plot_grassmann_heatmap,
)


def load_adapter_weights(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load LoRA A and B matrices from saved adapter."""
    adapter_path = checkpoint_dir / "adapter.pt"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")
    state_dict = torch.load(adapter_path, map_location="cpu", weights_only=True)
    # float32 so downstream SVD / scipy calls accept the tensors
    return {k: v.float() for k, v in state_dict.items()}


def load_base_model_weights(model_name: str):
    """Load base model weights for amplification factor analysis."""
    from transformers import AutoModelForCausalLM
    print(f"Loading base model '{model_name}' (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    return model


def group_lora_weights_by_module(
    state_dict: dict[str, torch.Tensor], module_type: str
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Group LoRA weights by module type into {layer_idx: (A, B)}.

    Keys look like model.layers.{i}.self_attn.{module_type}.lora_A / lora_B.
    """
    lora_weights = {}

    for key, tensor in state_dict.items():
        parts = key.split(".")
        if len(parts) < 6:
            continue

        try:
            layer_idx = int(parts[2])
            module_name = parts[4]

            if module_name != module_type:
                continue

            if "lora_A" in key:
                if layer_idx not in lora_weights:
                    lora_weights[layer_idx] = [None, None]
                lora_weights[layer_idx][0] = tensor
            elif "lora_B" in key:
                if layer_idx not in lora_weights:
                    lora_weights[layer_idx] = [None, None]
                lora_weights[layer_idx][1] = tensor
        except (ValueError, IndexError):
            continue

    # keep only layers that have both A and B
    result = {}
    for idx, (A, B) in lora_weights.items():
        if A is not None and B is not None:
            result[idx] = (A, B)

    return result


def get_base_weight(
    model, layer_idx: int, module_type: str
) -> torch.Tensor | None:
    """Return the base weight for a layer/module, or None if not found.

    Tries several submodule paths since the decoder may be nested differently
    depending on whether the model is a causal-LM wrapper or a bare decoder.
    """
    candidate_paths = [
        f"model.layers.{layer_idx}.self_attn.{module_type}",
        f"layers.{layer_idx}.self_attn.{module_type}",
        f"model.model.layers.{layer_idx}.self_attn.{module_type}",
    ]
    for path in candidate_paths:
        try:
            module = model.get_submodule(path)
        except AttributeError:
            continue
        return module.weight.data
    return None


def compute_grassmann_similarity(A1: torch.Tensor, A2: torch.Tensor) -> float:
    """Grassmann similarity in [0, 1] between the subspaces spanned by A1 and A2
    (shape (r, d)), 1 means identical subspaces."""
    A1_np = A1.detach().cpu().numpy()
    A2_np = A2.detach().cpu().numpy()

    try:
        angles = subspace_angles(A1_np.T, A2_np.T)
        return float(np.mean(np.cos(angles) ** 2))
    except ValueError:
        # rank-deficient matrices
        return 0.0


def compute_similarity_grid(
    A1: torch.Tensor, A2: torch.Tensor, max_k: int | None = None
) -> np.ndarray:
    """Normalized similarity grid between the top directions of two LoRA-A runs.

    Returns a (max_k, max_k) grid where entry (i-1, j-1) is the similarity in
    [0, 1] between A1's top-i and A2's top-j directions (defaults to all r).
    """
    A1_np = A1.detach().cpu().numpy().astype(np.float64)
    A2_np = A2.detach().cpu().numpy().astype(np.float64)

    # right singular vectors, ordered by singular value
    _, _, Vt1 = np.linalg.svd(A1_np, full_matrices=False)
    _, _, Vt2 = np.linalg.svd(A2_np, full_matrices=False)

    r = min(Vt1.shape[0], Vt2.shape[0])
    max_k = r if max_k is None else min(max_k, r)

    M = Vt1[:max_k] @ Vt2[:max_k].T

    # squared overlap accumulated over top-i/top-j, normalised by min(i, j)
    cumsum = np.cumsum(np.cumsum(M**2, axis=0), axis=1)
    i_idx = np.arange(1, max_k + 1).reshape(-1, 1)
    j_idx = np.arange(1, max_k + 1).reshape(1, -1)
    grid = cumsum / np.minimum(i_idx, j_idx)
    return grid


def compute_amplification_factor(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    base_W: torch.Tensor,
    alpha: float,
    r: int,
) -> dict[str, float]:
    """Amplification factor of the LoRA update for one weight matrix.

    Builds ΔW = (α/r)·B·A and measures how large it is relative to the part of W
    living in ΔW's leading directions. W is also projected onto its own top-r
    subspace and a random subspace for comparison, and the factor is computed
    over the top-k directions of ΔW for several k.

    Returns:
        dict with the full-r norms plus a "topk" sub-dict per k.
    """
    delta_W = (alpha / r) * (lora_B @ lora_A)

    def proj_norm(U_r: torch.Tensor, Vt_r: torch.Tensor) -> float:
        return torch.norm(U_r.T @ base_W @ Vt_r.T, p="fro").item()

    # Sd already includes the α/r scaling
    Ud, Sd, Vtd = torch.linalg.svd(delta_W, full_matrices=False)
    proj_delta = proj_norm(Ud[:, :r], Vtd[:r, :])

    Uw, _, Vtw = torch.linalg.svd(base_W, full_matrices=False)
    proj_w = proj_norm(Uw[:, :r], Vtw[:r, :])

    # random orthonormal subspace as a baseline
    gen = torch.Generator(device=base_W.device).manual_seed(0)
    rand_out = torch.randn(base_W.shape[0], r, generator=gen,
                           device=base_W.device, dtype=base_W.dtype)
    rand_in = torch.randn(base_W.shape[1], r, generator=gen,
                          device=base_W.device, dtype=base_W.dtype)
    q_out, _ = torch.linalg.qr(rand_out)
    q_in, _ = torch.linalg.qr(rand_in)
    proj_rand = proj_norm(q_out, q_in.T)

    delta_norm = torch.norm(delta_W, p="fro").item()
    w_norm = torch.norm(base_W, p="fro").item()

    amplification = float("inf") if proj_delta < 1e-10 else delta_norm / proj_delta

    # same factor restricted to ΔW's leading k directions
    sd_sq = (Sd**2).cpu()
    topk: dict[str, dict[str, float]] = {}
    for k in [kk for kk in (1, 2, 4, 8, 16, 32, r) if kk <= r]:
        delta_k_norm = float(torch.sqrt(sd_sq[:k].sum()))
        proj_delta_k = proj_norm(Ud[:, :k], Vtd[:k, :])
        proj_rand_k = proj_norm(q_out[:, :k], q_in[:, :k].T)
        amp_k = float("inf") if proj_delta_k < 1e-10 else delta_k_norm / proj_delta_k
        topk[str(k)] = {
            "amplification": amp_k,
            "delta_norm": delta_k_norm,
            "proj_onto_delta_subspace": proj_delta_k,
            "proj_onto_random_subspace": proj_rand_k,
        }

    return {
        "amplification": amplification,
        "delta_norm": delta_norm,
        "W_norm": w_norm,
        "proj_onto_delta_subspace": proj_delta,
        "proj_onto_W_subspace": proj_w,
        "proj_onto_random_subspace": proj_rand,
        "topk": topk,
    }


def analyze_grassmann(
    state_dict: dict[str, torch.Tensor],
    module_types: list[str],
    output_path: Path,
    n_layers: int = 28,
    representative_layers: list[int] | None = None,
) -> dict[str, Any]:
    """
    E3a: Compute Grassmann similarity heatmap between layers.

    Returns dict with similarity matrices for each module type.
    """
    if representative_layers is None:
        # evenly spaced across model depth
        representative_layers = [0, 9, 18, 27]

    all_data = {}

    for module_type in module_types:
        print(f"  Processing {module_type}...")
        lora_weights = group_lora_weights_by_module(state_dict, module_type)

        if not lora_weights:
            print(f"    Warning: No weights found for {module_type}")
            continue

        layer_indices = sorted(lora_weights.keys())
        n_actual = len(layer_indices)

        if n_actual == 0:
            continue

        similarity_matrix = np.zeros((n_actual, n_actual))
        for i, layer_i in enumerate(layer_indices):
            for j, layer_j in enumerate(layer_indices):
                A_i, _ = lora_weights[layer_i]
                A_j, _ = lora_weights[layer_j]
                similarity_matrix[i, j] = compute_grassmann_similarity(A_i, A_j)

        # Extract representative layers if we have enough
        if n_actual >= len(representative_layers):
            rep_indices = [
                layer_indices.index(l) if l in layer_indices else 0
                for l in representative_layers
            ]
            rep_matrix = similarity_matrix[np.ix_(rep_indices, rep_indices)]
            layer_names = [f"L{l}" for l in representative_layers]
        else:
            rep_matrix = similarity_matrix
            layer_names = [f"L{l}" for l in layer_indices]

        if len(module_types) > 1:
            output_path = output_path.parent / f"{output_path.stem}_{module_type}{output_path.suffix}"

        plot_grassmann_heatmap(rep_matrix, layer_names, module_type, output_path)

        all_data[module_type] = {
            "similarity_matrix": rep_matrix.tolist(),
            "layers": layer_names,
        }

    return all_data


def analyze_cross_seed(
    state_dict_1: dict[str, torch.Tensor],
    state_dict_2: dict[str, torch.Tensor],
    module_types: list[str],
    output_path: Path,
) -> dict[str, Any]:
    """E3b: cross-seed subspace similarity.

    For each module, compares the two seeds' LoRA-A directions and produces the
    layer-averaged similarity grid, per-layer top-k similarity (k = 1, 2, 4, 8),
    and the full-subspace mean.
    """
    all_data = {}
    grid_cap = 16  # how many top directions to show in the heatmap
    top_ks = [1, 2, 4, 8]

    for module_type in module_types:
        print(f"  Processing {module_type}...")
        weights_1 = group_lora_weights_by_module(state_dict_1, module_type)
        weights_2 = group_lora_weights_by_module(state_dict_2, module_type)

        if not weights_1 or not weights_2:
            print(f"    Warning: Missing weights for {module_type}")
            continue

        layer_indices = sorted(weights_1.keys())

        per_layer: dict[str, dict[str, float]] = {}
        grid_sum = None
        n_grids = 0

        for layer_idx in layer_indices:
            if layer_idx not in weights_2:
                continue
            A_1, _ = weights_1[layer_idx]
            A_2, _ = weights_2[layer_idx]

            grid = compute_similarity_grid(A_1, A_2)
            r_eff = grid.shape[0]

            # accumulate the top-left block for the averaged heatmap
            cap = min(grid_cap, r_eff)
            block = grid[:cap, :cap]
            grid_sum = block if grid_sum is None else grid_sum + block
            n_grids += 1

            # diagonal entries are the top-k self-similarities
            per_layer[f"L{layer_idx}"] = {
                f"top{k}": float(grid[k - 1, k - 1])
                for k in top_ks
                if k <= r_eff
            }
            per_layer[f"L{layer_idx}"]["full"] = float(grid[-1, -1])

        if grid_sum is None:
            print(f"    Warning: no overlapping layers for {module_type}")
            continue

        avg_grid = grid_sum / n_grids

        def _mean_over_layers(key: str) -> float:
            vals = [d[key] for d in per_layer.values() if key in d]
            return float(np.mean(vals)) if vals else float("nan")

        # random-chance baseline for the top-k similarity is k/d
        d_in = int(A_1.shape[1])
        r_full = int(A_1.shape[0])
        topk_means = {f"top{k}": _mean_over_layers(f"top{k}") for k in top_ks}
        full_mean = _mean_over_layers("full")
        baseline_per_k = {f"top{k}": k / d_in for k in top_ks}
        signal_ratio = {
            f"top{k}": (topk_means[f"top{k}"] / baseline_per_k[f"top{k}"])
            if baseline_per_k[f"top{k}"] > 0 else float("nan")
            for k in top_ks
        }

        # Per-module filename so multiple module types don't overwrite.
        if len(module_types) > 1:
            module_output = (
                output_path.parent
                / f"{output_path.stem}_{module_type}{output_path.suffix}"
            )
        else:
            module_output = output_path

        plot_cross_seed(
            avg_grid, per_layer, top_ks, baseline_per_k, module_type, module_output
        )

        all_data[module_type] = {
            "per_layer": per_layer,
            "topk_means": topk_means,
            "random_baseline_per_k": baseline_per_k,
            "signal_ratio_over_chance": signal_ratio,
            "full_subspace_mean": full_mean,
            "random_floor_full": float(r_full / d_in),
            "avg_grid": avg_grid.tolist(),
        }

        print(
            "    Mean top-1={top1:.3f} | top-2={top2:.3f} | top-4={top4:.3f} | "
            "top-8={top8:.3f}".format(**topk_means)
        )
        print(
            "    Signal vs chance: top-1={top1:.0f}x | top-2={top2:.0f}x | "
            "top-4={top4:.0f}x | top-8={top8:.0f}x".format(**signal_ratio)
        )
        print(
            f"    Full-subspace mean={full_mean:.3f} "
            f"(full random floor r/d={r_full / d_in:.3f})"
        )

    return all_data


def analyze_amplification(
    state_dict: dict[str, torch.Tensor],
    module_types: list[str],
    base_model,
    output_path: Path,
    alpha: float = 128,
    r: int = 64,
) -> dict[str, Any]:
    """E3c: amplification factor analysis.

    Per layer and module, computes the amplification factor and the comparison
    projections, then saves the JSON and plot.
    """
    all_data = {}

    column_keys = [
        "delta_norm",
        "W_norm",
        "proj_onto_delta_subspace",
        "proj_onto_W_subspace",
        "proj_onto_random_subspace",
    ]

    for module_type in module_types:
        print(f"  Processing {module_type}...")
        lora_weights = group_lora_weights_by_module(state_dict, module_type)

        if not lora_weights:
            print(f"    Warning: No weights found for {module_type}")
            continue

        factors = {}
        rows = {}
        for layer_idx, (A, B) in lora_weights.items():
            base_W = get_base_weight(base_model, layer_idx, module_type)
            if base_W is None:
                print(f"    Warning: No base weight for layer {layer_idx}")
                continue

            row = compute_amplification_factor(A, B, base_W, alpha, r)
            rows[f"layer_{layer_idx}"] = row
            factors[f"layer_{layer_idx}"] = row["amplification"]

        if not factors:
            print(
                f"    Warning: no base weights resolved for {module_type}; "
                "skipping (check --base-model matches the trained model)."
            )
            continue

        amp_values = list(factors.values())
        column_means = {
            key: float(np.mean([rows[layer][key] for layer in rows]))
            for key in column_keys
        }

        # Average the top-k amplification sweep across layers.
        k_values = sorted(
            {int(k) for row in rows.values() for k in row["topk"]}
        )
        topk_amplification_means = {
            str(k): float(np.mean([
                rows[layer]["topk"][str(k)]["amplification"]
                for layer in rows if str(k) in rows[layer]["topk"]
            ]))
            for k in k_values
        }

        all_data[module_type] = {
            "factors": factors,
            "mean": float(np.mean(amp_values)),
            "std": float(np.std(amp_values)),
            "min": float(min(amp_values)),
            "max": float(max(amp_values)),
            "table7_means": column_means,
            "topk_amplification_means": topk_amplification_means,
            "per_layer_rows": rows,
        }

        print(f"    Mean amplification (full r={r}): {all_data[module_type]['mean']:.3f}")
        print(f"    Std: {all_data[module_type]['std']:.3f}")
        print(
            "    Mean ‖ΔW‖_F={delta_norm:.3f} | "
            "‖UᵀWVᵀ‖_F (ΔW-subspace)={proj_onto_delta_subspace:.3f} | "
            "(W-subspace)={proj_onto_W_subspace:.3f} | "
            "(random)={proj_onto_random_subspace:.3f}".format(**column_means)
        )
        print(
            "    Amplification by top-k: "
            + " | ".join(
                f"k={k}:{v:.2f}x" for k, v in topk_amplification_means.items()
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved amplification factors to {output_path}")

    plot_amplification_table(all_data, output_path.with_suffix(".png"))

    return all_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E3 subspace analysis")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing adapter.pt from training",
    )
    parser.add_argument(
        "--checkpoint-dir-2",
        type=Path,
        help="Second checkpoint directory (for cross-seed analysis)",
    )
    parser.add_argument(
        "--analysis",
        choices=["grassmann", "cross-seed", "amplification"],
        required=True,
        help="Type of analysis to perform",
    )
    parser.add_argument(
        "--module-types",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Module types to analyze (e.g., q_proj, v_proj, gate_proj)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path (PNG for plots, JSON for data)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
        help="Base model name for amplification factor analysis",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=128,
        help="LoRA alpha value (for amplification factor)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="LoRA rank r (for amplification factor)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[1/3] Loading adapter weights from {args.checkpoint_dir}...")
    state_dict = load_adapter_weights(args.checkpoint_dir)
    print(f"      Loaded {len(state_dict)} tensors")

    if args.analysis == "grassmann":
        print(f"\n[2/3] Computing Grassmann similarity heatmap...")
        data = analyze_grassmann(
            state_dict,
            args.module_types,
            args.output,
            representative_layers=[0, 9, 18, 27],
        )

        # Also save raw data as JSON
        json_path = args.output.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved raw data to {json_path}")

    elif args.analysis == "cross-seed":
        if not args.checkpoint_dir_2:
            print("ERROR: --checkpoint-dir-2 required for cross-seed analysis")
            return 1

        print(f"\n[2/3] Loading second checkpoint from {args.checkpoint_dir_2}...")
        state_dict_2 = load_adapter_weights(args.checkpoint_dir_2)
        print(f"      Loaded {len(state_dict_2)} tensors")

        print(f"\n[3/3] Computing cross-seed similarity...")
        data = analyze_cross_seed(
            state_dict, state_dict_2, args.module_types, args.output
        )

        # Save raw data
        json_path = args.output.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved raw data to {json_path}")

    elif args.analysis == "amplification":
        print(f"\n[2/3] Loading base model for amplification analysis...")
        base_model = load_base_model_weights(args.base_model)

        print(f"\n[3/3] Computing amplification factors (alpha={args.alpha}, r={args.rank})...")
        data = analyze_amplification(
            state_dict,
            args.module_types,
            base_model,
            args.output,
            alpha=args.alpha,
            r=args.rank,
        )

    print("\nAnalysis completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
