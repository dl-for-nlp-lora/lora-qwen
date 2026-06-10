#!/usr/bin/env python3
"""E3 subspace analysis scripts for LoRA paper reproduction.

Implements the subspace analyses from Hu et al. 2021, §7.2-7.3:

- E3a: Grassmann similarity heatmap between layers (paper Fig. 3)
- E3b: Cross-seed subspace similarity (paper Fig. 4)
- E3c: Amplification factor analysis (paper Tab. 7)

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

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import subspace_angles


def load_adapter_weights(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load LoRA A and B matrices from saved adapter."""
    adapter_path = checkpoint_dir / "adapter.pt"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")
    state_dict = torch.load(adapter_path, map_location="cpu", weights_only=True)
    # Convert BFloat16/Float16 to Float32 for scipy compatibility
    return {k: v.float() for k, v in state_dict.items()}


def load_base_model_weights(model_name: str):
    """Load base model weights for amplification factor analysis."""
    from transformers import AutoModelForCausalLM
    print(f"Loading base model '{model_name}' (this may take a minute)...")
    # Load in float32 for analysis compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32
    )
    return model


def group_lora_weights_by_module(
    state_dict: dict[str, torch.Tensor], module_type: str
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """
    Group LoRA weights by module type (e.g., all q_proj layers).

    Returns dict: {layer_idx: (A_matrix, B_matrix)}

    Key pattern: model.layers.{i}.self_attn.{module_type}.lora_A
    """
    lora_weights = {}

    for key, tensor in state_dict.items():
        # Parse key: model.layers.{i}.self_attn.{module_type}.lora_A/B
        parts = key.split(".")
        if len(parts) < 6:
            continue

        try:
            layer_idx = int(parts[2])  # model.layers.{i}
            attn_part = parts[3]  # self_attn
            module_name = parts[4]  # e.g., q_proj

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

    # Convert lists to tuples and filter incomplete entries
    result = {}
    for idx, (A, B) in lora_weights.items():
        if A is not None and B is not None:
            result[idx] = (A, B)

    return result


def get_base_weight(
    model, layer_idx: int, module_type: str
) -> torch.Tensor | None:
    """Extract base weight matrix for a specific layer and module type."""
    try:
        # Pattern: model.layers.{i}.self_attn.{module_type}.weight
        weight_path = f"layers.{layer_idx}.self_attn.{module_type}.weight"
        parts = weight_path.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        return module.weight.data
    except (AttributeError, IndexError):
        return None


def compute_grassmann_similarity(A1: torch.Tensor, A2: torch.Tensor) -> float:
    """
    Compute Grassmann similarity between two subspaces.

    Args:
        A1, A2: matrices of shape (r, d) spanning subspaces

    Returns:
        similarity in [0, 1] where 1 = identical subspaces

    Uses principal angles: similarity = mean(cos²(θ_k))
    """
    # Ensure matrices are 2D and in numpy format
    A1_np = A1.detach().cpu().numpy()
    A2_np = A2.detach().cpu().numpy()

    # Compute principal angles between subspaces
    # scipy.linalg.subspace_angles expects (d, r) shape
    try:
        angles = subspace_angles(A1_np.T, A2_np.T)
        similarity = float(np.mean(np.cos(angles) ** 2))
        return similarity
    except ValueError:
        # Fallback for rank-deficient matrices
        return 0.0


def compute_amplification_factor(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    base_W: torch.Tensor,
    alpha: float,
    r: int,
) -> float:
    """
    Compute amplification factor from paper §7.3.

    ΔW = (α/r) × B × A
    Amplification = ‖ΔW‖_F / ‖UᵀWVᵀ‖_F

    Where U, V are top-r singular vectors of W.

    Returns:
        Amplification factor (ratio of norms)
    """
    # Compute ΔW = (α/r) × B × A
    delta_W = (alpha / r) * (lora_B @ lora_A)

    # SVD of base weight W = U @ S @ Vt
    U, S, Vt = torch.linalg.svd(base_W, full_matrices=False)

    # Extract top-r singular vectors
    U_r = U[:, :r]  # (d_out, r)
    V_r = Vt[:r, :]  # (r, d_in)

    # Project base weight onto top-r subspace: U_r.T @ W @ V_r.T
    # This gives the r×r matrix of W in the top singular subspace
    W_projected = U_r.T @ base_W @ V_r.T

    # Compute Frobenius norms
    delta_norm = torch.norm(delta_W, p="fro")
    base_norm = torch.norm(W_projected, p="fro")

    if base_norm < 1e-10:
        # Avoid division by zero
        return float("inf") if delta_norm > 0 else 1.0

    return (delta_norm / base_norm).item()


def plot_grassmann_heatmap(
    similarity_matrix: np.ndarray,
    layer_names: list[str],
    module_type: str,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Create heatmap visualization like paper Fig. 3."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap with viridis colormap
    im = ax.imshow(
        similarity_matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto"
    )

    # Label axes with layer indices
    ax.set_xticks(range(len(layer_names)))
    ax.set_yticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, fontsize=10)
    ax.set_yticklabels(layer_names, fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Grassmann Similarity")
    cbar.ax.tick_params(labelsize=10)

    # Title and labels
    title = f"{module_type} - Inter-layer Subspace Similarity"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)

    # Add text annotations for values
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
        # Evenly spaced layers across the model depth
        representative_layers = [0, 9, 18, 27]

    all_data = {}

    for module_type in module_types:
        print(f"  Processing {module_type}...")
        lora_weights = group_lora_weights_by_module(state_dict, module_type)

        if not lora_weights:
            print(f"    Warning: No weights found for {module_type}")
            continue

        # Compute pairwise similarity matrix
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
            # Insert module_type into filename
            output_path = output_path.parent / f"{output_path.stem}_{module_type}{output_path.suffix}"

        # Plot heatmap
        plot_grassmann_heatmap(rep_matrix, layer_names, module_type, output_path)

        # Store data
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
    """
    E3b: Cross-seed subspace similarity analysis.

    Compare LoRA subspaces learned with different random seeds.
    """
    all_data = {}

    for module_type in module_types:
        print(f"  Processing {module_type}...")
        weights_1 = group_lora_weights_by_module(state_dict_1, module_type)
        weights_2 = group_lora_weights_by_module(state_dict_2, module_type)

        if not weights_1 or not weights_2:
            print(f"    Warning: Missing weights for {module_type}")
            continue

        # Compute cross-seed similarity for each layer
        layer_indices = sorted(weights_1.keys())
        similarities = []

        for layer_idx in layer_indices:
            if layer_idx not in weights_2:
                continue
            A_1, _ = weights_1[layer_idx]
            A_2, _ = weights_2[layer_idx]
            sim = compute_grassmann_similarity(A_1, A_2)
            similarities.append((layer_idx, sim))

        # Plot as bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        layers = [f"L{idx}" for idx, _ in similarities]
        sims = [sim for _, sim in similarities]

        bars = ax.bar(range(len(layers)), sims, color="steelblue")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Grassmann Similarity (Seed 1 vs Seed 2)", fontsize=12)
        ax.set_title(
            f"{module_type} - Cross-Seed Subspace Similarity (r=64)", fontsize=14
        )
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)

        # Add value labels on bars
        for bar, val in zip(bars, sims):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved cross-seed plot to {output_path}")
        plt.close()

        # Store data
        all_data[module_type] = {
            "similarities": {f"L{idx}": float(sim) for idx, sim in similarities},
            "mean_similarity": float(np.mean(sims)),
            "std_similarity": float(np.std(sims)),
        }

    return all_data


def analyze_amplification(
    state_dict: dict[str, torch.Tensor],
    module_types: list[str],
    base_model,
    output_path: Path,
    alpha: float = 128,
    r: int = 64,
) -> dict[str, Any]:
    """
    E3c: Amplification factor analysis.

    Computes ‖ΔW‖_F / ‖UᵀWVᵀ‖_F for each layer and module type.
    """
    all_data = {}

    for module_type in module_types:
        print(f"  Processing {module_type}...")
        lora_weights = group_lora_weights_by_module(state_dict, module_type)

        if not lora_weights:
            print(f"    Warning: No weights found for {module_type}")
            continue

        factors = {}
        for layer_idx, (A, B) in lora_weights.items():
            base_W = get_base_weight(base_model, layer_idx, module_type)
            if base_W is None:
                print(f"    Warning: No base weight for layer {layer_idx}")
                continue

            factor = compute_amplification_factor(A, B, base_W, alpha, r)
            factors[f"layer_{layer_idx}"] = factor

        all_data[module_type] = {
            "factors": factors,
            "mean": float(np.mean(list(factors.values()))),
            "std": float(np.std(list(factors.values()))),
            "min": float(min(factors.values())),
            "max": float(max(factors.values())),
        }

        # Print summary
        print(f"    Mean amplification: {all_data[module_type]['mean']:.3f}")
        print(f"    Std: {all_data[module_type]['std']:.3f}")

    # Save as JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved amplification factors to {output_path}")

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
