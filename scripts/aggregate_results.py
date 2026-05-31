#!/usr/bin/env python3
"""Aggregate experiment results into tables and figures for LoRA paper reproduction.

Reads JSON result files from GSM8K evaluations and produces:
- Summary tables (replicating paper Tab. 5, Tab. 6)
- Rank vs accuracy plots (replicating paper Fig. 2)
- Markdown/JSON summaries for easy reference

Run:
    # Generate all tables and figures
    python scripts/aggregate_results.py \\
      --results-dir results \\
      --output-dir results/summary

    # Generate specific table only
    python scripts/aggregate_results.py \\
      --results-dir results \\
      --table E1 \\
      --output results/E1_table.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Figures will not be generated.")


def load_experiment_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Load all JSON result files from results directory.

    Returns dict mapping experiment ID to result data.
    """
    results = {}

    for json_file in results_dir.glob("*.json"):
        # Skip summary files
        if "summary" in str(json_file):
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            # Extract experiment ID from filename or metadata
            exp_id = json_file.stem

            # Try to extract from metadata if available
            if "lora_config" in data:
                config_path = Path(data["lora_config"])
                if config_path.name != "all_linear.yaml":
                    # Use config name as experiment ID
                    exp_id = config_path.stem

            results[exp_id] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results


def extract_accuracy(result_data: dict[str, Any]) -> float | None:
    """Extract GSM8K accuracy from result JSON."""
    try:
        # For base vs FT comparison files
        if "ft" in result_data:
            return result_data["ft"]["accuracy"]
        # For single model eval
        if "accuracy" in result_data:
            return result_data["accuracy"]
    except (KeyError, TypeError):
        pass
    return None


def create_e1_table(results: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """
    Generate table for E1 (which matrices to adapt).

    Replicates paper Tab. 5 layout.
    """
    configs = {
        "A-base": {
            "targets": "(none — base)",
            "rank": "—",
            "trainable": "0",
            "result_key": "gsm8k_base_vs_ft",
        },
        "E0b_full_ft": {
            "targets": "(all params)",
            "rank": "—",
            "trainable": "1.7B",
            "result_key": None,  # Not yet implemented
        },
        "E1a_q_proj": {
            "targets": "q_proj",
            "rank": "16",
            "trainable": "~18M",
            "result_key": None,
        },
        "E1b_v_proj": {
            "targets": "v_proj",
            "rank": "16",
            "trainable": "~18M",
            "result_key": None,
        },
        "E1c_qv_proj": {
            "targets": "q_proj, v_proj",
            "rank": "8",
            "trainable": "~18M",
            "result_key": None,
        },
        "E1d_attention": {
            "targets": "q, k, v, o_proj",
            "rank": "4",
            "trainable": "~18M",
            "result_key": None,
        },
        "E1e_all_linear": {
            "targets": "all 7 linear",
            "rank": "2",
            "trainable": "~18M",
            "result_key": None,
        },
    }

    rows = []
    for exp_id, config in configs.items():
        accuracy = None
        result_key = config.get("result_key") or exp_id

        # Try to find matching result
        for key, data in results.items():
            if result_key in key or key.startswith(exp_id.split("_")[0]):
                acc = extract_accuracy(data)
                if acc is not None:
                    accuracy = acc
                    break

        rows.append(
            {
                "config": exp_id.replace("_", " "),
                "targets": config["targets"],
                "rank": config["rank"],
                "trainable": config["trainable"],
                "accuracy": f"{accuracy:.3f}" if accuracy is not None else "N/A",
            }
        )

    return rows


def create_e2_table(results: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """
    Generate table for E2 (rank sweep).

    Replicates paper Tab. 6 layout.
    """
    ranks = [1, 2, 4, 8, 16, 64]
    rows = []

    # Get base accuracy for delta calculation
    base_acc = None
    for key, data in results.items():
        if "base" in key or "A-base" in key:
            base_acc = extract_accuracy(data)
            if base_acc is not None:
                break

    for r in ranks:
        exp_id = f"E2_r{r}"
        accuracy = None

        # Try to find matching result
        for key, data in results.items():
            if exp_id in key or f"r{r}" in key:
                acc = extract_accuracy(data)
                if acc is not None:
                    accuracy = acc
                    break

        # Calculate delta from base
        delta = None
        if accuracy is not None and base_acc is not None:
            delta = accuracy - base_acc

        # Trainable params estimate (approximate)
        # For Qwen3-1.7B with 28 layers, ~7 linear modules per layer
        # Each LoRA param: r × (d_in + d_out) per module
        trainable_m = (
            r * 1.14
        )  # Rough estimate, actual depends on module dimensions

        rows.append(
            {
                "config": f"E2_r{r}",
                "rank": str(r),
                "trainable": f"~{trainable_m:.1f}M",
                "accuracy": f"{accuracy:.3f}" if accuracy is not None else "N/A",
                "delta": f"{delta:+.3f}" if delta is not None else "N/A",
            }
        )

    return rows


def create_rank_accuracy_plot(
    results: dict[str, dict[str, Any]], output_path: Path
) -> None:
    """
    Generate rank vs accuracy plot (paper Fig. 2 style).

    X-axis: log₂(rank)
    Y-axis: GSM8K accuracy
    """
    if not HAS_MATPLOTLIB:
        print("Skipping rank-accuracy plot (matplotlib not available)")
        return

    ranks = [1, 2, 4, 8, 16, 64]
    accuracies = []

    for r in ranks:
        exp_id = f"E2_r{r}"
        acc = None

        for key, data in results.items():
            if exp_id in key or f"r{r}" in key:
                acc = extract_accuracy(data)
                if acc is not None:
                    break

        accuracies.append(acc)

    # Filter out None values
    valid_data = [(r, a) for r, a in zip(ranks, accuracies) if a is not None]
    if not valid_data:
        print("No E2 results found for rank-accuracy plot")
        return

    ranks_plot, accs_plot = zip(*valid_data)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        np.log2(ranks_plot), accs_plot, marker="o", linestyle="-", linewidth=2, markersize=8
    )

    ax.set_xlabel("log₂(Rank)", fontsize=12)
    ax.set_ylabel("GSM8K Accuracy", fontsize=12)
    ax.set_title("LoRA Rank vs Accuracy (Qwen3-1.7B)", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add data point labels
    for r, acc in zip(ranks_plot, accs_plot):
        ax.annotate(
            f"r={r}",
            (np.log2(r), acc),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved rank-accuracy plot to {output_path}")
    plt.close()


def format_table(rows: list[dict[str, str]], title: str = "") -> str:
    """Format a list of dicts as a markdown table."""
    if not rows:
        return ""

    # Get column headers from first row
    headers = list(rows[0].keys())

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, h in enumerate(headers):
            widths[i] = max(widths[i], len(str(row.get(h, ""))))

    # Build table
    lines = []
    if title:
        lines.append(f"### {title}")
        lines.append("")

    # Header row
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)

    # Separator
    sep_line = "-+-".join("-" * w for w in widths)
    lines.append(sep_line)

    # Data rows
    for row in rows:
        data_line = " | ".join(
            str(row.get(h, "")).ljust(widths[i]) for i, h in enumerate(headers)
        )
        lines.append(data_line)

    return "\n".join(lines)


def save_table_markdown(
    rows: list[dict[str, str]], output_path: Path, title: str = ""
) -> None:
    """Save table as markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_md = format_table(rows, title)
    with open(output_path, "w") as f:
        f.write(table_md)
        f.write("\n")
    print(f"Saved table to {output_path}")


def save_table_json(
    rows: list[dict[str, Any]], output_path: Path
) -> None:
    """Save table data as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved table data to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/summary"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--table",
        choices=["E1", "E2", "all"],
        default="all",
        help="Which table(s) to generate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Single output file (overrides --output-dir)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[1/3] Loading experiment results from {args.results_dir}...")
    results = load_experiment_results(args.results_dir)
    print(f"      Loaded {len(results)} result files")

    if not results:
        print("Warning: No results found. Run experiments first.")
        return 1

    # Determine output directory
    output_dir = args.output_dir
    if args.output:
        output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate E1 table
    if args.table in ["E1", "all"]:
        print(f"\n[2/3] Generating E1 table (which matrices to adapt)...")
        e1_rows = create_e1_table(results)

        if args.output and args.table == "E1":
            output_path = args.output
        else:
            output_path = output_dir / "E1_table.md"

        save_table_markdown(e1_rows, output_path, title="E1: Which Matrices to Adapt")

        # Also save as JSON
        json_path = output_path.with_suffix(".json")
        save_table_json(e1_rows, json_path)

        # Print to console
        print("\n" + format_table(e1_rows, "E1: Which Matrices to Adapt"))

    # Generate E2 table and plot
    if args.table in ["E2", "all"]:
        print(f"\n[3/3] Generating E2 table and plot (rank sweep)...")
        e2_rows = create_e2_table(results)

        if args.output and args.table == "E2":
            output_path = args.output
        else:
            output_path = output_dir / "E2_table.md"

        save_table_markdown(e2_rows, output_path, title="E2: Optimal Rank (r)")

        # Save as JSON
        json_path = output_path.with_suffix(".json")
        save_table_json(e2_rows, json_path)

        # Print to console
        print("\n" + format_table(e2_rows, "E2: Optimal Rank (r)"))

        # Generate plot
        if HAS_MATPLOTLIB:
            plot_path = output_dir / "E2_rank_vs_accuracy.png"
            create_rank_accuracy_plot(results, plot_path)

    print("\nAggregation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
