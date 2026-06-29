# E3 Subspace Analysis Guide

This guide explains how to run the E3 subspace analyses from the LoRA paper (Hu et al. 2021, §7.2-7.3) using your Qwen3-1.7B-Base checkpoints.

## Prerequisites

Make sure you have the analysis dependencies installed:

```bash
pip install -e ".[dev]"
```

This installs `matplotlib` and `scipy` which are required for the analysis scripts.

## Fixed Issues (Latest Update)

The script has been updated to fix several issues:

1. **BFloat16 compatibility**: Automatically converts BFloat16 checkpoints to float32 for scipy compatibility
2. **Filename overwriting**: When analyzing multiple module types, each gets its own output file (e.g., `E3a_q_proj.png`, `E3a_v_proj.png`)
3. **Colormap**: Updated to grey-red-purple diverging colormap matching the paper's Fig. 3
4. **Cross-rank analysis**: New `cross-rank` mode to compare different ranks (r=8 vs r=64) as shown in paper Fig. 3

## E3a: Cross-Rank Subspace Similarity (Paper Fig. 3)

This analysis compares the LoRA subspaces learned at different ranks (e.g., r=8 vs r=64) to show whether low-rank and high-rank adapters learn similar subspaces.

### Command

```bash
python scripts/analyze_subspace.py \
  --checkpoint-dir checkpoints/E2_r64_attention \
  --checkpoint-dir-2 checkpoints/E2_r2_attention \
  --analysis cross-rank \
  --module-types q_proj v_proj \
  --output results/E3/E3a_cross_rank_r2_vs_r32.png \
  --rank-low 2 --rank-high 32
```

**Note:** Adjust the checkpoint paths and rank values based on your available checkpoints. The script expects:
- `--checkpoint-dir`: High-rank checkpoint (e.g., r=32 or r=64)
- `--checkpoint-dir-2`: Low-rank checkpoint (e.g., r=2 or r=8)

### Output

- **PNG file**: Heatmap showing Grassmann similarity between low-rank and high-rank A matrices for each layer
- **JSON file**: Raw similarity matrices and per-layer statistics

### Interpretation

High similarity (>0.8) indicates that even low-rank LoRA captures the same subspace directions as high-rank LoRA, supporting the paper's finding that the optimal subspace is low-dimensional.

## E3a: Layer-to-Layer Grassmann Similarity (Within Same Rank)

This analysis shows the similarity between LoRA subspaces across different layers within the same checkpoint.

### Command

```bash
python scripts/analyze_subspace.py \
  --checkpoint-dir checkpoints/E1d_attention \
  --analysis grassmann \
  --module-types q_proj v_proj k_proj o_proj \
  --output results/E3/E3a_layer_similarity.png
```

### Output

When analyzing multiple module types, the script automatically creates separate files:
- `E3a_layer_similarity_q_proj.png`
- `E3a_layer_similarity_v_proj.png`
- `E3a_layer_similarity_k_proj.png`
- `E3a_layer_similarity_o_proj.png`

Each file shows a 4×4 heatmap of the representative layers (0, 9, 18, 27).

## E3b: Cross-Seed Subspace Similarity (Paper Fig. 4)

This analysis compares LoRA subspaces from two independent training runs with different random seeds to assess convergence consistency.

### Command

```bash
python scripts/analyze_subspace.py \
  --checkpoint-dir checkpoints/E2_r64_attention \
  --checkpoint-dir-2 checkpoints/E2_r64_attention_seed2 \
  --analysis cross-seed \
  --module-types q_proj v_proj \
  --output results/E3/E3b_cross_seed.png
```

### Output

- **PNG file**: Bar chart showing Grassmann similarity for each layer between the two seeds
- **JSON file**: Per-layer similarities, mean, and standard deviation

### Interpretation

High cross-seed similarity suggests that LoRA converges to similar solutions regardless of initialization, indicating a well-behaved optimization landscape.

## E3c: Amplification Factor Analysis (Paper Tab. 7)

This analysis computes the ratio of the LoRA update norm to the base weight norm projected onto the top-r singular subspace.

### Command

```bash
python scripts/analyze_subspace.py \
  --checkpoint-dir checkpoints/E2_r64_attention \
  --analysis amplification \
  --module-types q_proj v_proj \
  --base-model Qwen/Qwen3-1.7B-Base \
  --alpha 64 --rank 32 \
  --output results/E3/E3c_amplification_r32.json
```

### Output

- **JSON file**: Amplification factors for each layer and module type, plus summary statistics (mean, std, min, max)

### Interpretation

Amplification factors >1 indicate that LoRA updates are larger than what would be expected from the top singular components of the base weights, suggesting LoRA learns beyond just scaling existing directions.

## Available Checkpoints

Your available checkpoints for E3 analysis:

### E1 Checkpoints (different module configurations)
- `checkpoints/E1a_q_proj` (r=14)
- `checkpoints/E1b_v_proj` (r=19)
- `checkpoints/E1c_qv_proj` (r=8)
- `checkpoints/E1d_attention` (r=4, all attention)
- `checkpoints/E1e_all_linear` (r=2, all 7 linear layers)

### E2 Checkpoints (rank sweep, attention-only)
- `checkpoints/E2_r1_attention` (r=1)
- `checkpoints/E2_r2_attention` (r=2)
- `checkpoints/E2_r8_attention` (r=8)
- `checkpoints/E2_r16_attention` (r=16)
- `checkpoints/E2_r32_attention` (r=32)
- `checkpoints/E2_r64_attention` (r=64)
- `checkpoints/E2_r64_attention_seed2` (r=64, second seed)

### Recommended E3 Analyses

1. **Cross-rank (E3a)**: Compare r=2 vs r=32 or r=2 vs r=64
   ```bash
   python scripts/analyze_subspace.py \
     --checkpoint-dir checkpoints/E2_r32_attention \
     --checkpoint-dir-2 checkpoints/E2_r2_attention \
     --analysis cross-rank \
     --module-types q_proj v_proj \
     --output results/E3/E3a_cross_rank_r2_vs_r32.png \
     --rank-low 2 --rank-high 32
   ```

2. **Cross-seed (E3b)**: Compare two r=64 seeds
   ```bash
   python scripts/analyze_subspace.py \
     --checkpoint-dir checkpoints/E2_r64_attention \
     --checkpoint-dir-2 checkpoints/E2_r64_attention_seed2 \
     --analysis cross-seed \
     --module-types q_proj v_proj \
     --output results/E3/E3b_cross_seed.png
   ```

3. **Amplification (E3c)**: Analyze r=32 checkpoint
   ```bash
   python scripts/analyze_subspace.py \
     --checkpoint-dir checkpoints/E2_r32_attention \
     --analysis amplification \
     --module-types q_proj v_proj \
     --base-model Qwen/Qwen3-1.7B-Base \
     --alpha 64 --rank 32 \
     --output results/E3/E3c_amplification_r32.json
   ```

## Troubleshooting

### Error: "Got unsupported ScalarType BFloat16"

This should now be automatically fixed. The script converts BFloat16 checkpoints to float32 on load.

### Error: "Adapter not found"

The script expects `adapter.pt` in each checkpoint directory. If your checkpoints use a different filename (e.g., `adapter_model.safetensors`), you'll need to either:
1. Rename the file to `adapter.pt`
2. Modify the `load_adapter_weights()` function in the script

### Out of memory when loading base model

The amplification analysis loads the full base model in float32. If you run out of memory:
1. Close other GPU applications
2. Use a machine with more RAM (CPU loading, not GPU)
3. The base model loading is CPU-only, so GPU memory shouldn't be a bottleneck

## Colormap

The script uses a custom grey-red-purple diverging colormap that matches the LoRA paper's Fig. 3:
- **Grey** (#808080): Low similarity (0.0)
- **Red** (#FF6B6B): Mid similarity (0.5)
- **Dark Purple** (#4B0082): High similarity (1.0)

This replaced the default "viridis" colormap to better match the paper's visual style.
