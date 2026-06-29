# E3 — subspace analyses

E3 looks *inside* the trained LoRA adapters to ask what they actually learned,
reproducing the subspace analyses from Hu et al. 2021 (§7.2–7.3). It is purely
post-hoc on existing `adapter.pt` checkpoints — no extra training, except that
E3b needs a second adapter trained with a different seed.

All analyses are driven by [`analyze_subspace.py`](analyze_subspace.py) (figure
rendering in [`subspace_plots.py`](subspace_plots.py)). Results for the Qwen2.5
checkpoints live under [`../results/E3/`](../results/E3/):

```
results/E3/
  e3a_output/   inter-layer + rank-vs-rank Grassmann (heatmaps + JSON)
  e3b/          cross-seed similarity (JSON + plots)
  e3c/          amplification factor for r=4 and r=64 (JSON + plots)
```

The three analyses share one idea: the LoRA update `ΔW = (α/r)·BA` only spans a
tiny `r`-dimensional **subspace**. Each analysis probes that subspace from a
different angle. Subspaces are compared with **principal angles**: the similarity
`φ = mean(cos²θ_k) ∈ [0, 1]` (1 = identical subspace, 0 = orthogonal), which is
invariant to how `A` is rotated/reordered.

## Common arguments

```
--checkpoint-dir     directory holding adapter.pt
--checkpoint-dir-2   second adapter (cross-seed only)
--analysis           grassmann | cross-seed | amplification
--module-types       e.g. q_proj v_proj   (default: q_proj v_proj)
--output             PNG for plots, JSON for data
--base-model         base weights for amplification (e.g. Qwen/Qwen2.5-1.5B)
--alpha --rank       scaling for the amplification ΔW reconstruction
```

## E3a — inter-layer Grassmann similarity (paper Fig. 3)

Do different layers learn the same correction, or specialize? Computes the
pairwise `φ` between the per-layer `A` subspaces.

```bash
python scripts/analyze_subspace.py \
  --checkpoint-dir checkpoints/headroom/qwen25_1.5b/e2_qv_r64 \
  --analysis grassmann \
  --module-types q_proj v_proj \
  --output results/E3/e3a_output/E3a_grassmann.png
```

**Reading.** Off-diagonal values near 0 (diagonal = 1) mean each layer's
subspace is almost orthogonal to the others — the adaptation is layer-specific,
not a redundant copy.

## E3b — cross-seed similarity (paper Fig. 4)

Is the learned subspace a real property of the task or an artifact of one random
init? Trains the *same* config twice with different seeds and compares the two
subspaces layer by layer, against the random-chance baseline `k/d`.

```bash
python scripts/analyze_subspace.py \
  --checkpoint-dir   checkpoints/headroom/qwen25_1.5b/e2_qv_r64 \
  --checkpoint-dir-2 checkpoints/headroom/qwen25_1.5b/e2_qv_r64_seed2 \
  --analysis cross-seed \
  --module-types q_proj v_proj \
  --output results/E3/e3b/e3b_cross_seed.png
```

**Reading.** The top-k directions sit clearly above the `k/d` chance line: the
two seeds rediscover the same leading directions, so the subspace is a stable,
reproducible feature of the task rather than noise. (This is the one analysis
that requires a second training run — a second-seed adapter.)

## E3c — amplification factor (paper Tab. 7)

Does LoRA boost directions the base model already emphasizes, or ones it had
suppressed? Compares `‖ΔW‖_F` against the base weight projected onto its own
top-`r` singular subspace, `‖UᵀWVᵀ‖_F`.

```bash
python scripts/analyze_subspace.py \
  --checkpoint-dir checkpoints/headroom/qwen25_1.5b/e2_qv_r64 \
  --analysis amplification \
  --module-types q_proj v_proj \
  --base-model Qwen/Qwen2.5-1.5B \
  --alpha 128 --rank 64 \
  --output results/E3/e3c/e3c_amplification_r64.json
```

**Reading (Qwen2.5 results).** The amplification factor is large at low rank and
shrinks as rank grows — mean `≈4.5×` at r=4 vs `≈1.0×` at r=64. The small-rank
adapter concentrates its update on a few task-relevant directions that the base
model under-weights; the high-rank adapter spreads the same total change over
many directions, so per-direction amplification falls toward 1. This matches the
paper's finding that LoRA amplifies task-specific, previously-suppressed
directions, and dovetails with E2 (a tiny rank already captures the useful
subspace).

## Notes

- Adapters are expected as `adapter.pt` (the custom backend's format). bf16
  checkpoints are auto-cast to fp32 for the SVD / principal-angle math.
- The amplification analysis loads the full base model in fp32 on CPU; it needs
  host RAM, not GPU memory.
