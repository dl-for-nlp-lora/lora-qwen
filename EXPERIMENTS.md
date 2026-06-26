# Experiments

Plan for what we reproduce from Hu et al. 2021 ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)), what we don't, and why.

## What the paper does


| §       | Experiment              | Models                         | Datasets              |
| ------- | ----------------------- | ------------------------------ | --------------------- |
| 5.2     | LoRA on NLU             | RoBERTa base/large (125M/355M) | GLUE (8 tasks)        |
| 5.3     | LoRA on big NLU         | DeBERTa XXL (1.5B)             | GLUE                  |
| 5.4     | LoRA on NLG             | GPT-2 medium/large (354M/774M) | E2E, WebNLG, DART     |
| 5.5     | Scale-up                | GPT-3 (175B)                   | WikiSQL, MNLI, SAMSum |
| 7.1     | Which weight matrices?  | GPT-3 (18M budget)             | WikiSQL, MNLI         |
| 7.2     | Optimal rank `r`?       | GPT-3 (+ GPT-2 in App. H.2)    | WikiSQL, MNLI         |
| 7.2/7.3 | Subspace / SVD analyses | GPT-3 checkpoints              | —                     |


## Out of scope (and why)

- **§5.5 GPT-3 175B** — 350 GB weights, multi-node cluster only.
- **§5.2 / §5.3 RoBERTa/DeBERTa on GLUE** — encoder + classification head. Would require ripping out the decoder-only Qwen3 pipeline we just built and replacing it with a different model family. Not a meaningful test of *our* LoRA implementation — it just tests `transformers`'s classifier head wrapper.
- **§5.4 GPT-2 NLG (E2E/WebNLG/DART)** — *could* run, but BLEU/METEOR on 2017-era surface-form NLG benchmarks adds little signal that GSM8K-style accuracy doesn't already give us, and would double compute.
- **Full §7.2 subspace analysis on 96 layers** — done in §7.2/H.1 on GPT-3's 96 decoder layers. Qwen3-1.7B has 28 layers; we run the same analysis but with `L=28` per Fig. 3/4.

## What we reproduce

Backbone: **Qwen3-1.7B-Base**, decoder-only, classical transformer (`q/k/v/o_proj` + `gate/up/down_proj`). Sits between GPT-2 medium (354M) and large (774M) in capacity, modern training data. LoRA implementation: in-house (`lora_qwen.lora.custom_backend`), verified against the `peft` reference (identity check + loss-trajectory + downstream accuracy parity within bf16 noise).

Downstream task: **MetaMathQA → GSM8K**.

- Not a paper dataset — GSM8K post-dates LoRA.
- Justification: paper datasets target encoders or 2017 NLG. For a 2024 decoder-only LM, a math-reasoning benchmark is the standard signal in current literature, and the ablations in §7 are *task-agnostic* — their scientific value (which matrices to adapt, what rank suffices) transfers cleanly.
- Train: 30k MetaMathQA examples × 1 epoch.
- Eval: full GSM8K test (1,319 problems), greedy decode, regex extraction of `#### N`.

### Experiments

**E0 — Baselines.** Reference points for everything else.

- E0a base model, no adaptation (zero-shot GSM8K, prompt as in `data/metamath.py`)
- E0b full fine-tuning (all params trainable), same data + steps as LoRA runs
- Single seed.

**E1 — Which weight matrices? (paper §7.1, Tab. 5)**
Fixed param budget (same *intent* as the paper). Vary which leaf modules receive LoRA, choose `r` per config so total trainable params stay constant. α fixed to `2·r` throughout.

Budget target: **≈1.61M trainable params** (28 layers). Note this is *not* the paper's 18M figure — that was sized for GPT-3 (175B). For Qwen3-1.7B a comparable adaptation lands in the low single-digit millions; the absolute number is irrelevant, only that it is held constant across configs.

LoRA params for a target set are `L · r · Σ(in+out)`. Because Qwen3 uses **GQA** (`k_proj`/`v_proj` out_features=1024 vs `q_proj`=2048), a naive "keep `r·#targets` constant" rule does **not** equalize the budget — `r` must be solved per config from the actual matrix dims.

| Config | Targets                          | r   | α   | Trainable | vs budget |
| ------ | -------------------------------- | --- | --- | --------- | --------- |
| E1a    | `q_proj`                         | 14  | 28  | 1.61M     | +0.0%     |
| E1b    | `v_proj`                         | 19  | 38  | 1.63M     | +1.8%     |
| E1c    | `q_proj, v_proj`                 | 8   | 16  | 1.61M     | +0.0%     |
| E1d    | `q_proj, k_proj, v_proj, o_proj` | 4   | 8   | 1.61M     | +0.0%     |
| E1e    | all attention + MLP (all-linear) | 2   | 4   | 2.18M     | +35.7%    |


> E1e is a *deviation* from the paper, which froze MLP layers (§4.2). In modern decoder LMs the MLP is the bulk of the parameters (gate/up/down ≈ 2/3 of trainable matmuls); whether adapting it helps is an open question worth a single data point. It is also **off-budget**: the all-linear target set is ~12× larger than `q_proj` alone, so the smallest integer rank (`r=2`) already overshoots the budget by ~36%. We keep `r=2` (the LoRA floor) and read E1e as a qualitative "does MLP adaptation help at all" point, **not** as an iso-budget row. The iso-budget comparison proper is E1a–E1d.

> **Single-seed caveat.** With one seed and n=1319, the binomial SE at acc≈0.70 is ≈0.013 (≈17 problems). Differences between E1 configs smaller than ~2 SE (≈0.026) are not distinguishable from noise — treat the E1 ranking as indicative, not conclusive.

Metric: GSM8K accuracy. Output: replicate Tab. 5 layout for Qwen3.

**E2 — Optimal rank? (paper §7.2, Tab. 6 / App. Tab. 18)**
Best **iso-budget** target set from E1a–E1d (the paper sweeps rank on the attention set; `q_proj, v_proj` is the canonical choice). Sweep `r ∈ {1, 2, 4, 8, 16, 64}`. α fixed to `2·r` per paper convention.
Output: rank-vs-accuracy curve, recreate Fig. 2 / Tab. 6 layout.

> Pick the E2 target set from E1a–E1d, **not** E1e: E1e is off-budget (see above), so its accuracy is not comparable. Sweeping rank on all-linear is a separate, optional extension (E1e-deep), not the §7.2 reproduction.

**E3 — Subspace analyses (paper §7.2 bottom / §7.3)**
Post-hoc on E2's checkpoints, no extra training compute.

- E3a Grassmann-similarity heatmap `φ(A_{r=8}, A_{r=64}, i, j)` for `ΔW_q`, `ΔW_v` (Fig. 3 layout, 4 representative layers of 28).
- E3b Two-seed `r=64` run → cross-seed subspace similarity (Fig. 4 layout). Adds **one** extra training run.
- E3c Amplification factor `‖ΔW‖_F / ‖U^⊤W V^⊤‖_F` for `r ∈ {4, 64}` (Tab. 7 layout).

**E4 — Headroom: when does LoRA help? (our own question)**
A controlled experiment isolating *one* factor: how much the base model already
masters the task. We pick two backbones that are identical in size and
architecture (1.5B, 28 layers, 12 Q / 2 KV heads, hidden 1536, GQA) and differ
**only** in pretraining math level:

| Backbone | base GSM8K-test (instruct @512) |
| -------- | ------------------------------- |
| Qwen2-1.5B   | 0.219 (weak — answers in code style, ignores `#### N`) |
| Qwen2.5-1.5B | 0.557 (strong — already follows the format) |

Data: GSM8K-train → GSM8K-test (in-distribution; the training answer format
`#### N` matches the eval format, no MetaMath mismatch). All runs use a **fixed
512-token budget** (self-terminating instruct/FT stop on EOS earlier; the cap
only binds for few-shot). A held-out 1000-example slice of GSM8K-**train** is the
dev set for the staged sweeps; GSM8K-test is reserved for the final numbers.

Per model we run the same funnel, passing the best config to the next stage:
epoch diagnostic → 3-point LR check → E1 target sweep (iso-budget ≈2.5M params,
GQA-solved per target set) → E2 rank sweep → final test matrix
(base 0-shot / few-shot / instruct + best LoRA + full FT).

Result: the LoRA gain over base-instruct is **+0.321** for the weak Qwen2-1.5B
but only **+0.031** for the strong Qwen2.5-1.5B — LoRA helps in proportion to the
pretraining headroom. Full tables, per-stage figures, and the slide writeup are
in [`analysis/headroom_summary.md`](analysis/headroom_summary.md); per-run JSONs
under `results_headroom/<model>/`.

> **Dev-metric caveat (contamination).** The dev slice is part of GSM8K-train,
> which the Qwen models appear to have seen in pretraining: the *untrained*
> Qwen2-1.5B base already scores **0.87 on dev-500 vs 0.25 on test-500** (same
> prompt/budget, ~0% truncation, matched difficulty). This is pretraining
> memorization of the train split, **not** a leakage bug — train/dev indices are
> disjoint and 0/1000 dev items have a ≥0.8-similar train neighbor. Dev is used
> only for *relative* within-model stage decisions (the offset cancels); every
> headline number is on the clean GSM8K-test split.

### Seeds

Single seed per config for E1/E2 (paper uses 3–5; we trade variance for coverage). One extra seed for the best-of-E2 row + one extra `r=64` run for E3b. Total: ~15 training runs.

## Outputs

Per run: `results/<exp_id>/summary.json` (config + final accuracy + loss curve), checkpoint under `checkpoints/<exp_id>/`. Aggregator script produces the §7.1 / §7.2 tables and §7.2 / §7.3 figures from the JSON summaries.

## Reproducibility notes

- All configs in `configs/exp/` (one YAML per run).
- Seed fixed in `configs/train/*.yaml`.
- bf16 throughout (matches what current LoRA work reports). Run E0b (full FT) in bf16 as well for an apples-to-apples comparison.
- GSM8K eval is greedy (`do_sample=False`); paper §5.4 uses beam search for NLG, but for math accuracy with `####` extraction greedy is the convention and removes one source of variance.
- Every config sets `rank`/`alpha`/`target_modules` **explicitly** — nothing is auto-solved. At run time the eval/smoke scripts print and persist a **LoRA budget report** (`rank`, `alpha`, scaling, matched modules, and the actual trainable-param count derived from the real model dims), and eval hard-fails if the params it unfroze don't match that report. So the defining knobs of each run live in the result JSON (`lora` block), not only in a YAML you have to go find.
- Pin `--batch-size` to the **same value** for every run in a comparison. Batched generation changes padding, and bf16 matmuls are not bit-exact across padding layouts, so the greedy path can diverge between an unbatched and a batched run even on the identical model. (Observed in the first E1 sweep: E1a was run with a different batch size than E1b–E1e, which is why their *base* completions were not byte-identical despite identical accuracy.)

