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
Fixed param budget ≈18M (same target as paper). Vary which leaf modules receive LoRA, choose `r` per config so total trainable params stay ≈18M.


| Config | Targets                          | r   |
| ------ | -------------------------------- | --- |
| E1a    | `q_proj`                         | 16  |
| E1b    | `v_proj`                         | 16  |
| E1c    | `q_proj, v_proj`                 | 8   |
| E1d    | `q_proj, k_proj, v_proj, o_proj` | 4   |
| E1e    | all attention + MLP (all-linear) | 2   |


> E1e is a *deviation* from the paper, which froze MLP layers (§4.2). In modern decoder LMs the MLP is the bulk of the parameters (gate/up/down ≈ 2/3 of trainable matmuls); whether adapting it helps is an open question worth a single data point.

Metric: GSM8K accuracy. Output: replicate Tab. 5 layout for Qwen3.

**E2 — Optimal rank? (paper §7.2, Tab. 6 / App. Tab. 18)**
Best target set from E1 (likely `q_proj, v_proj`). Sweep `r ∈ {1, 2, 4, 8, 16, 64}`. α fixed to `2·r` per paper convention.
Output: rank-vs-accuracy curve, recreate Fig. 2 / Tab. 6 layout.

**E3 — Subspace analyses (paper §7.2 bottom / §7.3)**
Post-hoc on E2's checkpoints, no extra training compute.

- E3a Grassmann-similarity heatmap `φ(A_{r=8}, A_{r=64}, i, j)` for `ΔW_q`, `ΔW_v` (Fig. 3 layout, 4 representative layers of 28).
- E3b Two-seed `r=64` run → cross-seed subspace similarity (Fig. 4 layout). Adds **one** extra training run.
- E3c Amplification factor `‖ΔW‖_F / ‖U^⊤W V^⊤‖_F` for `r ∈ {4, 64}` (Tab. 7 layout).

### Seeds

Single seed per config for E1/E2 (paper uses 3–5; we trade variance for coverage). One extra seed for the best-of-E2 row + one extra `r=64` run for E3b. Total: ~15 training runs.

## Outputs

Per run: `results/<exp_id>/summary.json` (config + final accuracy + loss curve), checkpoint under `checkpoints/<exp_id>/`. Aggregator script produces the §7.1 / §7.2 tables and §7.2 / §7.3 figures from the JSON summaries.

## Reproducibility notes

- All configs in `configs/exp/` (one YAML per run).
- Seed fixed in `configs/train/*.yaml`.
- bf16 throughout (matches what current LoRA work reports). Run E0b (full FT) in bf16 as well for an apples-to-apples comparison.
- GSM8K eval is greedy (`do_sample=False`); paper §5.4 uses beam search for NLG, but for math accuracy with `####` extraction greedy is the convention and removes one source of variance.

