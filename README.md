# lora-qwen

University ML project reproducing the LoRA paper (Hu et al., 2021) on modern
decoder-only Qwen models and extending its ablations to a question the paper
does not answer: **how the LoRA gain depends on what the base model already
knows.** Downstream task: MetaMathQA / GSM8K-train → GSM8K-test.

The codebase is split into a stable **infrastructure layer** (model loading,
data, training, evaluation) and a pluggable **LoRA backend**. Two backends ship
and are interchangeable per config: a HuggingFace `peft` reference and an
in-house implementation (`custom`) verified against it (identity check +
loss-trajectory + downstream-accuracy parity within bf16 noise).

## What's in here

- **LoRA reproduction (paper §7.1 / §7.2):** which weight matrices to adapt
  (E1, iso-budget target sweep) and what rank suffices (E2, rank sweep), plus a
  subspace/SVD analysis (E3).
- **Headroom study (our own question, E4):** a controlled pair —
  **Qwen2-1.5B vs Qwen2.5-1.5B**, identical size/architecture, differing only in
  pretraining math level — showing that the LoRA gain shrinks as the base model
  already masters the task, and turns *negative* on a strong reference backbone
  (Qwen3-1.7B). See [`analysis/headroom_summary.md`](analysis/headroom_summary.md).
- **Mechanism + cost analysis:** where fine-tuning gains/losses actually come
  from (format compliance, non-termination, over-compression) and per-sample
  latency, in [`analysis/ft_gains_analysis.md`](analysis/ft_gains_analysis.md).

The full experimental plan, paper mapping, and scope decisions live in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Models

| Backbone | Role | Base config |
| --- | --- | --- |
| `Qwen/Qwen2-1.5B` | controlled pair (low math headroom) | `configs/qwen2_1.5b/base.yaml` |
| `Qwen/Qwen2.5-1.5B` | controlled pair (high math headroom) | `configs/qwen25_1.5b/base.yaml` |
| `Qwen/Qwen3-1.7B-Base` | reference backbone + original LoRA reproduction | `configs/qwen3_1.7b/base.yaml` |

All evaluation uses the **instruct** prompt (step-by-step + `#### N` answer
format) at a 512-token budget, scored on the full GSM8K-test (1,319 problems)
with greedy decoding and regex answer extraction.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1. Verify LoRA injection: load model, patch, identity check, generate
python scripts/smoke_setup.py --config configs/all_linear.yaml

# 2. Verify end-to-end training: LoRA-FT for a few steps, save adapter
python scripts/smoke_ft.py

# 3. Score base vs fine-tuned on a few GSM8K test problems
python scripts/eval_gsm8k.py --num-problems 30
```

First run downloads the model weights into the HuggingFace cache.

### GPU evaluation

`eval_gsm8k.py` batches generation; the default batch size is hardware-aware
(`16` on CUDA, `1` on MPS/CPU). Generation is the wall-time bottleneck
(autoregressive, one forward per token), so batch size is the main throughput
knob — raise it on bigger GPUs, halve it if you OOM.

```bash
python scripts/eval_gsm8k.py --num-problems 1319 --batch-size 32   # A40 48G
python scripts/eval_gsm8k.py --num-problems 1319 --batch-size 64   # A100 80G
```

## Training & sweeps

A single LoRA run (data/train configs default to GSM8K-train + the full recipe):

```bash
python scripts/train_lora.py --lora-config configs/E15b_qwen25/qv.yaml \
  --train-prompt instruct \
  --save-dir checkpoints/qwen25_qv --output results/qwen25_qv.json

# rank/alpha can be overridden from the CLI for rank sweeps (alpha defaults to 2*rank):
python scripts/train_lora.py --lora-config configs/E15b_qwen25/qv.yaml --rank 8 \
  --save-dir checkpoints/qwen25_qv_r8 --output results/qwen25_qv_r8.json
```

The headroom study runs a staged funnel per model (epoch diagnostic → LR check →
E1 target sweep → E2 rank sweep → final test matrix), orchestrated by
[`scripts/run_headroom_pipeline.py`](scripts/run_headroom_pipeline.py). Target/
rank are **selected on a held-out dev slice of GSM8K-train**; all reported
numbers are re-scored on the clean GSM8K-test split
([`scripts/reeval_sweeps_test.py`](scripts/reeval_sweeps_test.py)). Figures and
the slide-ready summary are produced by
[`scripts/plot_headroom.py`](scripts/plot_headroom.py).

> **Note on the dev slice.** The Qwen base models appear to have seen
> GSM8K-train during pretraining (the untrained base already scores far higher on
> the dev slice than on test). The dev metric is therefore used only for
> *relative* in-model selection, where the constant offset cancels; every
> headline number is on GSM8K-test.

## Architecture

```
                          scripts/ (smoke, train, eval, sweeps, analysis, plots)
                                       |
        lora_qwen.{model, data, training, evaluation, sanity}   (infrastructure)
                                       |
              lora_qwen.lora.{apply_lora, save_adapter, load_adapter}   (dispatcher)
                                       |
                          +------------+------------+
                          |                         |
                   peft_backend              custom_backend
                  (reference impl)        (in-house, verified)
```

## Layout

```
src/lora_qwen/
  config.py              LoraSetupConfig + YAML loader
  model.py               model + tokenizer loading, trainable-param report
  sanity.py              identity check: base vs LoRA-patched model
  lora/
    __init__.py          public API: apply_lora, save_adapter, load_adapter
    apply.py             backend dispatcher
    backend.py           LoRABackend Protocol + runtime validator
    peft_backend.py      HuggingFace peft reference implementation
    custom_backend.py    in-house LoRA (LoRALayer, LinearWithLoRA, apply/save/load)
  data/
    registry.py          DatasetSpec, @register, SupervisedDataset (completion-only masking)
    collator.py          SupervisedCollator (dynamic padding, ignore_index=-100)
    config.py            DataConfig + YAML loader
    metamath.py          MetaMathQA loader + shared Problem:/Solution: prompt
    gsm8k_train.py       GSM8K-train loader with fixed held-out dev slice
  training/
    config.py            TrainConfig + YAML loader
    loop.py              backend-agnostic AdamW loop (warmup+cosine, grad accum, clip)
  evaluation/
    extract.py           regex answer extraction (#### / "The answer is" / last-number)
    gsm8k.py             GSM8K loader + zeroshot/fewshot/instruct prompts
    runner.py            batched generation scoring + truncation accounting
configs/
  qwen2_1.5b/, qwen25_1.5b/, qwen3_1.7b/   per-model base + LoRA configs
  E15b_qwen2/, E15b_qwen25/                 iso-budget target sweeps (E1)
  E1_*/, E2/, E_*/                          paper-reproduction sweep grids
  data/, train/                             dataset + training recipes
scripts/
  smoke_setup.py, smoke_ft.py              setup + training validation
  train_lora.py, train_full_ft.py          LoRA / full fine-tuning entry points
  eval_gsm8k.py, eval_base_truncation.py   evaluation (with truncation tracking)
  run_headroom_pipeline.py                 staged funnel per model
  reeval_sweeps_test.py                    re-score dev-selected sweeps on test
  run_reference_point.py, run_ft_analysis.py, bench_latency.py
  plot_*.py, categorize_diffs.py, analyze_subspace.py, aggregate_results.py
analysis/                headroom + ft-gains writeups and category data
results_headroom/        per-model funnel outputs, figures, final test results
```

## Adding a new dataset

Drop a module in [`src/lora_qwen/data/`](src/lora_qwen/data/) that yields
`Example(prompt, response)` objects and registers itself:

```python
from lora_qwen.data.registry import Example, register

@register("my_dataset", description="...")
def load_my_dataset(*, split="train", max_examples=None, **kwargs):
    ...
    yield Example(prompt=..., response=...)
```

Import the module once in
[`src/lora_qwen/data/__init__.py`](src/lora_qwen/data/__init__.py) to trigger
registration, then reference it by name from a `configs/data/*.yaml`.
