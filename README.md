# lora-qwen

University ML project: reproducing the LoRA paper (Hu et al., 2021) on `Qwen/Qwen3-1.7B-Base` and extending it with the "all linear layers" configuration (the paper's explicit future work).

The framework is split into a stable **infrastructure layer** (this repo) and a pluggable **LoRA backend** (one file you replace). Implement the backend and everything else — loading, data, training, evaluation — runs as-is.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1. Verify LoRA injection: load model, patch, check identity, generate
python scripts/smoke_setup.py --config configs/all_linear.yaml

# 2. Verify end-to-end training: load dataset, LoRA-FT for 30 steps, save adapter
python scripts/smoke_ft.py

# 3. Measure base vs fine-tuned on GSM8K test problems
python scripts/eval_gsm8k.py --num-problems 30
```

First run downloads ~3.4 GB of model weights into the HuggingFace cache.

## Experimental conditions

| Config | Target modules                                               | File                              |
| ------ | ------------------------------------------------------------ | --------------------------------- |
| A      | _(none — base model)_                                        | `configs/base.yaml`               |
| B      | `q_proj`, `v_proj`                                           | `configs/attention_only.yaml`     |
| C      | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | `configs/all_linear.yaml`      |

## For colleagues: plug in your LoRA implementation

Everything is in place except the actual LoRA math. See the full guide in [`src/lora_qwen/lora/README.md`](src/lora_qwen/lora/README.md). TL;DR:

1. Fill in [`src/lora_qwen/lora/custom_backend.py`](src/lora_qwen/lora/custom_backend.py) — `LoRALayer`, `LinearWithLoRA`, `apply`, `save`, `load`.
2. Copy a LoRA config and switch `backend: "custom"`.
3. Run the same three scripts above against your config to validate.

The contract your backend must satisfy is defined in [`src/lora_qwen/lora/backend.py`](src/lora_qwen/lora/backend.py).

## Architecture

```
                scripts
                   |
       +-----------+-----------+
       |           |           |
  smoke_setup  smoke_ft   eval_gsm8k
       |           |           |
       v           v           v
  lora_qwen.{model, data, training, evaluation, sanity}    (infrastructure)
                        |
                        v
        lora_qwen.lora.{apply_lora, save_adapter, load_adapter}   (dispatcher)
                        |
              +---------+---------+
              |                   |
         peft_backend       custom_backend   (colleagues implement this)
        (reference impl)
```

## Layout

```
src/lora_qwen/
  config.py              LoraSetupConfig + YAML loader
  model.py               model + tokenizer loading, linear-module listing, trainable-param report
  sanity.py              identity check: base vs LoRA-patched model
  lora/
    README.md            contract + implementation guide (read this if you're a colleague)
    __init__.py          public API: apply_lora, save_adapter, load_adapter
    apply.py             backend dispatcher
    backend.py           LoRABackend Protocol + runtime validator
    peft_backend.py      HuggingFace peft reference implementation
    custom_backend.py    stub with TODO markers — the group's own implementation goes here
  data/
    __init__.py          public API
    registry.py          DatasetSpec, @register, SupervisedDataset (completion-only masking)
    collator.py          SupervisedCollator (dynamic padding, ignore_index=-100)
    config.py            DataConfig + YAML loader
    metamath.py          MetaMathQA loader (first concrete dataset)
  training/
    __init__.py
    config.py            TrainConfig + YAML loader
    loop.py              train() - backend-agnostic AdamW loop + warmup+cosine + grad accum + clip
  evaluation/
    __init__.py
    extract.py           regex answer extraction (#### / "The answer is" / last-number fallback)
    gsm8k.py             GSM8K loader + prompt formatting (shared with training prompt)
    runner.py            generation-based scoring loop
configs/
  base.yaml, attention_only.yaml, all_linear.yaml    LoRA conditions A / B / C
  data/metamath.yaml                                 dataset config
  train/smoke.yaml                                   30-step training recipe
scripts/
  smoke_setup.py         validates model load + LoRA injection + identity
  smoke_ft.py            short MetaMathQA fine-tune, confirms loss decreases
  eval_gsm8k.py          base vs LoRA-FT GSM8K comparison
results/                 JSON summaries per run
```

## Adding a new dataset

Drop a module in [`src/lora_qwen/data/`](src/lora_qwen/data/) that yields `Example(prompt, response)` objects and registers itself:

```python
from lora_qwen.data.registry import Example, register

@register("my_dataset", description="...")
def load_my_dataset(*, split="train", max_examples=None, **kwargs):
    ...
    yield Example(prompt=..., response=...)
```

Import the module once in [`src/lora_qwen/data/__init__.py`](src/lora_qwen/data/__init__.py) to trigger registration, then reference it by name from a `configs/data/*.yaml`.
