# LoRA backend — implementation guide

This is the only subsystem you need to touch. Everything else (model loading, data pipeline, training orchestration, evaluation) is already wired up and will call into whatever backend you build here.

## The contract

A backend is a Python module that exposes three functions:

```python
def apply(model: nn.Module, config: LoraSetupConfig) -> nn.Module: ...
def save(model: nn.Module, save_dir: str | Path) -> None: ...
def load(base_model: nn.Module, save_dir: str | Path, config: LoraSetupConfig) -> nn.Module: ...
```

See [`backend.py`](backend.py) for the formal Protocol and a runtime validator that the dispatcher uses.

The three semantic invariants:

1. **Structure:** after `apply(base_model, config)`, every base parameter is frozen and only your LoRA parameters have `requires_grad=True`.
2. **Identity at init:** the patched model's logits must equal the base model's logits on any input, because `B=0` initially ⇒ `ΔW·x = BA·x = 0`. This is the central LoRA correctness test. The smoke script checks it automatically.
3. **Save/load round-trip:** `load(fresh_base, save_dir, config)` must produce a model numerically equivalent to the trained model that was saved.

## Your task

Fill in [`custom_backend.py`](custom_backend.py). The file is already scaffolded with `NotImplementedError` markers and docstrings. You are implementing:

| Class / function     | What it does                                                        |
| -------------------- | ------------------------------------------------------------------- |
| `LoRALayer`          | Low-rank update `(α/r) · B(A(x))`, Kaiming-A, zero-B, dropout       |
| `LinearWithLoRA`     | Frozen base `nn.Linear` + parallel `LoRALayer`; forward sums them   |
| `apply`              | Walk the model, replace target Linears with `LinearWithLoRA`        |
| `save`               | Dump the trainable (LoRA) tensors only                              |
| `load`               | Re-apply structure, then load saved tensors                         |

You do NOT need to touch:

- Model loading (`lora_qwen/model.py`)
- Data pipeline (`lora_qwen/data/`)
- Training loop (`lora_qwen/training/`)
- Evaluation (`lora_qwen/evaluation/`)
- Any script under `scripts/`

## How to run your implementation

Point a config at your backend (copy an existing one and change `backend:`):

```bash
cp configs/all_linear.yaml configs/all_linear_custom.yaml
# edit the copy: set   backend: "custom"
```

Then run the existing scripts against it:

```bash
# Step 1: injection correctness (identity check must pass)
python scripts/smoke_setup.py --config configs/all_linear_custom.yaml

# Step 2: a short FT must make the loss decrease
python scripts/smoke_ft.py --lora configs/all_linear_custom.yaml

# Step 3: GSM8K base-vs-FT must show a non-zero delta
python scripts/eval_gsm8k.py --lora-config configs/all_linear_custom.yaml \
    --adapter-dir checkpoints/smoke_ft_custom --num-problems 20
```

If step 1 fails, your injection is wrong (not the LoRA math). If step 1 passes but step 2's loss doesn't decrease, your gradient flow is wrong (typically an autograd detach somewhere or wrong `requires_grad` flags). If step 2 passes but step 3 gives the same accuracy as base, your `save`/`load` probably isn't restoring trained weights.

## Comparing against the peft reference

To be confident your impl is equivalent to the peft reference, train both with **the same seed, same data order, same hyperparameters, same adapter config** and compare the final loss trajectory. Small numerical differences are expected (different init RNG, possibly different epsilon in Adam eps or dropout masks); large drifts (>0.1 in final loss) point at a bug.

Example:

```bash
# Reference run
python scripts/smoke_ft.py --lora configs/all_linear.yaml
cp checkpoints/smoke_ft/adapter_model.safetensors /tmp/peft_adapter.st

# Your run (same seed set in configs/train/smoke.yaml)
python scripts/smoke_ft.py --lora configs/all_linear_custom.yaml
```

## Paper reference

Hu et al. 2021, LoRA: Low-Rank Adaptation of Large Language Models — <https://arxiv.org/abs/2106.09685>

- §4.1: the method (`ΔW = BA`, init `A` Kaiming, init `B` zero)
- §4.2: which modules to target (the experiment we're reproducing)

## Architecture one-liner

```
scripts / training / evaluation  --->  apply_lora / save_adapter / load_adapter
                                            |
                                            v
                                       dispatcher
                                       /        \
                                 peft_backend    custom_backend  <-- YOU
```
