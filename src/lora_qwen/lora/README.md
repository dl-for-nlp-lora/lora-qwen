# LoRA backend

This subpackage owns LoRA injection. Everything else (model loading, data,
training, evaluation) is backend-agnostic and goes through the three public
functions in [`__init__.py`](__init__.py):

```python
apply_lora(model, config)                    # attach adapters, freeze base
save_adapter(model, save_dir, config)        # persist trained adapter
load_adapter(base_model, save_dir, config)   # restore for evaluation
```

The dispatcher in [`apply.py`](apply.py) selects the implementation from
`config.backend`. Two backends ship and are fully interchangeable per config:

| `backend:` | Module | Role |
| --- | --- | --- |
| `peft` | [`peft_backend.py`](peft_backend.py) | HuggingFace `peft`, reference implementation |
| `custom` | [`custom_backend.py`](custom_backend.py) | in-house from-scratch LoRA |

The two are verified equivalent: the patched model passes the identity check at
init, the custom backend tracks the `peft` loss trajectory, and downstream GSM8K
accuracy matches within bf16 noise.

## The contract

A backend is a module exposing three functions; the formal Protocol and a
runtime validator live in [`backend.py`](backend.py):

```python
def apply(model: nn.Module, config: LoraSetupConfig) -> nn.Module: ...
def save(model: nn.Module, save_dir: str | Path) -> None: ...
def load(base_model: nn.Module, save_dir: str | Path, config: LoraSetupConfig) -> nn.Module: ...
```

Three semantic invariants any backend must satisfy:

1. **Structure.** After `apply`, every base parameter is frozen and only the
   LoRA parameters have `requires_grad=True`.
2. **Identity at init.** The patched model's logits equal the base model's,
   because `B = 0` initially ⇒ `ΔW·x = (α/r)·BA·x = 0`. This is the central LoRA
   correctness test; `scripts/smoke_setup.py` checks it via
   `lora_qwen.sanity.compare_logits`.
3. **Save/load round-trip.** `load(fresh_base, save_dir, config)` reproduces a
   model numerically equivalent to the trained one that was saved.

## How the custom backend works

[`custom_backend.py`](custom_backend.py) is a compact reference for the LoRA
method (Hu et al. 2021, §4.1):

- **`LoRALayer`** — the low-rank update `(α/r) · B(A(x))`. `A` is Kaiming-uniform
  (drawn in fp32, then cast to the param dtype to preserve entropy under bf16),
  `B` is zero-initialized, with dropout on the input.
- **`LinearWithLoRA`** — wraps a frozen `nn.Linear` and adds the parallel
  `LoRALayer`; forward returns `base(x) + lora(x)`. LoRA params inherit the
  wrapped layer's device/dtype.
- **`apply`** — freezes the model, walks `named_modules()`, and replaces every
  `nn.Linear` whose leaf name is in `config.target_modules` with a
  `LinearWithLoRA`.
- **`save`** — dumps only the trainable tensors (selected by `requires_grad`, not
  a name pattern) to `adapter.pt` (a few MB; base weights are never saved).
- **`load`** — re-runs `apply` to rebuild the structure, then loads `adapter.pt`
  with `strict=False`.

## Validating a backend

Point a config at a backend (copy one and set `backend:` accordingly), then run:

```bash
# 1. Injection correctness — identity check must pass
python scripts/smoke_setup.py --config configs/all_linear.yaml

# 2. End-to-end training — loss must decrease
python scripts/smoke_ft.py --lora configs/all_linear.yaml

# 3. GSM8K base-vs-FT — must show a non-zero delta
python scripts/eval_gsm8k.py --lora-config configs/all_linear.yaml \
    --adapter-dir checkpoints/smoke_ft --num-problems 20
```

If step 1 fails the injection is wrong (not the LoRA math); if step 1 passes but
step 2's loss is flat the gradient flow is wrong (a stray detach or wrong
`requires_grad`); if step 2 passes but step 3 matches the base, `save`/`load`
isn't restoring trained weights.

## Paper reference

Hu et al. 2021, *LoRA: Low-Rank Adaptation of Large Language Models* —
<https://arxiv.org/abs/2106.09685> (§4.1 method, §4.2 target-module choice).
