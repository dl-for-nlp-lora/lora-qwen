"""End-to-end smoke test for the LoRA setup.

Steps:
 1. Load Qwen base model + tokenizer from config.
 2. Introspect + summarize nn.Linear modules (so we can verify target names).
 3. Capture base logits for a fixed prompt.
 4. Apply LoRA (peft backend, in-place wrap) according to config.
 5. Capture patched logits, assert identity (because B=0 at init).
 6. Print trainable-param summary.
 7. Generate a short sample.

Run:
    python scripts/smoke_setup.py --config configs/all_linear.yaml
    python scripts/smoke_setup.py --config configs/attention_only.yaml
    python scripts/smoke_setup.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lora_qwen.config import LoraSetupConfig  # noqa: E402
from lora_qwen.lora import apply_lora  # noqa: E402
from lora_qwen.model import (  # noqa: E402
    load_model_and_tokenizer,
    lora_budget_report,
    print_trainable_params,
    summarize_linear_modules,
)
from lora_qwen.sanity import (  # noqa: E402
    assert_identity,
    capture_logits,
    compare_logits,
)

SMOKE_PROMPT = "The capital of France is"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA setup smoke test")
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "all_linear.yaml",
        help="Path to a LoRA setup YAML config",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=SMOKE_PROMPT,
        help="Prompt used for identity check and sample generation",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Tokens to generate in the sample output",
    )
    p.add_argument(
        "--skip-identity",
        action="store_true",
        help="Skip the base-vs-patched identity check",
    )
    return p.parse_args()


def _print_linear_summary(model: torch.nn.Module) -> None:
    summary = summarize_linear_modules(model)
    leaves_sorted = sorted(summary.items(), key=lambda kv: (-kv[1], kv[0]))
    rendered = ", ".join(f"{name}={count}" for name, count in leaves_sorted)
    print(f"[2/7] nn.Linear leaf-name counts: {rendered}")


def _sample_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def main() -> int:
    args = parse_args()
    config = LoraSetupConfig.from_yaml(args.config)
    print(f"[0/7] Config: {config.describe()}")

    print(f"[1/7] Loading model + tokenizer ({config.model_name}, {config.dtype})...")
    model, tokenizer, device = load_model_and_tokenizer(config)
    print(f"      -> device={device}, param dtype={next(model.parameters()).dtype}")

    _print_linear_summary(model)

    if config.target_modules is not None:
        print(lora_budget_report(model, config).render(prefix="      "))

    do_identity = (not args.skip_identity) and config.target_modules is not None
    base_logits = None
    if do_identity:
        print("[3/7] Capturing base-model logits for identity check...")
        base_logits = capture_logits(model, tokenizer, prompt=args.prompt)
    else:
        print("[3/7] Skipping base-logit capture (config A or --skip-identity).")

    print(f"[4/7] Applying LoRA (backend={config.backend}, targets={config.target_modules})...")
    model = apply_lora(model, config)

    if do_identity:
        print("[5/7] Capturing patched-model logits + comparing...")
        patched_logits = capture_logits(model, tokenizer, prompt=args.prompt)
        result = compare_logits(
            base_logits,
            patched_logits,
            dtype=next(model.parameters()).dtype,
        )
        print(f"      -> {result.summary()}")
        assert_identity(result)
    else:
        print("[5/7] Identity check skipped.")

    print("[6/7] Trainable params report:")
    print_trainable_params(model, prefix="      ")

    print(f"[7/7] Sample generation (prompt={args.prompt!r}, greedy, "
          f"max_new_tokens={args.max_new_tokens})...")
    completion = _sample_generate(model, tokenizer, args.prompt, args.max_new_tokens)
    print(f"      >>> {completion!r}")

    print("\nSmoke setup finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
