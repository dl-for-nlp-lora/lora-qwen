"""GSM8K evaluation: compare the base model against a LoRA-adapted variant.

Loads the base model twice (once clean, once with the saved adapter attached)
and evaluates both on the same problems. Loading twice (vs adapter-toggling)
keeps the script backend-agnostic: it works identically whether you trained
with the peft reference or with the group's custom backend.

Run:
    python scripts/eval_gsm8k.py --num-problems 30
    python scripts/eval_gsm8k.py --lora-config configs/all_linear.yaml \\
        --adapter-dir checkpoints/smoke_ft --num-problems 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lora_qwen.config import LoraSetupConfig  # noqa: E402
from lora_qwen.evaluation import (  # noqa: E402
    EvalResult,
    evaluate_gsm8k,
    load_gsm8k,
    resolve_prompt_fn,
)
from lora_qwen.lora import load_adapter  # noqa: E402
from lora_qwen.model import (  # noqa: E402
    load_model_and_tokenizer,
    lora_budget_report,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GSM8K eval: base vs LoRA-FT")
    p.add_argument("--lora-config", type=Path, default=REPO_ROOT / "configs" / "all_linear.yaml")
    p.add_argument(
        "--adapter-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "smoke_ft",
        help="Directory saved by save_adapter (backend-specific format)",
    )
    p.add_argument("--num-problems", type=int, default=30)
    p.add_argument("--split", choices=["test", "dev"], default="test",
                   help="GSM8K split: 'test' (final reporting) or 'dev' (held-out "
                        "slice of train, for sweeps).")
    p.add_argument("--base-prompt", choices=["zeroshot", "fewshot", "instruct"],
                   default="zeroshot",
                   help="Prompt for the BASE pass.")
    p.add_argument("--ft-prompt", choices=["zeroshot", "fewshot", "instruct"],
                   default="zeroshot",
                   help="Prompt for the FT pass (match training prompt for fair comparison).")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Generation batch size. Default is hardware-aware: 16 on CUDA "
            "(safe on T4 16GB; A40 48GB / A100 can comfortably run 32-64), "
            "1 on MPS / CPU. Larger batches give near-linear speedups until "
            "you hit VRAM."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "gsm8k_base_vs_ft.json",
    )
    p.add_argument("--no-save-completions", action="store_true",
                   help="Don't persist per-example completions (smaller JSON)")
    p.add_argument(
        "--skip-base",
        action="store_true",
        help=(
            "Skip the base-model pass. The frozen base is bit-identical for "
            "every run in a fixed (device, batch-size) cohort, so when batch-"
            "evaluating many adapters on one machine the base only needs to be "
            "scored once. The 'base' block is then omitted from the JSON."
        ),
    )
    p.add_argument(
        "--full-model-dir",
        type=Path,
        default=None,
        help="Load a full fine-tuned checkpoint (E0b) instead of base+adapter.",
    )
    return p.parse_args()


def _summary_dict(result: EvalResult, *, save_completions: bool) -> dict:
    n_trunc = sum(1 for ex in result.examples if ex.truncated)
    n_trunc_correct = sum(1 for ex in result.examples if ex.truncated and ex.correct)
    out = {
        "name": result.name,
        "accuracy": result.accuracy,
        "correct": result.correct,
        "total": result.total,
        "wall_time_sec": result.wall_time_sec,
        # Generation hit max_new_tokens without EOS (length-cut, format-agnostic).
        "truncated": n_trunc,
        "truncated_frac": n_trunc / result.total if result.total else 0.0,
        "truncated_correct": n_trunc_correct,
        "accuracy_excl_truncated": (
            (result.correct - n_trunc_correct) / (result.total - n_trunc)
            if (result.total - n_trunc) else 0.0
        ),
    }
    if save_completions:
        out["examples"] = [
            {
                "index": ex.index,
                "gt": ex.gt,
                "predicted": ex.predicted,
                "correct": ex.correct,
                "truncated": ex.truncated,
                "completion": ex.completion,
            }
            for ex in result.examples
        ]
    return out


def main() -> int:
    args = parse_args()
    lora_cfg = LoraSetupConfig.from_yaml(args.lora_config)
    full_ft = args.full_model_dir is not None
    if full_ft:
        if not args.full_model_dir.exists():
            raise SystemExit(f"Full-model dir {args.full_model_dir} does not exist.")
    elif not args.adapter_dir.exists():
        raise SystemExit(
            f"Adapter dir {args.adapter_dir} does not exist. "
            f"Run `python scripts/smoke_ft.py` first."
        )

    print(f"[1/5] Loading GSM8K {args.split} (first {args.num_problems} problems)...")
    problems = list(load_gsm8k(split=args.split, max_examples=args.num_problems))
    print(f"      -> {len(problems)} problems")
    base_prompt_fn = resolve_prompt_fn(args.base_prompt)
    ft_prompt_fn = resolve_prompt_fn(args.ft_prompt)

    # ---- Pass A: base ------------------------------------------------------
    base_res = None
    if full_ft and args.skip_base:
        from lora_qwen.model import resolve_device

        device = resolve_device(lora_cfg.device)
        budget = None
        print("\n[2/5] Skipping base load (--full-model-dir + --skip-base).")
        print("[3/5] Skipping base pass.")
    else:
        print(f"\n[2/5] Loading base model ({lora_cfg.model_name})...")
        base_model, tokenizer, device = load_model_and_tokenizer(lora_cfg)

        budget = lora_budget_report(base_model, lora_cfg)
        print(budget.render(prefix="      "))

        if args.skip_base:
            print("[3/5] Skipping base pass (--skip-base): base is bit-identical "
                  "across a fixed (device, batch-size) cohort.")
        else:
            print(f"[3/5] Scoring base on {device}:")
            base_res = evaluate_gsm8k(
                base_model, tokenizer, problems,
                name="base",
                max_new_tokens=args.max_new_tokens,
                device=device,
                batch_size=args.batch_size,
                prompt_fn=base_prompt_fn,
            )
            print("  " + base_res.summary() + f"  | {base_res.wall_time_sec:.1f}s")

        del base_model
        import gc
        gc.collect()
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Pass B: fine-tuned ------------------------------------------------
    if full_ft:
        print(f"\n[4/5] Loading full fine-tuned model from {args.full_model_dir}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ft_model = AutoModelForCausalLM.from_pretrained(
            args.full_model_dir, dtype=resolve_dtype(lora_cfg.dtype),
        )
        tokenizer = AutoTokenizer.from_pretrained(args.full_model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        ft_model.to(device)
        ft_model.eval()
        actual_trainable = sum(p.numel() for p in ft_model.parameters())
        print(f"      full FT checkpoint loaded ({actual_trainable:,} params)")
    else:
        print(f"\n[4/5] Loading base + attaching adapter from {args.adapter_dir}...")
        ft_model, tokenizer, device = load_model_and_tokenizer(lora_cfg)
        ft_model = load_adapter(ft_model, args.adapter_dir, lora_cfg)
        ft_model.to(device)
        ft_model.eval()

        actual_trainable = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
        if actual_trainable != budget.expected_trainable:
            raise SystemExit(
                "Trainable-param mismatch: config/report expects "
                f"{budget.expected_trainable:,} but the attached adapter has "
                f"{actual_trainable:,}. The config does not match this checkpoint."
            )
        print(f"      budget check OK: {actual_trainable:,} trainable params")

    print(f"[5/5] Scoring FT on {device}:")
    ft_res = evaluate_gsm8k(
        ft_model, tokenizer, problems,
        name="ft",
        max_new_tokens=args.max_new_tokens,
        device=device,
        batch_size=args.batch_size,
        prompt_fn=ft_prompt_fn,
    )
    print("  " + ft_res.summary() + f"  | {ft_res.wall_time_sec:.1f}s")

    print("\n=== Summary ===")
    if base_res is not None:
        delta = ft_res.accuracy - base_res.accuracy
        print(f"  base accuracy : {base_res.accuracy:.3f}  ({base_res.correct}/{base_res.total})")
        print(f"  ft   accuracy : {ft_res.accuracy:.3f}  ({ft_res.correct}/{ft_res.total})")
        print(f"  delta         : {delta:+.3f}  ({ft_res.correct - base_res.correct:+d} correct)")
    else:
        delta = None
        print(f"  ft   accuracy : {ft_res.accuracy:.3f}  ({ft_res.correct}/{ft_res.total})")
        print("  base          : skipped (--skip-base)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_compl = not args.no_save_completions
    payload = {
        "lora_config": str(args.lora_config),
        "adapter_dir": str(args.adapter_dir) if not full_ft else None,
        "full_model_dir": str(args.full_model_dir) if full_ft else None,
        "model_name": lora_cfg.model_name,
        "backend": lora_cfg.backend if not full_ft else "full_ft",
        "device": str(device),
        "split": args.split,
        "base_prompt": args.base_prompt,
        "ft_prompt": args.ft_prompt,
        "num_problems": len(problems),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "lora": None if full_ft else {**budget.to_dict(), "actual_trainable_params": actual_trainable},
        "full_ft_params": actual_trainable if full_ft else None,
        "base": _summary_dict(base_res, save_completions=save_compl) if base_res is not None else None,
        "ft": _summary_dict(ft_res, save_completions=save_compl),
        "delta_accuracy": delta,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
