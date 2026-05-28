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
)
from lora_qwen.lora import load_adapter  # noqa: E402
from lora_qwen.model import load_model_and_tokenizer  # noqa: E402


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
    p.add_argument("--max-new-tokens", type=int, default=256)
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
    return p.parse_args()


def _summary_dict(result: EvalResult, *, save_completions: bool) -> dict:
    out = {
        "name": result.name,
        "accuracy": result.accuracy,
        "correct": result.correct,
        "total": result.total,
        "wall_time_sec": result.wall_time_sec,
    }
    if save_completions:
        out["examples"] = [
            {
                "index": ex.index,
                "gt": ex.gt,
                "predicted": ex.predicted,
                "correct": ex.correct,
                "completion": ex.completion,
            }
            for ex in result.examples
        ]
    return out


def main() -> int:
    args = parse_args()
    lora_cfg = LoraSetupConfig.from_yaml(args.lora_config)
    if not args.adapter_dir.exists():
        raise SystemExit(
            f"Adapter dir {args.adapter_dir} does not exist. "
            f"Run `python scripts/smoke_ft.py` first."
        )

    print(f"[1/5] Loading GSM8K test (first {args.num_problems} problems)...")
    problems = list(load_gsm8k(split="test", max_examples=args.num_problems))
    print(f"      -> {len(problems)} problems")

    # ---- Pass A: base ------------------------------------------------------
    print(f"\n[2/5] Loading base model ({lora_cfg.model_name})...")
    base_model, tokenizer, device = load_model_and_tokenizer(lora_cfg)

    print(f"[3/5] Scoring base on {device}:")
    base_res = evaluate_gsm8k(
        base_model, tokenizer, problems,
        name="base",
        max_new_tokens=args.max_new_tokens,
        device=device,
        batch_size=args.batch_size,
    )
    print("  " + base_res.summary() + f"  | {base_res.wall_time_sec:.1f}s")

    # Free the base model before loading the FT one; on Mac unified memory we
    # can comfortably hold one 1.7B model at a time.
    del base_model
    import gc
    gc.collect()
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Pass B: fine-tuned ------------------------------------------------
    print(f"\n[4/5] Loading base + attaching adapter from {args.adapter_dir}...")
    ft_model, tokenizer, device = load_model_and_tokenizer(lora_cfg)
    ft_model = load_adapter(ft_model, args.adapter_dir, lora_cfg)
    ft_model.to(device)
    ft_model.eval()

    print(f"[5/5] Scoring FT on {device}:")
    ft_res = evaluate_gsm8k(
        ft_model, tokenizer, problems,
        name="ft",
        max_new_tokens=args.max_new_tokens,
        device=device,
        batch_size=args.batch_size,
    )
    print("  " + ft_res.summary() + f"  | {ft_res.wall_time_sec:.1f}s")

    delta = ft_res.accuracy - base_res.accuracy
    print("\n=== Summary ===")
    print(f"  base accuracy : {base_res.accuracy:.3f}  ({base_res.correct}/{base_res.total})")
    print(f"  ft   accuracy : {ft_res.accuracy:.3f}  ({ft_res.correct}/{ft_res.total})")
    print(f"  delta         : {delta:+.3f}  ({ft_res.correct - base_res.correct:+d} correct)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_compl = not args.no_save_completions
    payload = {
        "lora_config": str(args.lora_config),
        "adapter_dir": str(args.adapter_dir),
        "model_name": lora_cfg.model_name,
        "backend": lora_cfg.backend,
        "device": str(device),
        "num_problems": len(problems),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "base": _summary_dict(base_res, save_completions=save_compl),
        "ft": _summary_dict(ft_res, save_completions=save_compl),
        "delta_accuracy": delta,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
