"""Base-only GSM8K eval with explicit truncation accounting.

Scores the *base* model under one or more prompt styles at a configurable
``max_new_tokens``, and reports — per run — the true truncation rate (generation
hit the length cap without emitting EOS) and the accuracy with truncated items
excluded. Used to separate the LoRA reasoning gain from the base model's
length-cap artifact under the verbose instruct / zeroshot prompts.

Usage (on a GPU pod):
    python scripts/eval_base_truncation.py \
        --prompts instruct zeroshot fewshot \
        --max-new-tokens 512 --num-problems 1319 --batch-size 32 \
        --out results/base_truncation_mnt512.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lora_qwen.config import LoraSetupConfig
from lora_qwen.evaluation import evaluate_gsm8k, load_gsm8k, resolve_prompt_fn
from lora_qwen.model import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lora-config", type=Path, default=Path("configs/E0a_base.yaml"))
    p.add_argument("--prompts", nargs="+", default=["instruct"],
                   choices=["zeroshot", "fewshot", "instruct"])
    p.add_argument("--split", choices=["test", "dev"], default="test")
    p.add_argument("--num-problems", type=int, default=1319)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--save-completions", action="store_true",
                   help="Persist per-example completions (large JSON).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = LoraSetupConfig.from_yaml(args.lora_config)
    problems = list(load_gsm8k(split=args.split, max_examples=args.num_problems))
    print(f"Loaded {len(problems)} {args.split} problems")

    model, tokenizer, device = load_model_and_tokenizer(cfg)
    model.eval()

    runs = {}
    for style in args.prompts:
        print(f"\n=== base / prompt={style} / max_new_tokens={args.max_new_tokens} ===")
        res = evaluate_gsm8k(
            model, tokenizer, problems,
            name=f"base_{style}",
            max_new_tokens=args.max_new_tokens,
            device=device,
            batch_size=args.batch_size,
            prompt_fn=resolve_prompt_fn(style),
        )
        n = res.total
        n_tr = sum(1 for e in res.examples if e.truncated)
        n_tr_corr = sum(1 for e in res.examples if e.truncated and e.correct)
        acc_excl = (
            (res.correct - n_tr_corr) / (n - n_tr) if (n - n_tr) else 0.0
        )
        entry = {
            "prompt": style,
            "max_new_tokens": args.max_new_tokens,
            "accuracy": res.accuracy,
            "correct": res.correct,
            "total": n,
            "wall_time_sec": res.wall_time_sec,
            "truncated": n_tr,
            "truncated_frac": n_tr / n if n else 0.0,
            "truncated_correct": n_tr_corr,
            "accuracy_excl_truncated": acc_excl,
        }
        if args.save_completions:
            entry["examples"] = [
                {"index": e.index, "gt": e.gt, "predicted": e.predicted,
                 "correct": e.correct, "truncated": e.truncated,
                 "completion": e.completion}
                for e in res.examples
            ]
        runs[style] = entry
        print(f"  acc={res.accuracy:.4f}  truncated={n_tr} ({100*entry['truncated_frac']:.1f}%)  "
              f"acc_excl_trunc={acc_excl:.4f}  wall={res.wall_time_sec:.0f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model_name": cfg.model_name,
        "split": args.split,
        "num_problems": len(problems),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "runs": runs,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
