"""Pin down the dev>>test gap on the GPU: score BASE and the best FT adapter on
dev-500 and test-500 under the identical eval path, and report accuracy +
truncation + mean completion length for each cell.

Logic:
  - If BASE(dev) ~ BASE(test): the two splits are equally hard for the untrained
    model -> the gap is induced by fine-tuning (the SFT distribution matches the
    train-derived dev slice better than the official test split).
  - If BASE(dev) >> BASE(test): the dev slice is intrinsically easier -> the gap
    is a split artifact, present even without any training.

Run on a pod (shared volume), e.g. for Qwen2-1.5B:
    python scripts/diag_dev_test_gap.py \
        --base-cfg configs/qwen2_1.5b/base.yaml \
        --ft-cfg   results_headroom/qwen2_1.5b/best_lora.yaml \
        --adapter  checkpoints/headroom/qwen2_1.5b/e2_attention_r1 \
        --n 500 --out results_headroom/qwen2_1.5b/diag_dev_test_gap.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lora_qwen.config import LoraSetupConfig
from lora_qwen.evaluation import evaluate_gsm8k, load_gsm8k, resolve_prompt_fn
from lora_qwen.lora import load_adapter
from lora_qwen.model import load_model_and_tokenizer


def _stats(res) -> dict:
    n = res.total
    n_tr = sum(1 for e in res.examples if e.truncated)
    clen = [len(e.completion) for e in res.examples]
    return {
        "accuracy": res.accuracy, "correct": res.correct, "total": n,
        "truncated": n_tr, "truncated_frac": n_tr / n if n else 0.0,
        "mean_completion_chars": sum(clen) / len(clen) if clen else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cfg", required=True)
    ap.add_argument("--ft-cfg", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--prompt", default="instruct")
    ap.add_argument("--mnt", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt_fn = resolve_prompt_fn(args.prompt)
    out: dict = {"prompt": args.prompt, "n": args.n, "cells": {}}

    # ---- BASE on dev + test ----
    base_cfg = LoraSetupConfig.from_yaml(args.base_cfg)
    bm, btok, dev = load_model_and_tokenizer(base_cfg)
    bm.eval()
    for split in ["dev", "test"]:
        probs = list(load_gsm8k(split=split, max_examples=args.n))
        r = evaluate_gsm8k(bm, btok, probs, name=f"base_{split}",
                           max_new_tokens=args.mnt, device=dev, batch_size=64,
                           prompt_fn=prompt_fn)
        out["cells"][f"base_{split}"] = _stats(r)
        print(f"base/{split}: acc={r.accuracy:.4f} "
              f"trunc={out['cells'][f'base_{split}']['truncated']}/{args.n}", flush=True)
    del bm

    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- FT adapter on dev + test ----
    ft_cfg = LoraSetupConfig.from_yaml(args.ft_cfg)
    fm, ftok, dev = load_model_and_tokenizer(ft_cfg)
    fm = load_adapter(fm, args.adapter, ft_cfg)
    fm.to(dev)
    fm.eval()
    for split in ["dev", "test"]:
        probs = list(load_gsm8k(split=split, max_examples=args.n))
        r = evaluate_gsm8k(fm, ftok, probs, name=f"ft_{split}",
                           max_new_tokens=args.mnt, device=dev, batch_size=64,
                           prompt_fn=prompt_fn)
        out["cells"][f"ft_{split}"] = _stats(r)
        print(f"ft/{split}: acc={r.accuracy:.4f} "
              f"trunc={out['cells'][f'ft_{split}']['truncated']}/{args.n}", flush=True)

    c = out["cells"]
    out["base_gap_dev_minus_test"] = c["base_dev"]["accuracy"] - c["base_test"]["accuracy"]
    out["ft_gap_dev_minus_test"] = c["ft_dev"]["accuracy"] - c["ft_test"]["accuracy"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\n=== GAP SUMMARY ===")
    print(f"base dev-test gap: {out['base_gap_dev_minus_test']:+.4f}")
    print(f"ft   dev-test gap: {out['ft_gap_dev_minus_test']:+.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
