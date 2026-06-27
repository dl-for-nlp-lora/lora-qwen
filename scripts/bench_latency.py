"""Per-sample latency benchmark: base vs LoRA-FT on GSM8K-test.

Why this exists: the headline question "how much faster is an FT run per sample"
is really about *generation length*. The base model (especially when it answers
in a verbose, self-narrating or code style) emits many more tokens per question
than the terse, `#### N`-formatted FT model, and decoding is roughly linear in
generated tokens. We measure this directly.

We run at **batch_size=1** so the number is a clean per-sample wall time (no
padding / batching confounds), greedy decoding, identical 512-token cap and
instruct prompt for both models. For each of N sampled test questions we record:
  - generated-token count (EOS-terminated length, the real driver)
  - wall time for that single generate() call
for the base model and for the base+adapter (FT). Output is a JSON with per-
example rows + aggregates, consumed by scripts/plot_ft_analysis.py.

Usage (on a pod):
    python scripts/bench_latency.py \
        --base-cfg configs/qwen2_1.5b/base.yaml \
        --ft-cfg   results_headroom/qwen2_1.5b/best_lora.yaml \
        --adapter  checkpoints/headroom/qwen2_1.5b/e2_attention_r1 \
        --n 200 --out results_headroom/qwen2_1.5b/latency.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lora_qwen.config import LoraSetupConfig
from lora_qwen.evaluation import load_gsm8k, resolve_prompt_fn
from lora_qwen.evaluation.extract import extract_number, numbers_match
from lora_qwen.lora import load_adapter
from lora_qwen.model import load_model_and_tokenizer


def _bench_model(model, tokenizer, device, problems, prompt_fn, mnt) -> list[dict]:
    model.eval()
    rows = []
    eos_id = tokenizer.eos_token_id
    # warm-up (cuda kernels / autograd graph) so the first timed call is fair
    with torch.no_grad():
        warm = tokenizer(prompt_fn(problems[0].question), return_tensors="pt").to(device)
        model.generate(**warm, max_new_tokens=8, do_sample=False,
                        pad_token_id=eos_id)
    for p in problems:
        enc = tokenizer(prompt_fn(p.question), return_tensors="pt").to(device)
        in_len = enc["input_ids"].shape[1]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                                  pad_token_id=eos_id)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen_ids = out[0, in_len:]
        gen_len = int(gen_ids.shape[0])
        # did it stop on EOS (self-terminated) or hit the cap (truncated)?
        truncated = bool((gen_ids != eos_id).all().item()) and gen_len >= mnt
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extract_number(text)
        rows.append({
            "gen_tokens": gen_len,
            "wall_sec": dt,
            "truncated": truncated,
            "correct": numbers_match(pred, p.answer_number),
        })
    return rows


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    toks = [r["gen_tokens"] for r in rows]
    secs = [r["wall_sec"] for r in rows]
    return {
        "n": n,
        "mean_gen_tokens": sum(toks) / n,
        "median_gen_tokens": sorted(toks)[n // 2],
        "mean_wall_sec": sum(secs) / n,
        "median_wall_sec": sorted(secs)[n // 2],
        "total_wall_sec": sum(secs),
        "truncated_frac": sum(r["truncated"] for r in rows) / n,
        "accuracy": sum(r["correct"] for r in rows) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cfg", required=True)
    ap.add_argument("--ft-cfg", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--prompt", default="instruct")
    ap.add_argument("--mnt", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt_fn = resolve_prompt_fn(args.prompt)
    problems = list(load_gsm8k(split="test", max_examples=args.n))
    print(f"benchmarking on {len(problems)} test problems (batch=1, prompt={args.prompt})")

    base_cfg = LoraSetupConfig.from_yaml(args.base_cfg)
    bm, btok, device = load_model_and_tokenizer(base_cfg)
    print("timing base ...", flush=True)
    base_rows = _bench_model(bm, btok, device, problems, prompt_fn, args.mnt)
    del bm
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ft_cfg = LoraSetupConfig.from_yaml(args.ft_cfg)
    fm, ftok, device = load_model_and_tokenizer(ft_cfg)
    fm = load_adapter(fm, args.adapter, ft_cfg)
    fm.to(device)
    print("timing ft ...", flush=True)
    ft_rows = _bench_model(fm, ftok, device, problems, prompt_fn, args.mnt)

    base_agg, ft_agg = _agg(base_rows), _agg(ft_rows)
    out = {
        "prompt": args.prompt, "max_new_tokens": args.mnt,
        "device": str(device),
        "base": {**base_agg, "rows": base_rows},
        "ft": {**ft_agg, "rows": ft_rows},
        "speedup_wall": base_agg["mean_wall_sec"] / ft_agg["mean_wall_sec"],
        "token_ratio": base_agg["mean_gen_tokens"] / ft_agg["mean_gen_tokens"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nbase:  {base_agg['mean_gen_tokens']:.0f} tok  "
          f"{base_agg['mean_wall_sec']*1000:.0f} ms/sample")
    print(f"ft:    {ft_agg['mean_gen_tokens']:.0f} tok  "
          f"{ft_agg['mean_wall_sec']*1000:.0f} ms/sample")
    print(f"speedup x{out['speedup_wall']:.2f}  token-ratio x{out['token_ratio']:.2f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
