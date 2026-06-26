"""Diagnose the dev>>test accuracy gap: is the held-out dev slice contaminated
by near-duplicate train questions (so a model trained on train memorizes dev)?

For every dev question we find its most similar TRAIN question (the data the
adapter actually saw) via token-set Jaccard similarity, and record whether the
gold answer matches. A large fraction of dev items with a high-similarity,
same-answer train neighbor explains an inflated dev number that does not
transfer to the clean GSM8K-test split.

CPU-only, no GPU. Run locally:
    python scripts/diag_dev_leakage.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from datasets import load_dataset  # noqa: E402

from lora_qwen.data.gsm8k_train import gsm8k_partition  # noqa: E402
from lora_qwen.evaluation.gsm8k import extract_number  # noqa: E402

WORD = re.compile(r"[a-z0-9]+")


def toks(s: str) -> set[str]:
    return set(WORD.findall(s.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    train_idx, dev_idx = gsm8k_partition(len(ds))

    train_q = [(i, ds[i]["question"], extract_number(ds[i]["answer"])) for i in train_idx]
    train_tok = [(i, toks(q), ans) for i, q, ans in train_q]

    dev = [(i, ds[i]["question"], extract_number(ds[i]["answer"])) for i in dev_idx]

    # Bucket train by a cheap length signature to limit comparisons.
    buckets: dict[int, list] = {}
    for i, tk, ans in train_tok:
        buckets.setdefault(len(tk) // 4, []).append((i, tk, ans))

    sims = []
    same_ans_hits = {0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0}
    near_dup_same_ans = 0
    examples = []
    for _di, dq, dans in dev:
        dtk = toks(dq)
        cands = []
        for b in (len(dtk) // 4 - 1, len(dtk) // 4, len(dtk) // 4 + 1):
            cands += buckets.get(b, [])
        best_s, best_i, best_ans = 0.0, -1, None
        for ti, ttk, tans in cands:
            s = jaccard(dtk, ttk)
            if s > best_s:
                best_s, best_i, best_ans = s, ti, tans
        sims.append(best_s)
        same = (best_ans == dans)
        for thr in same_ans_hits:
            if best_s >= thr and same:
                same_ans_hits[thr] += 1
        if best_s >= 0.8 and same:
            near_dup_same_ans += 1
            if len(examples) < 4:
                examples.append((round(best_s, 3), dans, dq[:90], ds[best_i]["question"][:90]))

    n = len(dev)
    import statistics
    print(f"dev items: {n}")
    print(f"max-similarity to any TRAIN question: "
          f"mean={statistics.mean(sims):.3f} median={statistics.median(sims):.3f} "
          f"p90={sorted(sims)[int(0.9*n)]:.3f} max={max(sims):.3f}")
    print("\ndev items whose closest train question is a near-duplicate WITH THE "
          "SAME gold answer:")
    for thr, c in sorted(same_ans_hits.items()):
        print(f"  Jaccard >= {thr}: {c}/{n} ({100*c/n:.1f}%)")
    print(f"\n=> {near_dup_same_ans}/{n} ({100*near_dup_same_ans/n:.1f}%) dev items have a "
          f">=0.80-similar train neighbor with identical answer (memorization route).")
    print("\nexamples (sim | gold | dev_q | nearest_train_q):")
    for s, a, dq, tq in examples:
        print(f"  sim={s} ans={a}\n    dev:   {dq}...\n    train: {tq}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
