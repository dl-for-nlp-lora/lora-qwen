"""GSM8K-train loader (for fine-tuning) with a held-out dev partition.

Source: https://huggingface.co/datasets/openai/gsm8k (config 'main').

Why this exists alongside ``metamath.py``: training directly on GSM8K-train
makes the training answer format (``#### N``, already present in the GSM8K
``answer`` field) identical to the GSM8K-test eval format, removing the
train/eval format mismatch that MetaMathQA introduced ("The answer is: N").

A fixed slice of GSM8K-**train** is reserved as a **dev** set so hyperparameter
sweeps never touch GSM8K-test. The partition is deterministic (fixed seed) and
shared with the evaluation side via :func:`gsm8k_partition`, so "train" and
"dev" mean the same indices everywhere.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from lora_qwen.data.metamath import format_prompt  # shared zero-shot prefix
from lora_qwen.data.registry import Example, register

HF_ID = "openai/gsm8k"
HF_CONFIG = "main"

# Size of the held-out dev slice carved from GSM8K-train, and the seed that
# fixes which examples land in it. Both are part of the experiment contract:
# changing them silently re-defines train/dev, so they live as constants.
DEV_SIZE = 1000
PARTITION_SEED = 1234


def gsm8k_partition(n_total: int) -> tuple[list[int], list[int]]:
    """Split ``range(n_total)`` deterministically into (train_idx, dev_idx).

    The last ``DEV_SIZE`` entries of a fixed-seed shuffle form the dev set.
    Returned index lists are sorted so iteration order is stable and identical
    across processes (training vs evaluation).
    """
    idx = list(range(n_total))
    random.Random(PARTITION_SEED).shuffle(idx)
    dev_idx = sorted(idx[-DEV_SIZE:])
    train_idx = sorted(idx[:-DEV_SIZE])
    return train_idx, dev_idx


def format_response(answer: str) -> str:
    """GSM8K answers already end in ``#### N``; just normalize whitespace."""
    return answer.strip() + "\n"


@register(
    "gsm8k_train",
    default_split="train",
    description=(
        "GSM8K-train (~7.5k grade-school math problems with CoT solutions "
        "ending in '#### N'). split='train' holds out a fixed dev slice "
        f"(last {DEV_SIZE} of a seed-{PARTITION_SEED} shuffle); split='dev' "
        "returns that slice. Training format matches GSM8K-test exactly."
    ),
)
def load_gsm8k_train(
    *,
    split: str = "train",
    max_examples: int | None = None,
    prompt_style: str = "zeroshot",
) -> Iterable[Example]:
    """Yield supervised GSM8K-train examples.

    ``prompt_style`` controls the (loss-masked) prompt the response is
    conditioned on during SFT:

    - ``"zeroshot"`` — the bare ``Problem:/Solution:`` prefix (default; the
      model must learn the format unaided).
    - ``"fewshot"`` — instruction + 2 exemplars.
    - ``"instruct"`` — instruction-only (step-by-step + ``#### N``), no exemplars.
    """
    from datasets import load_dataset

    if split not in {"train", "dev"}:
        raise ValueError(
            f"gsm8k_train supports split in {{'train','dev'}}, got '{split}'. "
            "(GSM8K-test is intentionally not reachable from the training loader.)"
        )
    if prompt_style not in {"zeroshot", "fewshot", "instruct"}:
        raise ValueError(
            f"prompt_style must be 'zeroshot'|'fewshot'|'instruct', got '{prompt_style}'"
        )

    if prompt_style == "zeroshot":
        _prompt_fn = format_prompt
    else:
        from lora_qwen.evaluation.gsm8k import resolve_prompt_fn
        _prompt_fn = resolve_prompt_fn(prompt_style)

    ds = load_dataset(HF_ID, HF_CONFIG, split="train")
    train_idx, dev_idx = gsm8k_partition(len(ds))
    keep = train_idx if split == "train" else dev_idx
    if max_examples is not None:
        keep = keep[:max_examples]

    for i in keep:
        row = ds[i]
        yield Example(
            prompt=_prompt_fn(row["question"]),
            response=format_response(row["answer"]),
        )
