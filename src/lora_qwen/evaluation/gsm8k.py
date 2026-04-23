"""GSM8K loader + prompt formatting.

Source: https://huggingface.co/datasets/openai/gsm8k (config 'main', split 'test')
Each row has ``question`` and ``answer`` (free-text solution terminated by
``#### N``).

We reuse the MetaMathQA prompt template so training and evaluation share the
exact same prefix — this is the whole point of aligning the two datasets.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from lora_qwen.data.metamath import format_prompt as _metamath_prompt
from lora_qwen.evaluation.extract import extract_number


HF_ID = "openai/gsm8k"


@dataclass
class GSM8KProblem:
    question: str
    answer_text: str         # full solution including "#### N"
    answer_number: float     # parsed ground-truth number


def format_prompt(question: str) -> str:
    """Same prefix as training (``Problem: ...\\n\\nSolution: ``)."""
    return _metamath_prompt(question)


def load_gsm8k(
    *,
    split: str = "test",
    max_examples: int | None = None,
    config: str = "main",
) -> Iterable[GSM8KProblem]:
    from datasets import load_dataset

    ds = load_dataset(HF_ID, config, split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    for row in ds:
        number = extract_number(row["answer"])
        if number is None:
            # GSM8K ground truth always has ####; skip any malformed row.
            continue
        yield GSM8KProblem(
            question=row["question"],
            answer_text=row["answer"],
            answer_number=number,
        )
