"""MetaMathQA loader.

Source: https://huggingface.co/datasets/meta-math/MetaMathQA
Fields we use:
    - ``query``    : the math problem
    - ``response`` : the chain-of-thought answer, ending in "The answer is: N"

Prompt template matches what GSM8K evaluation will use so training and
evaluation share the exact same format (important for the ``####`` answer
extraction to work after fine-tuning).
"""

from __future__ import annotations

from collections.abc import Iterable

from lora_qwen.data.registry import Example, register


HF_ID = "meta-math/MetaMathQA"

PROMPT_TEMPLATE = "Problem: {question}\n\nSolution: "


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def format_response(response: str) -> str:
    # Trailing newline gives the tokenizer a clean break before EOS.
    return response.strip() + "\n"


@register(
    "metamath",
    default_split="train",
    description=(
        "MetaMathQA (~395k math word problems with CoT solutions). "
        "Paired with GSM8K for evaluation: the 'The answer is: N' suffix "
        "in the response aligns with #### answer extraction after FT."
    ),
)
def load_metamath(
    *,
    split: str = "train",
    max_examples: int | None = None,
    seed: int = 0,
    shuffle: bool = True,
) -> Iterable[Example]:
    from datasets import load_dataset

    ds = load_dataset(HF_ID, split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    for row in ds:
        yield Example(
            prompt=format_prompt(row["query"]),
            response=format_response(row["response"]),
        )
