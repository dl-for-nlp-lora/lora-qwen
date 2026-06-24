"""GSM8K loader + prompt formatting.

Source: https://huggingface.co/datasets/openai/gsm8k (config 'main').
Each row has ``question`` and ``answer`` (free-text solution terminated by
``#### N``).

Three prompt builders (``zeroshot`` / ``fewshot`` / ``instruct``):

- :func:`format_prompt` — the bare ``Problem: ...\\n\\nSolution: `` prefix,
  identical to training. Used for the fine-tuned model (it learned to emit the
  ``#### N`` format and to stop, so it needs no instruction).
- :func:`format_prompt_fewshot` — instruction + 2 hand-written exemplars + the
  question.
- :func:`format_prompt_instruct` — instruction only (step-by-step + ``#### N``
  format spec), **no solved examples**.

Splits: ``test`` is the full GSM8K-test (final reporting only). ``dev`` returns
the fixed held-out slice of GSM8K-**train** defined in
:mod:`lora_qwen.data.gsm8k_train`, so sweeps validate without ever touching
test, and dev never overlaps the training data.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from lora_qwen.data.metamath import format_prompt as _metamath_prompt
from lora_qwen.evaluation.extract import extract_number


HF_ID = "openai/gsm8k"

# Few-shot exemplars. Hand-written (not from GSM8K) so there is zero risk of
# leaking a test/dev item, while still demonstrating chain-of-thought style,
# the "#### N" final-answer line, and stopping.
_FEWSHOT_EXEMPLARS: tuple[tuple[str, str], ...] = (
    (
        "A baker bakes 12 trays of 8 muffins each, then sells 53 muffins. "
        "How many muffins are left?",
        "He bakes 12 * 8 = 96 muffins. After selling 53, he has 96 - 53 = 43 left.\n#### 43",
    ),
    (
        "A tank holds 240 liters and is 3/4 full. Then 30 liters are drained. "
        "How many liters remain?",
        "Three quarters of 240 is 240 * 3/4 = 180 liters. After draining 30, "
        "180 - 30 = 150 remain.\n#### 150",
    ),
)

_FEWSHOT_INSTRUCTION = (
    "Solve the math word problem. Show the steps, then give the final answer on "
    'its own line as "#### <number>".'
)

# Instruction-only prompt: format + CoT nudge, no solved examples.
_INSTRUCT_INSTRUCTION = (
    "Think step by step. Show your reasoning, then write the final numeric "
    'answer alone on its own line as #### <number>.'
)


@dataclass
class GSM8KProblem:
    question: str
    answer_text: str         # full solution including "#### N"
    answer_number: float     # parsed ground-truth number


def format_prompt(question: str) -> str:
    """Zero-shot: same prefix as training (``Problem: ...\\n\\nSolution: ``)."""
    return _metamath_prompt(question)


def format_prompt_fewshot(question: str) -> str:
    """Few-shot prompt: instruction + 2 exemplars + the question.

    Each block reuses the exact ``Problem:/Solution:`` prefix so the format the
    base model is nudged toward matches what the FT model emits.
    """
    blocks = [_FEWSHOT_INSTRUCTION, ""]
    for q, a in _FEWSHOT_EXEMPLARS:
        blocks.append(f"{_metamath_prompt(q)}{a}")
        blocks.append("")
    blocks.append(_metamath_prompt(question).rstrip())
    return "\n".join(blocks)


def format_prompt_instruct(question: str) -> str:
    """Instruction-only prompt: step-by-step + ``#### N`` format, no exemplars."""
    return f"{_INSTRUCT_INSTRUCTION}\n\n{_metamath_prompt(question).rstrip()}"


PromptStyle = str  # "zeroshot" | "fewshot" | "instruct"


def resolve_prompt_fn(style: PromptStyle):
    """Map a prompt-style name to the corresponding formatter."""
    if style == "zeroshot":
        return format_prompt
    if style == "fewshot":
        return format_prompt_fewshot
    if style == "instruct":
        return format_prompt_instruct
    raise ValueError(f"prompt style must be 'zeroshot'|'fewshot'|'instruct', got {style!r}")


def load_gsm8k(
    *,
    split: str = "test",
    max_examples: int | None = None,
    config: str = "main",
) -> Iterable[GSM8KProblem]:
    """Yield GSM8K problems.

    ``split='test'`` -> full GSM8K-test. ``split='dev'`` -> the fixed held-out
    slice of GSM8K-train (see :mod:`lora_qwen.data.gsm8k_train`); use it for all
    sweep decisions so test stays untouched until the final run.
    """
    from datasets import load_dataset

    if split == "dev":
        from lora_qwen.data.gsm8k_train import gsm8k_partition

        ds = load_dataset(HF_ID, config, split="train")
        _, dev_idx = gsm8k_partition(len(ds))
        rows = (ds[i] for i in dev_idx)
    else:
        ds = load_dataset(HF_ID, config, split=split)
        rows = (ds[i] for i in range(len(ds)))

    yielded = 0
    for row in rows:
        if max_examples is not None and yielded >= max_examples:
            break
        number = extract_number(row["answer"])
        if number is None:
            # GSM8K ground truth always has ####; skip any malformed row.
            continue
        yielded += 1
        yield GSM8KProblem(
            question=row["question"],
            answer_text=row["answer"],
            answer_number=number,
        )
