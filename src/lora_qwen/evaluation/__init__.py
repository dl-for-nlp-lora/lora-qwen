"""Evaluation pipelines (generation-based task scoring).

Mirrors the data layer's structure: each task is a module that registers a
loader + scoring strategy, so adding new benchmarks later (HumanEval, IFEval,
MBPP, ...) only requires one file + a `@register_task` call.
"""

from lora_qwen.evaluation.extract import extract_number, numbers_match
from lora_qwen.evaluation.gsm8k import GSM8KProblem, format_prompt, load_gsm8k
from lora_qwen.evaluation.runner import EvalResult, evaluate_gsm8k

__all__ = [
    "EvalResult",
    "GSM8KProblem",
    "evaluate_gsm8k",
    "extract_number",
    "format_prompt",
    "load_gsm8k",
    "numbers_match",
]
