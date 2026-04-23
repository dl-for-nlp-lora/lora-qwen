"""Generic generation-based scoring loop for math-task evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from lora_qwen.evaluation.extract import extract_number, numbers_match
from lora_qwen.evaluation.gsm8k import GSM8KProblem, format_prompt


@dataclass
class PerExample:
    index: int
    prompt: str
    completion: str
    gt: float
    predicted: float | None
    correct: bool


@dataclass
class EvalResult:
    name: str
    correct: int = 0
    total: int = 0
    wall_time_sec: float | None = None
    examples: list[PerExample] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"[{self.name}] accuracy={self.accuracy:.3f} "
            f"({self.correct}/{self.total})"
        )


def _generate_completion(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]
    with torch.inference_mode():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Return only the generated continuation, not the prompt echo.
    return tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=True)


def evaluate_gsm8k(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    problems: list[GSM8KProblem],
    *,
    name: str,
    max_new_tokens: int = 256,
    device: torch.device | None = None,
    verbose_every: int = 10,
) -> EvalResult:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    result = EvalResult(name=name, total=0, correct=0)
    start = time.perf_counter()
    try:
        for i, problem in enumerate(problems):
            prompt = format_prompt(problem.question)
            completion = _generate_completion(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens, device=device,
            )
            predicted = extract_number(completion)
            is_correct = numbers_match(predicted, problem.answer_number)
            result.examples.append(PerExample(
                index=i, prompt=prompt, completion=completion,
                gt=problem.answer_number, predicted=predicted, correct=is_correct,
            ))
            result.total += 1
            if is_correct:
                result.correct += 1

            if verbose_every and (i + 1) % verbose_every == 0:
                print(f"    [{name}] {i + 1}/{len(problems)}: "
                      f"acc so far {result.accuracy:.3f} ({result.correct}/{result.total})")
    finally:
        if was_training:
            model.train()

    result.wall_time_sec = time.perf_counter() - start
    return result
