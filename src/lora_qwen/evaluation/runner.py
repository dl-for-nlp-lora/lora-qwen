"""Generic generation-based scoring loop for math-task scoring.

Generation is batched: GSM8K-style decoding is autoregressive and therefore
the wall-time bottleneck of the whole pipeline. Sending multiple prompts
through ``model.generate`` at once amortises the per-step kernel launches
and the KV-cache setup, giving roughly an N x speedup up to the point where
memory becomes the limit. See :func:`evaluate_gsm8k` for the batch-size
heuristic.
"""

from __future__ import annotations

import time
from collections.abc import Callable
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


def _generate_completions_batched(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    *,
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    """Generate continuations for ``prompts`` in a single ``model.generate`` call.

    Left-padding is required for autoregressive decoding: with right-padding
    the model would condition on trailing PAD tokens and produce garbage.
    We restore the tokenizer's original ``padding_side`` afterwards so we
    don't surprise the training path, which expects right-padded inputs.
    """
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)
        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # With left-padding every row has the same prompt-block width, so a
        # single slice gives clean continuations for the whole batch.
        gen_only = out_ids[:, enc["input_ids"].shape[1]:]
        return tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    finally:
        tokenizer.padding_side = original_side


def _default_batch_size(device: torch.device) -> int:
    """Pick a generation batch size that's safe on small GPUs but worth it.

    Sized to fit a Qwen3-1.7B forward + KV cache (prompt <= ~512 tok, up to
    256 new tokens) on a 16 GB GPU (T4) with headroom. Larger GPUs (L4 24G,
    A40 48G, A100 40/80G) will still fit a 4-8x larger batch -- bump
    --batch-size on the command line if you have the VRAM.

    MPS sticks to batch=1: ``model.generate`` with batched left-padded
    inputs has been flaky on Apple Silicon backends in recent PyTorch
    releases, and MPS is single-stream anyway so we don't lose much.
    """
    if device.type == "cuda":
        return 16
    return 1


def evaluate_gsm8k(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    problems: list[GSM8KProblem],
    *,
    name: str,
    max_new_tokens: int = 256,
    device: torch.device | None = None,
    batch_size: int | None = None,
    verbose_every: int = 10,
    prompt_fn: Callable[[str], str] = format_prompt,
) -> EvalResult:
    """Score ``problems`` and return an :class:`EvalResult`.

    ``batch_size`` defaults to a hardware-appropriate value via
    :func:`_default_batch_size`. Override it explicitly when you know
    your VRAM budget.

    ``prompt_fn`` builds the prompt from a question; defaults to the zero-shot
    training prefix. Pass ``format_prompt_fewshot`` or ``format_prompt_instruct``
    to evaluate under a different prompt style.
    """
    device = device or next(model.parameters()).device
    if batch_size is None:
        batch_size = _default_batch_size(device)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    was_training = model.training
    model.eval()

    result = EvalResult(name=name, total=0, correct=0)
    start = time.perf_counter()
    last_logged = 0
    try:
        for batch_start in range(0, len(problems), batch_size):
            batch = problems[batch_start : batch_start + batch_size]
            prompts = [prompt_fn(p.question) for p in batch]
            completions = _generate_completions_batched(
                model, tokenizer, prompts,
                max_new_tokens=max_new_tokens, device=device,
            )
            for offset, (problem, prompt, completion) in enumerate(
                zip(batch, prompts, completions)
            ):
                predicted = extract_number(completion)
                is_correct = numbers_match(predicted, problem.answer_number)
                result.examples.append(PerExample(
                    index=batch_start + offset,
                    prompt=prompt,
                    completion=completion,
                    gt=problem.answer_number,
                    predicted=predicted,
                    correct=is_correct,
                ))
                result.total += 1
                if is_correct:
                    result.correct += 1

            if verbose_every and result.total - last_logged >= verbose_every:
                print(
                    f"    [{name}] {result.total}/{len(problems)}: "
                    f"acc so far {result.accuracy:.3f} "
                    f"({result.correct}/{result.total})"
                )
                last_logged = result.total
    finally:
        if was_training:
            model.train()

    result.wall_time_sec = time.perf_counter() - start
    return result
