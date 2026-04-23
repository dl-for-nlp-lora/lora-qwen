"""Sanity checks that verify the LoRA patching is structurally correct.

The key invariant: because LoRA initializes ``B = 0``, the delta ``BA x`` is zero
at init, so the patched model's logits must match the base model's (up to
negligible numerical noise). If this check fails, something is wrong with the
patching itself, not with training.

Usage pattern (peft wraps the base model in-place, so we must capture base
logits before applying LoRA)::

    base_logits = capture_logits(model, tokenizer, prompt)
    model = apply_lora(model, config)
    patched_logits = capture_logits(model, tokenizer, prompt)
    result = compare_logits(base_logits, patched_logits, dtype=model.dtype)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase


@dataclass
class IdentityCheckResult:
    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    num_mismatches: int
    total_elements: int
    tolerance: float

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Identity check {status}: "
            f"max|diff|={self.max_abs_diff:.3e} mean|diff|={self.mean_abs_diff:.3e} "
            f"(tol={self.tolerance:.0e}, {self.num_mismatches:,}/{self.total_elements:,} off)"
        )


def capture_logits(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str = "The capital of France is",
) -> torch.Tensor:
    """Return float32 CPU logits for ``prompt`` under a deterministic forward pass."""
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits.detach().float().cpu()
    finally:
        if was_training:
            model.train()


def _default_tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        return 1e-5
    if dtype in (torch.bfloat16, torch.float16):
        return 5e-3
    return 1e-4


def compare_logits(
    base_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    tolerance: float | None = None,
) -> IdentityCheckResult:
    if base_logits.shape != patched_logits.shape:
        raise RuntimeError(
            f"Logits shape mismatch: base={tuple(base_logits.shape)} "
            f"patched={tuple(patched_logits.shape)}"
        )
    tol = tolerance if tolerance is not None else _default_tolerance(dtype)
    diff = (base_logits - patched_logits).abs()
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    mismatches = int((diff > tol).sum())
    return IdentityCheckResult(
        passed=max_diff <= tol,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        num_mismatches=mismatches,
        total_elements=diff.numel(),
        tolerance=tol,
    )


def assert_identity(result: IdentityCheckResult) -> None:
    """Raise with a detailed diff report if the identity check failed."""
    if result.passed:
        return
    raise AssertionError(
        "LoRA identity check FAILED: patched model diverges from base model at init.\n"
        f"  tolerance      : {result.tolerance:.3e}\n"
        f"  max |diff|     : {result.max_abs_diff:.3e}\n"
        f"  mean |diff|    : {result.mean_abs_diff:.3e}\n"
        f"  mismatched     : {result.num_mismatches:,} / {result.total_elements:,}"
    )
