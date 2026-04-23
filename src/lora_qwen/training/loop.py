"""Minimal supervised-FT training loop for LoRA-patched causal LMs.

Explicit PyTorch (no ``trl.SFTTrainer``) so every step is visible and easy to
diff against what colleagues will produce with their own LoRA backend.

Responsibility split:
    - This file: optimization orchestration (AdamW + schedule + grad accum +
      grad clip + logging). It is backend-agnostic — it only sees parameters
      via ``requires_grad``.
    - The LoRA backend: model structure, init, save/load (see
      ``lora_qwen.lora``).
    - The caller: saves the adapter after ``train()`` returns, via
      ``lora_qwen.lora.save_adapter``.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from lora_qwen.training.config import TrainConfig


@dataclass
class TrainResult:
    steps_done: int
    losses: list[tuple[int, float]] = field(default_factory=list)   # (step, avg loss)
    final_loss: float | None = None
    wall_time_sec: float | None = None

    def loss_trend(self) -> tuple[float | None, float | None]:
        """First and last logged loss, useful for a quick 'did it decrease?' check."""
        if not self.losses:
            return None, None
        return self.losses[0][1], self.losses[-1][1]


def _resolve_autocast_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name.lower()]


def _build_scheduler(
    optim: torch.optim.Optimizer, *, total_steps: int, warmup_steps: int, kind: str
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        if kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if kind == "linear":
            return 1.0 - progress
        if kind == "constant":
            return 1.0
        raise ValueError(f"Unknown lr_schedule '{kind}'")

    return LambdaLR(optim, lr_lambda=lr_lambda)


def _autocast_ctx(device: torch.device, dtype: torch.dtype):
    """Autocast where it actually helps (CUDA); no-op on MPS/CPU."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    # On MPS the parameters are already in bf16 and autocast is flaky in some
    # PyTorch builds, so we stay out of it.
    return contextlib.nullcontext()


def train(
    model: nn.Module,
    dataset: Dataset,
    collator,
    config: TrainConfig,
    *,
    device: torch.device,
) -> TrainResult:
    """Run supervised fine-tuning over ``dataset`` and return a :class:`TrainResult`.

    The model is trained in place. Saving is the caller's responsibility
    (use :func:`lora_qwen.lora.save_adapter`) so the training loop stays
    backend-agnostic.
    """
    torch.manual_seed(config.seed)

    loader = DataLoader(
        dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "No trainable parameters. Did `apply_lora` run? "
            "(Base model has all params frozen; LoRA params are what you train.)"
        )

    optim = AdamW(
        trainable,
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    micro_per_epoch = len(loader)
    macro_per_epoch = micro_per_epoch // config.grad_accum_steps
    total_macro = (
        config.max_steps
        if config.max_steps is not None
        else macro_per_epoch * config.num_epochs
    )
    warmup_steps = max(1, int(round(total_macro * config.warmup_ratio)))
    scheduler = _build_scheduler(
        optim, total_steps=total_macro, warmup_steps=warmup_steps, kind=config.lr_schedule
    )

    autocast_dtype = _resolve_autocast_dtype(config.autocast_dtype)
    result = TrainResult(steps_done=0)

    model.train()
    start = time.perf_counter()
    macro_step = 0
    micro_in_accum = 0
    running_loss = 0.0
    running_count = 0
    optim.zero_grad(set_to_none=True)

    try:
        epoch_budget = config.num_epochs if config.max_steps is None else 10_000
        for epoch in range(epoch_budget):
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                with _autocast_ctx(device, autocast_dtype):
                    out = model(**batch)
                    loss = out.loss / config.grad_accum_steps

                loss.backward()
                running_loss += float(out.loss.detach().cpu())
                running_count += 1
                micro_in_accum += 1

                if micro_in_accum == config.grad_accum_steps:
                    if config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                    optim.step()
                    scheduler.step()
                    optim.zero_grad(set_to_none=True)
                    macro_step += 1
                    micro_in_accum = 0

                    if macro_step % config.log_every == 0 or macro_step == total_macro:
                        avg = running_loss / max(1, running_count)
                        lr = scheduler.get_last_lr()[0]
                        print(
                            f"  step {macro_step:>4d}/{total_macro:<4d} | "
                            f"loss {avg:.4f} | lr {lr:.2e}"
                        )
                        result.losses.append((macro_step, avg))
                        running_loss = 0.0
                        running_count = 0

                    if macro_step >= total_macro:
                        raise StopIteration
            if config.max_steps is None and epoch + 1 >= config.num_epochs:
                break
    except StopIteration:
        pass

    result.steps_done = macro_step
    result.wall_time_sec = time.perf_counter() - start
    if result.losses:
        result.final_loss = result.losses[-1][1]
    return result
