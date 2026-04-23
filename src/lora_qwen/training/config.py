"""Hyperparameters for the supervised fine-tuning loop."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    # Scheduling
    max_steps: int | None = None          # if set, stops at this many optimizer steps
    num_epochs: int = 1
    warmup_ratio: float = 0.03

    # Optimizer
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 1.0
    lr_schedule: str = "cosine"           # "cosine" | "linear" | "constant"

    # Batching
    per_device_batch_size: int = 2
    grad_accum_steps: int = 4             # effective batch = per_device * grad_accum

    # Mixed precision. Qwen on MPS keeps bf16 param dtype; loss computation
    # benefits from an autocast window only on CUDA. We keep this explicit.
    autocast_dtype: str = "bfloat16"

    # Logging
    log_every: int = 5
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        allowed = {f.name for f in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown train config keys in {path}: {sorted(unknown)}")
        if "betas" in raw and isinstance(raw["betas"], list):
            raw["betas"] = tuple(raw["betas"])
        return cls(**raw)

    def describe(self) -> str:
        eff_bs = self.per_device_batch_size * self.grad_accum_steps
        steps = f"max_steps={self.max_steps}" if self.max_steps else f"epochs={self.num_epochs}"
        return (
            f"[train] {steps} lr={self.learning_rate} schedule={self.lr_schedule} "
            f"bs={self.per_device_batch_size}x{self.grad_accum_steps}(eff={eff_bs}) "
            f"autocast={self.autocast_dtype} grad_clip={self.max_grad_norm}"
        )
