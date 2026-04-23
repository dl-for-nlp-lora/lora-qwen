"""Configuration dataclass for the LoRA setup pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoraSetupConfig:
    """All tunable inputs for loading a base model and patching it with LoRA.

    When ``target_modules`` is ``None`` the model is returned untouched (config A).
    """

    model_name: str = "Qwen/Qwen3-1.7B-Base"
    target_modules: list[str] | None = None
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    dtype: str = "bfloat16"
    device: str = "auto"
    backend: str = "peft"
    name: str = "unnamed"

    @classmethod
    def from_yaml(cls, path: str | Path) -> LoraSetupConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        allowed = {f.name for f in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    def describe(self) -> str:
        tm = "none (base model)" if self.target_modules is None else ", ".join(self.target_modules)
        return (
            f"[{self.name}] model={self.model_name} | backend={self.backend} | "
            f"rank={self.rank} alpha={self.alpha} dropout={self.dropout} | "
            f"dtype={self.dtype} device={self.device} | targets=({tm})"
        )
