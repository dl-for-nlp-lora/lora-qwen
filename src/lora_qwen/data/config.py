"""Data-side config (separated from ``LoraSetupConfig`` to keep concerns clean)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    name: str = "metamath"
    dataset: str = "metamath"          # registry key
    split: str = "train"
    max_examples: int | None = None
    max_length: int = 512

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        allowed = {f.name for f in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown data config keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    def describe(self) -> str:
        return (
            f"[data:{self.name}] dataset={self.dataset} split={self.split} "
            f"max_examples={self.max_examples} max_length={self.max_length}"
        )
