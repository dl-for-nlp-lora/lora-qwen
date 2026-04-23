"""Backend-agnostic LoRA entry points.

Everything above this file (scripts, sanity checks, training, evaluation)
depends only on these three functions — so swapping backends is a config-
only change:

    apply_lora(model, config)             # after loading the base model
    save_adapter(model, save_dir, config) # after training
    load_adapter(base_model, save_dir, config) # at eval time

The selected backend is determined by ``config.backend`` and must implement
the contract in :mod:`lora_qwen.lora.backend`.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from torch import nn

from lora_qwen.config import LoraSetupConfig
from lora_qwen.lora.backend import validate_backend_module


_BACKEND_MODULES: dict[str, str] = {
    "peft": "lora_qwen.lora.peft_backend",
    "custom": "lora_qwen.lora.custom_backend",
}


@lru_cache(maxsize=None)
def _load_backend(name: str) -> ModuleType:
    try:
        module_path = _BACKEND_MODULES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown LoRA backend '{name}'. Known: {sorted(_BACKEND_MODULES)}"
        ) from exc
    module = importlib.import_module(module_path)
    validate_backend_module(module)
    return module


def apply_lora(model: nn.Module, config: LoraSetupConfig) -> nn.Module:
    """Attach LoRA adapters to ``model`` using the configured backend.

    ``target_modules=None`` is a legal no-op (condition A, base model).
    """
    if config.target_modules is None:
        return model
    backend = _load_backend(config.backend)
    return backend.apply(model, config)


def save_adapter(model: nn.Module, save_dir: str | Path, config: LoraSetupConfig) -> None:
    """Persist adapter weights via the configured backend. No-op for condition A."""
    if config.target_modules is None:
        return
    backend = _load_backend(config.backend)
    backend.save(model, save_dir)


def load_adapter(
    base_model: nn.Module, save_dir: str | Path, config: LoraSetupConfig
) -> nn.Module:
    """Re-apply structure + restore saved weights onto a freshly loaded base model."""
    if config.target_modules is None:
        return base_model
    backend = _load_backend(config.backend)
    return backend.load(base_model, save_dir, config)
