"""LoRA backend contract.

A backend is a Python module that exposes three functions:

- ``apply(model, config) -> nn.Module``
    Attach LoRA adapters to the (already-loaded) base model and return the
    patched model. The returned model must have every base parameter frozen
    (``requires_grad=False``) and only the new LoRA parameters trainable.

    Invariant: immediately after ``apply`` the patched model's forward pass
    must be numerically identical to the base model (because B is zero-
    initialized). This is checked by :func:`lora_qwen.sanity.compare_logits`
    and must pass before any training is attempted.

- ``save(model, save_dir) -> None``
    Persist adapter weights to ``save_dir``. Format is backend-specific.
    What matters is that ``load`` can reconstruct the same patched model
    from the same base model + ``save_dir``.

- ``load(base_model, save_dir, config) -> nn.Module``
    Re-apply adapters to a freshly loaded base model and restore the
    trained weights from ``save_dir``. Must return the equivalent of
    ``apply(base_model, config)`` followed by loading the trained tensors.

This module defines a :class:`LoRABackend` Protocol (for type-checkers / IDE
hints) and a runtime :func:`validate_backend_module` helper so the dispatcher
fails with a clear message if a backend is missing a function.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Protocol, runtime_checkable

from torch import nn

from lora_qwen.config import LoraSetupConfig


@runtime_checkable
class LoRABackend(Protocol):
    """Structural type that every LoRA backend must satisfy."""

    def apply(self, model: nn.Module, config: LoraSetupConfig) -> nn.Module: ...

    def save(self, model: nn.Module, save_dir: str | Path) -> None: ...

    def load(
        self, base_model: nn.Module, save_dir: str | Path, config: LoraSetupConfig
    ) -> nn.Module: ...


_REQUIRED = ("apply", "save", "load")


def validate_backend_module(module: ModuleType) -> None:
    """Raise a helpful error if ``module`` doesn't implement the backend contract."""
    missing = [name for name in _REQUIRED if not callable(getattr(module, name, None))]
    if missing:
        raise TypeError(
            f"Backend '{module.__name__}' is missing required callables: {missing}. "
            f"Every LoRA backend must define {list(_REQUIRED)} — see "
            f"lora_qwen/lora/README.md for the contract."
        )
