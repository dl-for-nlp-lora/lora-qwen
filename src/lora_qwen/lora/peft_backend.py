"""Reference LoRA backend — HuggingFace ``peft``.

Deliberately thin. This is the ground-truth implementation the infrastructure
(loading, sanity checks, training, evaluation) is validated against. Colleagues
implement their own in :mod:`lora_qwen.lora.custom_backend` and compare.

Implements the three-function contract from :mod:`lora_qwen.lora.backend`:
``apply``, ``save``, ``load``.
"""

from __future__ import annotations

from pathlib import Path

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn

from lora_qwen.config import LoraSetupConfig


def _make_config(config: LoraSetupConfig) -> LoraConfig:
    assert config.target_modules, "peft backend requires non-empty target_modules"
    return LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def apply(model: nn.Module, config: LoraSetupConfig) -> nn.Module:
    return get_peft_model(model, _make_config(config))


def save(model: nn.Module, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "save_pretrained"):
        raise TypeError(
            "peft.save() called on a non-PeftModel. Did you apply a different backend?"
        )
    model.save_pretrained(save_dir)


def load(base_model: nn.Module, save_dir: str | Path, config: LoraSetupConfig) -> nn.Module:
    # `config` is unused for peft (the adapter config is stored alongside the
    # weights), but the signature must match the contract.
    del config
    return PeftModel.from_pretrained(base_model, save_dir)
