"""LoRA injection subpackage.

Public API:
    apply_lora(model, config)                   attach adapters
    save_adapter(model, save_dir, config)       persist after training
    load_adapter(base_model, save_dir, config)  restore for evaluation

Backends plug in via ``config.backend``:
    "peft"   : HuggingFace peft, known-good reference
    "custom" : the group's from-scratch implementation (see custom_backend.py)

See ``lora/README.md`` for the backend contract.
"""

from lora_qwen.lora.apply import apply_lora, load_adapter, save_adapter

__all__ = ["apply_lora", "load_adapter", "save_adapter"]
