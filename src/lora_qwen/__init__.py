"""LoRA reproduction + extension on Qwen3-1.7B-Base."""

from lora_qwen.config import LoraSetupConfig
from lora_qwen.lora import apply_lora, load_adapter, save_adapter
from lora_qwen.model import (
    list_linear_modules,
    load_model_and_tokenizer,
    print_trainable_params,
    summarize_linear_modules,
)
from lora_qwen.sanity import (
    IdentityCheckResult,
    assert_identity,
    capture_logits,
    compare_logits,
)

__all__ = [
    "IdentityCheckResult",
    "LoraSetupConfig",
    "apply_lora",
    "assert_identity",
    "capture_logits",
    "compare_logits",
    "list_linear_modules",
    "load_adapter",
    "load_model_and_tokenizer",
    "print_trainable_params",
    "save_adapter",
    "summarize_linear_modules",
]
