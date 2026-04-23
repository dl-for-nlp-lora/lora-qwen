"""Model + tokenizer loading and introspection helpers."""

from __future__ import annotations

from collections import Counter

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from lora_qwen.config import LoraSetupConfig


_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_device(requested: str) -> torch.device:
    """Resolve 'auto' to the best locally available device (mps > cuda > cpu)."""
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype '{name}'. Known: {sorted(_DTYPE_MAP)}") from exc


def load_model_and_tokenizer(
    config: LoraSetupConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device]:
    """Load the base model and tokenizer according to ``config``.

    Returns the model on the resolved device in inference mode, the tokenizer
    (with pad_token guaranteed), and the resolved device for downstream use.
    """
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=dtype)
    model.to(device)
    model.eval()

    return model, tokenizer, device


def list_linear_modules(model: nn.Module) -> list[tuple[str, tuple[int, int]]]:
    """Return [(qualified_name, (in_features, out_features)), ...] for every nn.Linear."""
    return [
        (name, (m.in_features, m.out_features))
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
    ]


def summarize_linear_modules(model: nn.Module) -> dict[str, int]:
    """Count nn.Linear modules grouped by their leaf name (e.g. 'q_proj', 'lm_head')."""
    counts: Counter[str] = Counter()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.rsplit(".", 1)[-1]
            counts[leaf] += 1
    return dict(counts)


def print_trainable_params(model: nn.Module, *, prefix: str = "") -> tuple[int, int]:
    """Log and return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(f"{prefix}Trainable params: {trainable:,} / {total:,} ({pct:.3f}%)")
    return trainable, total
