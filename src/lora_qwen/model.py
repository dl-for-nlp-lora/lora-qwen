"""Model + tokenizer loading and introspection helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

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


@dataclass
class LoRABudgetReport:
    """What a config *actually* costs, derived from real model dims — not assumed.

    Computed from the base model's matched ``nn.Linear`` shapes, so it is correct
    under GQA (where k/v_proj are narrower than q_proj) and for any model. The
    point is transparency: every run prints and persists rank/alpha/targets and
    the resulting trainable-param budget, so nothing about the experiment is
    hidden in a YAML the reader has to go dig up.
    """

    target_modules: list[str]
    rank: int
    alpha: int
    dropout: float
    matched_modules: int
    per_target: dict[str, int]            # leaf name -> # matched nn.Linear
    expected_trainable: int               # r * Σ(in+out) over matched modules
    total_params: int

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank if self.rank else 0.0

    @property
    def pct_of_total(self) -> float:
        return 100.0 * self.expected_trainable / self.total_params if self.total_params else 0.0

    def render(self, *, prefix: str = "") -> str:
        tgt = ", ".join(self.target_modules) if self.target_modules else "(none)"
        per = ", ".join(f"{k}×{v}" for k, v in sorted(self.per_target.items()))
        lines = [
            f"{prefix}LoRA budget report",
            f"{prefix}  targets        : {tgt}",
            f"{prefix}  rank           : {self.rank}",
            f"{prefix}  alpha          : {self.alpha}   (scaling α/r = {self.scaling:.4g})",
            f"{prefix}  dropout        : {self.dropout}",
            f"{prefix}  matched modules: {self.matched_modules}   ({per})",
            f"{prefix}  trainable      : {self.expected_trainable:,} "
            f"({self.pct_of_total:.3f}% of {self.total_params:,})",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "target_modules": self.target_modules,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "scaling": self.scaling,
            "matched_modules": self.matched_modules,
            "per_target": self.per_target,
            "expected_trainable_params": self.expected_trainable,
            "total_params": self.total_params,
            "pct_of_total": self.pct_of_total,
        }


def lora_budget_report(base_model: nn.Module, config: LoraSetupConfig) -> LoRABudgetReport:
    """Derive the LoRA budget for ``config`` from ``base_model``'s real dims.

    Pass the **unpatched** base model: this reads the ``nn.Linear`` shapes that
    would be wrapped. ``expected_trainable`` is ``Σ_modules r·(in+out)`` — the
    exact count of LoRA A/B entries that ``apply`` will make trainable.
    """
    targets = config.target_modules or []
    per_target: Counter[str] = Counter()
    expected = 0
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets:
            leaf = name.rsplit(".", 1)[-1]
            per_target[leaf] += 1
            expected += config.rank * (module.in_features + module.out_features)
    total = sum(p.numel() for p in base_model.parameters())
    return LoRABudgetReport(
        target_modules=list(targets),
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        matched_modules=sum(per_target.values()),
        per_target=dict(per_target),
        expected_trainable=expected,
        total_params=total,
    )
