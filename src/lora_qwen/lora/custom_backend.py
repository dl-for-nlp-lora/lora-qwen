"""Group's own LoRA backend — THIS IS WHERE YOU IMPLEMENT.

You only need to fill in this file. The surrounding infrastructure (model
loading, data pipeline, training loop, evaluation, sanity checks) is already
wired up and will use your implementation as soon as ``backend: "custom"`` is
set in the YAML config.

Contract (see ``backend.py`` for the formal definition):

    apply(model, config) -> nn.Module
        Return a new model where every ``nn.Linear`` whose leaf name matches
        ``config.target_modules`` has been replaced by a ``LinearWithLoRA``
        wrapper. Base weights frozen, LoRA params trainable, B init to zero
        (so forward is identical to the base model right after apply).

    save(model, save_dir) -> None
        Dump only the LoRA parameters (a few MB). The base model weights are
        NOT saved — they're reconstructed by re-loading the base model.

    load(base_model, save_dir, config) -> nn.Module
        Call apply(base_model, config) to reconstruct the module structure,
        then fill the LoRA tensors from save_dir.

Verification workflow:

    # Injection correctness: patched model must match base at init (B=0).
    $ python scripts/smoke_setup.py --config configs/all_linear.yaml  \\
        # after setting `backend: "custom"` in that yaml, or via a copy

    # End-to-end training: loss must decrease.
    $ python scripts/smoke_ft.py   --lora configs/<your-custom-yaml>

    # Results parity: compare against the peft reference on the same seed
    # and same adapter weights.

The paper you're reproducing: Hu et al. 2021, https://arxiv.org/abs/2106.09685
Relevant sections: §4.1 (method), §4.2 (target-module choice).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from lora_qwen.config import LoraSetupConfig


# =========================================================================
# TODO(group): the three classes/functions below are the ones to implement.
# =========================================================================


class LoRALayer(nn.Module):
    """Low-rank update: ``ΔWx = (α / r) · B(A(x))``.

    Init convention (from the paper):
        A : Kaiming-uniform on a ``(rank, in_features)`` matrix
        B : zeros on a ``(out_features, rank)`` matrix
    so the product BA is zero at initialization and the patched model's
    forward is identical to the base model's.

    Args:
        in_features:  input dimension of the wrapped Linear
        out_features: output dimension of the wrapped Linear
        rank:         LoRA rank r
        alpha:        LoRA scaling numerator; effective scale = alpha / rank
        dropout:      dropout probability applied to x before A
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        # Scaling as per paper
        self.scaling = alpha/rank
        
        # Matrix A: dims (r x in)
        self.lora_A = nn.Parameter(torch.empty((rank, in_features)))
        # Matrix B: dims (out x r)
        self.lora_B = nn.Parameter(torch.empty((out_features, rank)))
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Initialise A with Kaiming-uniform (with a=5**0.5)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        # Initialise B with zeros
        nn.init.zeros_(self.lora_B)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Applying dropout to input
        x = self.dropout(x)
        # Projecting down: x @ lora_A.T - resulting shape (batch, rank)
        after_A = F.linear(x, self.lora_A)
        # Projecting back up: after_A @ lora_B.T - resulting shape (batch, out_features)
        after_B = F.linear(after_A, self.lora_B)
        # Apply scaling factor (alpha / r)
        return after_B * self.scaling


class LinearWithLoRA(nn.Module):
    """Frozen base ``nn.Linear`` plus a parallel ``LoRALayer``.

    Forward: ``base(x) + lora(x)``. Base weights are frozen in __init__.
    """

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        # Store original layer
        self.base = base
        
        # Freeze params of original layer
        for param in self.base.parameters():
            param.requires_grad = False
            
        # Create parallel LoRA path
        self.lora = LoRALayer(
            in_features=base.in_features,
            out_features=base.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Combine frozen input with trained delta
        return self.base(x) + self.lora(x)


def apply(model: nn.Module, config: LoraSetupConfig) -> nn.Module:
    """Swap every target ``nn.Linear`` for a ``LinearWithLoRA``, freeze the rest.

    Steps:
        1. Freeze every parameter in `model`.
        2. Walk `model.named_modules()`; when a leaf's name is in
           `config.target_modules`, replace it with a `LinearWithLoRA`.
        3. Return the (now mutated) model.

    Hint: use ``model.get_submodule(parent)`` + ``setattr`` to replace a
    named submodule.
    """
    # TODO(group): implement the patching walk.
    raise NotImplementedError("custom_backend.apply")


def save(model: nn.Module, save_dir: str | Path) -> None:
    """Dump only trainable tensors (LoRA A and B) as a plain ``state_dict``."""
    # TODO(group): build a dict of the `requires_grad=True` params, save with
    # ``torch.save`` or ``safetensors.save_file`` to ``save_dir/adapter.pt``.
    raise NotImplementedError("custom_backend.save")


def load(base_model: nn.Module, save_dir: str | Path, config: LoraSetupConfig) -> nn.Module:
    """Re-apply LoRA structure + load saved adapter tensors."""
    # TODO(group): call apply(base_model, config), then torch.load the
    # adapter file and `load_state_dict(..., strict=False)` onto the model.
    raise NotImplementedError("custom_backend.load")
