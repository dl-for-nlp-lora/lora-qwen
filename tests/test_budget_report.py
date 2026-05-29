"""Unit tests for the LoRA budget report (pure dim math, no model download)."""

from __future__ import annotations

from torch import nn

from lora_qwen.config import LoraSetupConfig
from lora_qwen.model import lora_budget_report


class _Attn(nn.Module):
    """One decoder layer's projections with Qwen3-style GQA shapes.

    hidden=2048, 16 q-heads / 8 kv-heads, head_dim=128 -> q_proj out=2048,
    k/v_proj out=1024. The narrower k/v is exactly what breaks the naive
    "r * #targets" budget heuristic, so it's the case worth pinning.
    """

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(2048, 2048, bias=False)
        self.k_proj = nn.Linear(2048, 1024, bias=False)
        self.v_proj = nn.Linear(2048, 1024, bias=False)
        self.o_proj = nn.Linear(2048, 2048, bias=False)


class _Model(nn.Module):
    def __init__(self, n_layers: int = 28) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Attn() for _ in range(n_layers))


def _cfg(targets: list[str], rank: int) -> LoraSetupConfig:
    return LoraSetupConfig(target_modules=targets, rank=rank, alpha=2 * rank)


def test_gqa_makes_q_cost_more_than_v_at_equal_rank() -> None:
    model = _Model()
    q = lora_budget_report(model, _cfg(["q_proj"], 16))
    v = lora_budget_report(model, _cfg(["v_proj"], 16))
    # q_proj: 28 * 16 * (2048+2048); v_proj: 28 * 16 * (2048+1024)
    assert q.expected_trainable == 28 * 16 * 4096
    assert v.expected_trainable == 28 * 16 * 3072
    # The whole reason the iso-budget heuristic was wrong:
    assert q.expected_trainable > v.expected_trainable


def test_isobudget_ranks_equalize_q_and_qv() -> None:
    model = _Model()
    q = lora_budget_report(model, _cfg(["q_proj"], 14))
    qv = lora_budget_report(model, _cfg(["q_proj", "v_proj"], 8))
    # q@14: 28*14*4096 = 1,605,632 ; q,v@8: 28*8*(4096+3072) = 1,605,632
    assert q.expected_trainable == 1_605_632
    assert qv.expected_trainable == 1_605_632


def test_matched_modules_and_per_target_counts() -> None:
    model = _Model(n_layers=28)
    rep = lora_budget_report(model, _cfg(["q_proj", "k_proj", "v_proj", "o_proj"], 4))
    assert rep.matched_modules == 28 * 4
    assert rep.per_target == {"q_proj": 28, "k_proj": 28, "v_proj": 28, "o_proj": 28}
    assert rep.scaling == 2.0  # alpha=2r


def test_unmatched_target_yields_zero_budget() -> None:
    model = _Model()
    rep = lora_budget_report(model, _cfg(["does_not_exist"], 8))
    assert rep.expected_trainable == 0
    assert rep.matched_modules == 0


def test_report_matches_actual_trainable_after_apply() -> None:
    """The report must equal what a real wrap actually unfreezes."""
    from lora_qwen.lora.custom_backend import apply

    model = _Model()
    cfg = _cfg(["q_proj", "v_proj"], 8)
    rep = lora_budget_report(model, cfg)
    apply(model, cfg)
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert actual == rep.expected_trainable


def test_to_dict_is_json_serializable() -> None:
    import json

    model = _Model()
    rep = lora_budget_report(model, _cfg(["q_proj"], 14))
    json.dumps(rep.to_dict())  # must not raise


def test_zero_rank_scaling_is_safe() -> None:
    model = _Model()
    rep = lora_budget_report(model, _cfg(["q_proj"], 0))
    assert rep.scaling == 0.0  # no ZeroDivisionError
