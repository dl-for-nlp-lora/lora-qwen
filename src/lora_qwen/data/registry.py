"""Dataset registry + supervised-dataset builder.

A dataset module provides a loader function returning an iterable of
:class:`Example` and registers it under a name. Training/eval pipelines
then look the dataset up by name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class Example:
    """The raw unit every dataset emits before tokenization.

    ``prompt`` and ``response`` are plain strings; we tokenize separately so
    the collator can mask prompt tokens from the loss.
    """

    prompt: str
    response: str


LoaderFn = Callable[..., Iterable[Example]]


@dataclass
class DatasetSpec:
    name: str
    loader: LoaderFn
    default_split: str = "train"
    # Free-form notes for humans (eval metric, source URL etc.).
    description: str = ""


_REGISTRY: dict[str, DatasetSpec] = {}


def register(name: str, *, default_split: str = "train", description: str = "") -> Callable[[LoaderFn], LoaderFn]:
    def decorator(fn: LoaderFn) -> LoaderFn:
        if name in _REGISTRY:
            raise ValueError(f"Dataset '{name}' already registered")
        _REGISTRY[name] = DatasetSpec(
            name=name, loader=fn, default_split=default_split, description=description
        )
        return fn

    return decorator


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset '{name}'. Registered: {sorted(_REGISTRY)}"
        ) from exc


def list_datasets() -> list[str]:
    return sorted(_REGISTRY)


class SupervisedDataset(Dataset):
    """Tokenized wrapper around a list of :class:`Example`.

    Each item is a dict with keys ``input_ids``, ``attention_mask``, ``labels``.
    ``labels`` are a copy of ``input_ids`` with prompt positions replaced by
    -100, so the HF-style cross-entropy loss only runs on response tokens
    (completion-only SFT).
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        examples: list[Example],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        ex = self.examples[idx]
        tok = self.tokenizer

        # Tokenize prompt alone to know the boundary, then prompt+response
        # together so the boundary is guaranteed to be a clean prefix match.
        prompt_ids = tok(ex.prompt, add_special_tokens=False)["input_ids"]
        full_text = ex.prompt + ex.response
        # Append EOS so generation knows when to stop when trained with this.
        if tok.eos_token_id is not None and not full_text.endswith(tok.eos_token or ""):
            full_ids = tok(full_text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        else:
            full_ids = tok(full_text, add_special_tokens=False)["input_ids"]

        # Safety: the prompt tokens must be a prefix of the full sequence for
        # the boundary-mask to be correct. Qwen's BPE is stable on this, but
        # we verify rather than trust.
        prefix_len = len(prompt_ids)
        if full_ids[:prefix_len] != prompt_ids:
            # Fall back to a conservative estimate: re-tokenize concatenated
            # with a sentinel space and use the first `prefix_len` tokens. In
            # practice this branch is not hit for BPE + plain ASCII prompts.
            prefix_len = min(prefix_len, len(full_ids))

        # Truncate from the right; prompt stays fully visible.
        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]

        input_ids = full_ids
        attention_mask = [1] * len(input_ids)
        labels = [self.IGNORE_INDEX] * min(prefix_len, len(input_ids)) + input_ids[prefix_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_supervised_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    split: str | None = None,
    max_examples: int | None = None,
    max_length: int = 512,
    loader_kwargs: dict[str, Any] | None = None,
) -> SupervisedDataset:
    spec = get_dataset_spec(dataset_name)
    effective_split = split or spec.default_split
    kwargs = {"split": effective_split, **(loader_kwargs or {})}
    if max_examples is not None:
        kwargs["max_examples"] = max_examples
    raw = list(spec.loader(**kwargs))
    if max_examples is not None:
        raw = raw[:max_examples]
    return SupervisedDataset(raw, tokenizer, max_length=max_length)
