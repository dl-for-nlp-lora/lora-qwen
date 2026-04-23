"""Dynamic-padding collator for supervised fine-tuning.

Pads ``input_ids`` / ``attention_mask`` / ``labels`` to the longest sequence
in the batch. Padded positions in ``labels`` are set to ``-100`` so they are
ignored by cross-entropy (same convention as HuggingFace).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


IGNORE_INDEX = -100


@dataclass
class SupervisedCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError("Tokenizer must have pad_token_id set before collation")

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        input_ids = torch.full((len(features), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        labels = torch.full((len(features), max_len), IGNORE_INDEX, dtype=torch.long)

        for i, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[i, :n] = torch.tensor(f["input_ids"], dtype=torch.long)
            attention_mask[i, :n] = torch.tensor(f["attention_mask"], dtype=torch.long)
            labels[i, :n] = torch.tensor(f["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
