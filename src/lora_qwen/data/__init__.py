"""Dataset loading, prompt formatting, and completion-only masking.

Architecture (designed to scale to multiple datasets later):

- ``Example``            : minimal schema every dataset emits (prompt + response).
- ``DatasetSpec``        : how to load and iterate raw samples from a source.
- ``registry``           : maps ``dataset name -> DatasetSpec``. A new dataset
                           adds one module (e.g. ``metamath.py``) that registers
                           itself via ``@register("name")``.
- ``build_supervised_dataset``: turns raw samples into tokenized tensors with
                                completion-only labels (prompt tokens masked
                                with -100 so loss only flows through the
                                response).
- ``SupervisedCollator`` : dynamic padding + label-masking at batch time.

The interface is deliberately narrow so colleagues can drop new datasets in
without touching training code.
"""

from lora_qwen.data.collator import SupervisedCollator
from lora_qwen.data.config import DataConfig
from lora_qwen.data.registry import (
    DatasetSpec,
    Example,
    build_supervised_dataset,
    get_dataset_spec,
    list_datasets,
    register,
)

# Importing concrete datasets triggers their @register() side-effect.
from lora_qwen.data import gsm8k_train, metamath  # noqa: F401

__all__ = [
    "DataConfig",
    "DatasetSpec",
    "Example",
    "SupervisedCollator",
    "build_supervised_dataset",
    "get_dataset_spec",
    "list_datasets",
    "register",
]
