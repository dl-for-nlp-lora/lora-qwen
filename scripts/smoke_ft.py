"""End-to-end smoke fine-tune.

Picks a LoRA config (default: all-linear), a data config (default: metamath),
and a tiny training recipe, then runs a short SFT and confirms the loss is
trending downward. This is a *pipeline validation*, not a real experiment —
replace the configs for the actual runs.

The script is backend-agnostic: whatever ``backend:`` the LoRA config
specifies is what gets used, via ``apply_lora`` / ``save_adapter``.

Run:
    python scripts/smoke_ft.py
    python scripts/smoke_ft.py --lora configs/attention_only.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lora_qwen.config import LoraSetupConfig  # noqa: E402
from lora_qwen.data import (  # noqa: E402
    DataConfig,
    SupervisedCollator,
    build_supervised_dataset,
)
from lora_qwen.lora import apply_lora, save_adapter  # noqa: E402
from lora_qwen.model import load_model_and_tokenizer, print_trainable_params  # noqa: E402
from lora_qwen.training import TrainConfig, train  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke fine-tune (pipeline validation)")
    p.add_argument("--lora", type=Path, default=REPO_ROOT / "configs" / "all_linear.yaml")
    p.add_argument("--data", type=Path, default=REPO_ROOT / "configs" / "data" / "metamath.yaml")
    p.add_argument("--train", type=Path, default=REPO_ROOT / "configs" / "train" / "smoke.yaml")
    p.add_argument(
        "--save-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "smoke_ft",
        help="Where to persist the trained adapter (None or '' disables saving)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    lora_cfg = LoraSetupConfig.from_yaml(args.lora)
    data_cfg = DataConfig.from_yaml(args.data)
    train_cfg = TrainConfig.from_yaml(args.train)

    print("Configs:")
    print(f"  {lora_cfg.describe()}")
    print(f"  {data_cfg.describe()}")
    print(f"  {train_cfg.describe()}")

    print("\n[1/4] Loading model + tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer(lora_cfg)

    print(f"[2/4] Building dataset '{data_cfg.dataset}' "
          f"(max_examples={data_cfg.max_examples}, max_length={data_cfg.max_length})...")
    dataset = build_supervised_dataset(
        data_cfg.dataset,
        tokenizer,
        split=data_cfg.split,
        max_examples=data_cfg.max_examples,
        max_length=data_cfg.max_length,
    )
    collator = SupervisedCollator(tokenizer)
    print(f"      -> {len(dataset)} examples")

    print(f"[3/4] Applying LoRA (backend={lora_cfg.backend}, targets={lora_cfg.target_modules})...")
    model = apply_lora(model, lora_cfg)
    print_trainable_params(model, prefix="      ")

    print(f"[4/4] Training for max_steps={train_cfg.max_steps} on {device}...")
    result = train(model, dataset, collator, train_cfg, device=device)

    if args.save_dir and str(args.save_dir):
        save_adapter(model, args.save_dir, lora_cfg)
        saved_msg = f"  adapter saved: {args.save_dir}"
    else:
        saved_msg = "  adapter saved: (skipped — no --save-dir)"

    print("\nResult:")
    print(f"  steps done   : {result.steps_done}")
    print(f"  wall time    : {result.wall_time_sec:.1f}s")
    first, last = result.loss_trend()
    if first is not None and last is not None:
        delta = last - first
        direction = "DECREASED" if delta < 0 else "INCREASED"
        print(f"  loss trend   : {first:.4f} -> {last:.4f} ({delta:+.4f}) {direction}")
    print(saved_msg)

    if first is not None and last is not None and last >= first:
        print("\nWARNING: loss did not decrease. On such a short run this can happen "
              "(especially with MPS bf16 noise), but worth investigating if persistent.")
        return 1

    print("\nSmoke fine-tune completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
