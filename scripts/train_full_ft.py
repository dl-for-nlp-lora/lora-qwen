"""Full fine-tuning (all params trainable) — E0b baseline.

Same data recipe and train hyperparameters as LoRA runs, but no adapters.
Saves a HuggingFace checkpoint directory for eval via ``eval_gsm8k.py
--full-model-dir``.
"""

from __future__ import annotations

import argparse
import json
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
from lora_qwen.evaluation import evaluate_gsm8k, load_gsm8k, resolve_prompt_fn  # noqa: E402
from lora_qwen.model import load_model_and_tokenizer, print_trainable_params  # noqa: E402
from lora_qwen.training import TrainConfig, train  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full fine-tune (E0b, all params)")
    p.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs" / "E0a_base.yaml")
    p.add_argument("--data-config", type=Path, default=REPO_ROOT / "configs" / "data" / "gsm8k_train.yaml")
    p.add_argument("--train-config", type=Path, default=REPO_ROOT / "configs" / "train" / "full_gsm8k.yaml")
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-prompt", choices=["zeroshot", "fewshot", "instruct"], default="instruct")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--lr-schedule", choices=["cosine", "linear", "constant"], default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--per-epoch-eval", action="store_true")
    p.add_argument("--dev-num-problems", type=int, default=1000)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="sgd",
                   help="Optimizer for full FT. SGD fits 24GB at seq 1024; AdamW needs shorter seq.")
    return p.parse_args()


def _apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.lr_schedule is not None:
        cfg.lr_schedule = args.lr_schedule
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def _transformers_version() -> str:
    import transformers

    return transformers.__version__


def main() -> int:
    args = parse_args()
    model_cfg = LoraSetupConfig.from_yaml(args.model_config)
    data_cfg = DataConfig.from_yaml(args.data_config)
    train_cfg = _apply_overrides(TrainConfig.from_yaml(args.train_config), args)

    print("Configs:")
    print(f"  model={model_cfg.model_name} dtype={model_cfg.dtype}")
    print(f"  {data_cfg.describe()}")
    print(f"  {train_cfg.describe()}")

    print("\n[1/4] Loading model + tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer(model_cfg)

    max_length = args.max_length
    if max_length is None:
        max_length = 1024 if args.train_prompt in {"fewshot", "instruct"} else data_cfg.max_length

    loader_kwargs = {"prompt_style": args.train_prompt} if data_cfg.dataset == "gsm8k_train" else None
    eval_prompt_fn = resolve_prompt_fn(args.train_prompt)

    print(f"\n[2/4] Building dataset (prompt={args.train_prompt}, max_length={max_length})...")
    dataset = build_supervised_dataset(
        data_cfg.dataset,
        tokenizer,
        split=data_cfg.split,
        max_examples=data_cfg.max_examples,
        max_length=max_length,
        loader_kwargs=loader_kwargs,
    )
    collator = SupervisedCollator(tokenizer)
    print(f"      -> {len(dataset)} training examples")

    print("[3/4] Unfreezing all parameters for full fine-tuning...")
    for p in model.parameters():
        p.requires_grad = True
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("      gradient checkpointing enabled")
    model.train()
    trainable, total = print_trainable_params(model, prefix="      ")
    if trainable != total:
        raise SystemExit(f"Expected all {total:,} params trainable, got {trainable:,}")

    dev_curve: list[dict] = []
    on_epoch_end = None
    if args.per_epoch_eval:
        dev_problems = list(load_gsm8k(split="dev", max_examples=args.dev_num_problems))
        print(f"      per-epoch eval enabled: {len(dev_problems)} dev problems")

        def on_epoch_end(epoch: int, m) -> None:  # noqa: ANN001
            ckpt = args.save_dir.parent / f"{args.save_dir.name}_ep{epoch}"
            ckpt.mkdir(parents=True, exist_ok=True)
            m.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            m.eval()
            res = evaluate_gsm8k(
                m, tokenizer, dev_problems,
                name=f"dev_ep{epoch}",
                max_new_tokens=args.max_new_tokens,
                device=device,
                batch_size=args.eval_batch_size,
                prompt_fn=eval_prompt_fn,
            )
            m.train()
            dev_curve.append({
                "epoch": epoch,
                "checkpoint": str(ckpt),
                "dev_accuracy": res.accuracy,
                "dev_correct": res.correct,
                "dev_total": res.total,
            })
            print(f"      [epoch {epoch}] dev acc={res.accuracy:.4f} "
                  f"({res.correct}/{res.total})")

    print(f"[4/4] Training on {device}...")
    result = train(
        model, dataset, collator, train_cfg, device=device,
        on_epoch_end=on_epoch_end, optimizer=args.optimizer,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"\n  checkpoint saved: {args.save_dir}")

    first, last = result.loss_trend()
    if first is not None and last is not None:
        print(f"  loss trend: {first:.4f} -> {last:.4f} ({last - first:+.4f})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": str(args.model_config),
        "data_config": str(args.data_config),
        "train_config": str(args.train_config),
        "save_dir": str(args.save_dir),
        "model_name": model_cfg.model_name,
        "mode": "full_ft",
        "device": str(device),
        "transformers_version": _transformers_version(),
        "dataset": data_cfg.dataset,
        "data_split": data_cfg.split,
        "max_examples": data_cfg.max_examples,
        "train_prompt": args.train_prompt,
        "max_length": max_length,
        "optimizer": args.optimizer,
        "train": {
            "num_epochs": train_cfg.num_epochs,
            "learning_rate": train_cfg.learning_rate,
            "lr_schedule": train_cfg.lr_schedule,
            "weight_decay": train_cfg.weight_decay,
            "warmup_ratio": train_cfg.warmup_ratio,
            "per_device_batch_size": train_cfg.per_device_batch_size,
            "grad_accum_steps": train_cfg.grad_accum_steps,
            "seed": train_cfg.seed,
        },
        "trainable_params": trainable,
        "total_params": total,
        "steps_done": result.steps_done,
        "wall_time_sec": result.wall_time_sec,
        "final_loss": result.final_loss,
        "loss_curve": result.losses,
        "dev_curve": dev_curve,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"  run metadata written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
