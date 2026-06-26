"""Train one LoRA adapter, with CLI hyperparameter overrides.

This is the real training entry point for the experiment pipeline (the older
``smoke_ft.py`` is only a pipeline-validation toy). It:

  1. loads the base model + a LoRA config (target modules / rank / alpha),
  2. builds a supervised dataset from a data config (default: gsm8k_train),
  3. applies LoRA and fine-tunes with the train config,
  4. saves the adapter and a metadata-rich run JSON.

Hyperparameters that we sweep (LR, #epochs, schedule, weight decay, seed) are
exposed as CLI overrides so a sweep is a loop over flags, not a pile of YAMLs.

Per-epoch diagnostic: ``--per-epoch-eval`` checkpoints the adapter and scores a
dev slice after every epoch in a *single* training run, giving the full
epoch-vs-accuracy curve for the price of one run (a checkpoint after epoch k is
exactly what a k-epoch run would have produced under a constant LR).

Examples:
    python scripts/train_lora.py \
        --lora-config configs/E2/E2_qv_r8.yaml \
        --data-config configs/data/gsm8k_train.yaml \
        --train-config configs/train/full_gsm8k.yaml \
        --save-dir checkpoints/v2/qv_r8 \
        --output results_v2/qv_r8.train.json

    # epochs diagnostic (constant LR, dev-eval after each epoch)
    python scripts/train_lora.py \
        --lora-config configs/E2/E2_qv_r8.yaml \
        --data-config configs/data/gsm8k_train.yaml \
        --train-config configs/train/diag_epochs.yaml \
        --save-dir checkpoints/v2/diag_qv_r8 \
        --per-epoch-eval --dev-num-problems 1000 \
        --output results_v2/diag_epochs_qv_r8.json
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
from lora_qwen.lora import apply_lora, save_adapter  # noqa: E402
from lora_qwen.model import (  # noqa: E402
    load_model_and_tokenizer,
    lora_budget_report,
    print_trainable_params,
)
from lora_qwen.training import TrainConfig, train  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one LoRA adapter (with HP overrides)")
    p.add_argument("--lora-config", type=Path, required=True)
    p.add_argument("--data-config", type=Path, default=REPO_ROOT / "configs" / "data" / "gsm8k_train.yaml")
    p.add_argument("--train-config", type=Path, default=REPO_ROOT / "configs" / "train" / "full.yaml")
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Run-metadata JSON path")
    p.add_argument("--train-prompt", choices=["zeroshot", "fewshot", "instruct"],
                   default="zeroshot",
                   help="Prompt style the responses are conditioned on during SFT. "
                        "'instruct' = step-by-step + #### format, no exemplars.")
    p.add_argument("--max-length", type=int, default=None,
                   help="Override tokenizer max_length (fewshot/instruct prompts "
                        "are longer; default 1024 for those styles).")
    # LoRA-config overrides (avoid a YAML per sweep point, e.g. the rank sweep).
    p.add_argument("--rank", type=int, default=None,
                   help="Override LoRA rank from the config (alpha follows unless set).")
    p.add_argument("--alpha", type=int, default=None,
                   help="Override LoRA alpha (default: 2*rank when --rank is given).")

    # Hyperparameter overrides (None => use the train-config value).
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--lr-schedule", choices=["cosine", "linear", "constant"], default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)

    # Dev diagnostics during training.
    p.add_argument("--per-epoch-eval", action="store_true",
                   help="Checkpoint + dev-eval after each epoch (epochs diagnostic).")
    p.add_argument("--eval-every-steps", type=int, default=None,
                   help="Checkpoint + dev-eval every N optimizer steps (intra-epoch curve).")
    p.add_argument("--eval-step-zero", action="store_true",
                   help="Dev-eval at step 0 (LoRA attached, before any optimizer step).")
    p.add_argument("--dev-num-problems", type=int, default=1000,
                   help="How many dev problems to score in diagnostic evals.")
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512)
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
    lora_cfg = LoraSetupConfig.from_yaml(args.lora_config)
    if args.rank is not None:
        lora_cfg.rank = args.rank
        lora_cfg.alpha = args.alpha if args.alpha is not None else 2 * args.rank
    elif args.alpha is not None:
        lora_cfg.alpha = args.alpha
    data_cfg = DataConfig.from_yaml(args.data_config)
    train_cfg = _apply_overrides(TrainConfig.from_yaml(args.train_config), args)

    print("Configs:")
    print(f"  {lora_cfg.describe()}")
    print(f"  {data_cfg.describe()}")
    print(f"  {train_cfg.describe()}")

    print("\n[1/4] Loading model + tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer(lora_cfg)
    budget = lora_budget_report(model, lora_cfg)
    print(budget.render(prefix="      "))

    # Few-shot prompts are ~3-4x longer; bump the default cap so the exemplars
    # are never truncated away (which would corrupt the prompt/response mask).
    max_length = args.max_length
    if max_length is None:
        max_length = 1024 if args.train_prompt in {"fewshot", "instruct"} else data_cfg.max_length

    loader_kwargs = {"prompt_style": args.train_prompt} if data_cfg.dataset == "gsm8k_train" else None
    eval_prompt_fn = resolve_prompt_fn(args.train_prompt)

    print(f"\n[2/4] Building dataset '{data_cfg.dataset}' (split={data_cfg.split}, "
          f"max_examples={data_cfg.max_examples}, prompt={args.train_prompt}, "
          f"max_length={max_length})...")
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

    print(f"[3/4] Applying LoRA (backend={lora_cfg.backend}, targets={lora_cfg.target_modules})...")
    model = apply_lora(model, lora_cfg)
    actual_trainable, _ = print_trainable_params(model, prefix="      ")
    if actual_trainable != budget.expected_trainable:
        raise SystemExit(
            f"Trainable-param mismatch: report expects {budget.expected_trainable:,} "
            f"but apply() unfroze {actual_trainable:,}."
        )

    dev_curve: list[dict] = []
    step_curve: list[dict] = []
    on_epoch_end = None
    on_macro_step = None
    eval_every_steps = args.eval_every_steps

    if args.per_epoch_eval and args.eval_every_steps is not None:
        raise SystemExit("Use either --per-epoch-eval or --eval-every-steps, not both.")

    dev_problems = None
    if args.per_epoch_eval or args.eval_every_steps is not None or args.eval_step_zero:
        dev_problems = list(load_gsm8k(split="dev", max_examples=args.dev_num_problems))

    def _record_step(step: int, m) -> None:  # noqa: ANN001
        ckpt = args.save_dir.parent / f"{args.save_dir.name}_step{step}"
        save_adapter(m, ckpt, lora_cfg)
        res = evaluate_gsm8k(
            m, tokenizer, dev_problems,
            name=f"dev_step{step}",
            max_new_tokens=args.max_new_tokens,
            device=device,
            batch_size=args.eval_batch_size,
            prompt_fn=eval_prompt_fn,
        )
        step_curve.append({
            "step": step,
            "checkpoint": str(ckpt),
            "dev_accuracy": res.accuracy,
            "dev_correct": res.correct,
            "dev_total": res.total,
        })
        print(f"      [step {step}] dev acc={res.accuracy:.4f} "
              f"({res.correct}/{res.total})")

    if args.eval_step_zero:
        print(f"      step-0 eval enabled: {len(dev_problems)} dev problems")
        model.eval()
        _record_step(0, model)
        model.train()

    if args.per_epoch_eval:
        print(f"      per-epoch eval enabled: {len(dev_problems)} dev problems")

        def on_epoch_end(epoch: int, m) -> None:  # noqa: ANN001
            ckpt = args.save_dir.parent / f"{args.save_dir.name}_ep{epoch}"
            save_adapter(m, ckpt, lora_cfg)
            res = evaluate_gsm8k(
                m, tokenizer, dev_problems,
                name=f"dev_ep{epoch}",
                max_new_tokens=args.max_new_tokens,
                device=device,
                batch_size=args.eval_batch_size,
                prompt_fn=eval_prompt_fn,
            )
            dev_curve.append({
                "epoch": epoch,
                "checkpoint": str(ckpt),
                "dev_accuracy": res.accuracy,
                "dev_correct": res.correct,
                "dev_total": res.total,
            })
            print(f"      [epoch {epoch}] dev acc={res.accuracy:.4f} "
                  f"({res.correct}/{res.total})")

    if args.eval_every_steps is not None:
        print(f"      per-step eval every {args.eval_every_steps} macro steps: "
              f"{len(dev_problems)} dev problems")
        on_macro_step = _record_step

    print(f"[4/4] Training (epochs={train_cfg.num_epochs}, lr={train_cfg.learning_rate}, "
          f"schedule={train_cfg.lr_schedule}) on {device}...")
    result = train(
        model, dataset, collator, train_cfg, device=device,
        on_epoch_end=on_epoch_end,
        on_macro_step=on_macro_step,
        eval_every_steps=eval_every_steps,
    )

    save_adapter(model, args.save_dir, lora_cfg)
    print(f"\n  adapter saved: {args.save_dir}")

    first, last = result.loss_trend()
    if first is not None and last is not None:
        print(f"  loss trend: {first:.4f} -> {last:.4f} ({last - first:+.4f})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lora_config": str(args.lora_config),
        "data_config": str(args.data_config),
        "train_config": str(args.train_config),
        "save_dir": str(args.save_dir),
        "model_name": lora_cfg.model_name,
        "backend": lora_cfg.backend,
        "device": str(device),
        "transformers_version": _transformers_version(),
        "dataset": data_cfg.dataset,
        "data_split": data_cfg.split,
        "max_examples": data_cfg.max_examples,
        "train_prompt": args.train_prompt,
        "max_length": max_length,
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
        "lora": {**budget.to_dict(), "actual_trainable_params": actual_trainable},
        "steps_done": result.steps_done,
        "wall_time_sec": result.wall_time_sec,
        "final_loss": result.final_loss,
        "loss_curve": result.losses,
        "dev_curve": dev_curve,
        "step_curve": step_curve,
        "eval_every_steps": eval_every_steps,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"  run metadata written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
