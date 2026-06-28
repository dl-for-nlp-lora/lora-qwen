"""Add one extra reference point to the headroom plot (e.g. a larger backbone at
the top of the headroom range).

Trains a single LoRA adapter on GSM8K-train (instruct, 1 epoch) with a FIXED
config carried over from the controlled 1.5B pair (so the recipe is identical),
then scores base {0-shot, few-shot, instruct} and the LoRA adapter @512 on
GSM8K-test. NOT a perfectly controlled point (the backbone differs in size/arch
from the 1.5B pair) — it is reported as a reference, not as a fourth cell of the
controlled experiment.

Usage (on a pod):
    python scripts/run_reference_point.py --tag qwen3_1.7b \
        --base-cfg configs/qwen3_1.7b/base.yaml \
        --target-cfg configs/qwen3_1.7b/qv.yaml \
        --rank 16 --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MNT = 512
TEST_N = 1319
TRAIN_PROMPT = "instruct"


def run(cmd: list[str], logfile: Path) -> None:
    print("[ref] $ " + " ".join(str(c) for c in cmd), flush=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO)
    if p.returncode != 0:
        tail = logfile.read_text().splitlines()[-25:]
        raise SystemExit(f"failed ({p.returncode}):\n" + "\n".join(tail))


def done(path: Path) -> bool:
    try:
        json.loads(Path(path).read_text())
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base-cfg", required=True)
    ap.add_argument("--target-cfg", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--results-root", default="/workspace/lora-qwen/results_headroom")
    ap.add_argument("--ckpt-root", default="/workspace/lora-qwen/checkpoints/headroom")
    args = ap.parse_args()

    res = Path(args.results_root) / f"ref_{args.tag}"
    ck = Path(args.ckpt_root) / f"ref_{args.tag}"
    logs = res / "logs"
    res.mkdir(parents=True, exist_ok=True)

    adapter = ck / f"qv_r{args.rank}"
    train_out = res / "train.json"
    if not done(train_out):
        run([
            sys.executable, "-u", "scripts/train_lora.py",
            "--lora-config", args.target_cfg,
            "--data-config", "configs/data/gsm8k_train.yaml",
            "--train-config", "configs/train/full.yaml",
            "--train-prompt", TRAIN_PROMPT, "--num-epochs", "1",
            "--learning-rate", str(args.lr), "--rank", str(args.rank),
            "--max-new-tokens", str(MNT),
            "--save-dir", str(adapter), "--output", str(train_out),
        ], logs / "train.log")

    base_out = res / "final_base_test.json"
    if not done(base_out):
        run([
            sys.executable, "-u", "scripts/eval_base_truncation.py",
            "--lora-config", args.base_cfg,
            "--prompts", "zeroshot", "fewshot", "instruct",
            "--split", "test", "--num-problems", str(TEST_N),
            "--max-new-tokens", str(MNT), "--batch-size", "64",
            "--out", str(base_out),
        ], logs / "final_base_test.log")

    import yaml  # noqa: PLC0415
    src = yaml.safe_load(Path(args.target_cfg).read_text())
    src["rank"] = args.rank
    src["alpha"] = 2 * args.rank
    bestcfg = res / "best_lora.yaml"
    bestcfg.write_text(yaml.safe_dump(src, sort_keys=False))
    lora_out = res / "final_lora_test.json"
    if not done(lora_out):
        run([
            sys.executable, "-u", "scripts/eval_gsm8k.py",
            "--lora-config", str(bestcfg), "--adapter-dir", str(adapter),
            "--split", "test", "--num-problems", str(TEST_N),
            "--base-prompt", "instruct", "--ft-prompt", "instruct", "--skip-base",
            "--batch-size", "64", "--max-new-tokens", str(MNT),
            "--no-save-completions", "--output", str(lora_out),
        ], logs / "final_lora_test.log")

    base_runs = json.loads(base_out.read_text())["runs"]
    summary = {
        "tag": args.tag, "rank": args.rank, "lr": args.lr,
        "base_test": {k: v["accuracy"] for k, v in base_runs.items()},
        "lora_test": json.loads(lora_out.read_text())["ft"]["accuracy"],
    }
    (res / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[ref] summary:", json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
