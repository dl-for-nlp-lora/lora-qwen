"""Staged LoRA funnel for ONE model of the headroom study, end to end.

Runs entirely on a single pod, writing every artifact to the shared network
volume (so it survives pod death and is visible to the local writeup). The
funnel passes the best config from each stage to the next:

    diag (confirm 1 epoch) -> LR -> E1 (which matrix) -> E2 (which rank)
    -> final test@512 matrix (base 0/few/instruct + best LoRA + Full FT)

Every train/eval run writes a JSON; the orchestrator is idempotent — a run
whose output JSON already exists (and parses) is skipped, so re-launching after
an interruption resumes where it stopped. Stage decisions are recorded in
``<results>/<model>/decisions.json``.

Usage (on the pod):
    python scripts/run_headroom_pipeline.py --model-key qwen2_1.5b
    python scripts/run_headroom_pipeline.py --model-key qwen25_1.5b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MODELS = {
    "qwen2_1.5b":  {"base": "configs/qwen2_1.5b/base.yaml",  "e1dir": "configs/E15b_qwen2"},
    "qwen25_1.5b": {"base": "configs/qwen25_1.5b/base.yaml", "e1dir": "configs/E15b_qwen25"},
}
TARGETS = ["q", "v", "qv", "attention", "all_linear"]
LRS = [2e-4, 4e-4, 8e-4]
RANKS = [1, 2, 4, 8, 16, 32, 64]
TRAIN_PROMPT = "instruct"          # self-terminating #### format
DEV_N = 500                        # held-out dev for sweeps
TEST_N = 1319                      # full GSM8K-test for the final matrix
MNT = 512


def log(msg: str) -> None:
    print(f"[pipe] {msg}", flush=True)


def run(cmd: list[str], logfile: Path) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO)
    if p.returncode != 0:
        tail = logfile.read_text().splitlines()[-25:]
        raise SystemExit(f"command failed ({p.returncode}); tail:\n" + "\n".join(tail))


def dev_acc(train_json: Path) -> float:
    d = json.loads(train_json.read_text())
    curve = d.get("dev_curve") or []
    if not curve:
        raise SystemExit(f"{train_json} has no dev_curve")
    return float(curve[-1]["dev_accuracy"])


def done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text())
        return True
    except Exception:
        return False


def train_eval(*, name: str, lora_cfg: str, save_dir: Path, out: Path,
               lr: float, rank: int | None, epochs: int, train_cfg: str,
               dev_n: int, logdir: Path) -> float:
    """One sweep point: train (1+ epochs) + dev-eval via --per-epoch-eval."""
    if done(out):
        log(f"skip {name} (cached) -> dev={dev_acc(out):.4f}")
        return dev_acc(out)
    cmd = [
        sys.executable, "-u", "scripts/train_lora.py",
        "--lora-config", lora_cfg,
        "--data-config", "configs/data/gsm8k_train.yaml",
        "--train-config", train_cfg,
        "--train-prompt", TRAIN_PROMPT,
        "--num-epochs", str(epochs),
        "--learning-rate", str(lr),
        "--per-epoch-eval", "--dev-num-problems", str(dev_n),
        "--eval-batch-size", "64", "--max-new-tokens", str(MNT),
        "--save-dir", str(save_dir), "--output", str(out),
    ]
    if rank is not None:
        cmd += ["--rank", str(rank)]
    run(cmd, logdir / f"{name}.log")
    a = dev_acc(out)
    log(f"{name}: dev={a:.4f}")
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=list(MODELS))
    ap.add_argument("--results-root", default="/workspace/lora-qwen/results_headroom")
    ap.add_argument("--ckpt-root", default="/workspace/lora-qwen/checkpoints/headroom")
    ap.add_argument("--skip-full-ft", action="store_true")
    args = ap.parse_args()

    mk = args.model_key
    m = MODELS[mk]
    base_cfg = m["base"]
    e1dir = m["e1dir"]
    res = Path(args.results_root) / mk
    ck = Path(args.ckpt_root) / mk
    logs = res / "logs"
    res.mkdir(parents=True, exist_ok=True)
    decisions_path = res / "decisions.json"
    decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}

    def save_decisions() -> None:
        decisions_path.write_text(json.dumps(decisions, indent=2))

    t0 = time.time()
    log(f"=== model {mk} ({base_cfg}) ===")

    # ---- Stage 1: epoch diagnostic (constant LR, q_proj heuristic) ----------
    diag_out = res / "diag_q.json"
    if not done(diag_out):
        run([
            sys.executable, "-u", "scripts/train_lora.py",
            "--lora-config", f"{e1dir}/q.yaml",
            "--data-config", "configs/data/gsm8k_train.yaml",
            "--train-config", "configs/train/diag_epochs.yaml",
            "--train-prompt", TRAIN_PROMPT, "--num-epochs", "3",
            "--per-epoch-eval", "--dev-num-problems", str(DEV_N),
            "--eval-batch-size", "64", "--max-new-tokens", str(MNT),
            "--save-dir", str(ck / "diag_q"), "--output", str(diag_out),
        ], logs / "diag_q.log")
    diag_curve = json.loads(diag_out.read_text())["dev_curve"]
    decisions["diag_epoch_curve"] = {c["epoch"]: c["dev_accuracy"] for c in diag_curve}
    best_epoch = max(diag_curve, key=lambda c: c["dev_accuracy"])["epoch"]
    decisions["diag_best_epoch"] = best_epoch
    # Plan: 1 epoch is the operating point regardless (cost), note if diag disagrees.
    decisions["epochs_used"] = 1
    save_decisions()
    log(f"diag epoch curve: {decisions['diag_epoch_curve']} (best={best_epoch}); using 1 epoch")

    # ---- Stage 2: LR check (q_proj, 1 epoch cosine) -------------------------
    lr_scores = {}
    for lr in LRS:
        tag = f"lr_{lr:.0e}".replace("-", "m")
        a = train_eval(name=tag, lora_cfg=f"{e1dir}/q.yaml",
                       save_dir=ck / tag, out=res / f"{tag}.json",
                       lr=lr, rank=None, epochs=1,
                       train_cfg="configs/train/full.yaml", dev_n=DEV_N, logdir=logs)
        lr_scores[f"{lr:.0e}"] = a
    best_lr = max(LRS, key=lambda lr: lr_scores[f"{lr:.0e}"])
    decisions["lr_scores"] = lr_scores
    decisions["best_lr"] = f"{best_lr:.0e}"
    save_decisions()
    log(f"LR scores: {lr_scores} -> best {best_lr:.0e}")

    # ---- Stage 3: E1 target sweep (iso-budget, best LR) ---------------------
    e1_scores = {}
    for t in TARGETS:
        tag = f"e1_{t}"
        a = train_eval(name=tag, lora_cfg=f"{e1dir}/{t}.yaml",
                       save_dir=ck / tag, out=res / f"{tag}.json",
                       lr=best_lr, rank=None, epochs=1,
                       train_cfg="configs/train/full.yaml", dev_n=DEV_N, logdir=logs)
        e1_scores[t] = a
    best_target = max(TARGETS, key=lambda t: e1_scores[t])
    decisions["e1_scores"] = e1_scores
    decisions["best_target"] = best_target
    save_decisions()
    log(f"E1 scores: {e1_scores} -> best target {best_target}")

    # ---- Stage 4: E2 rank sweep (best target, best LR) ----------------------
    e2_scores = {}
    for r in RANKS:
        tag = f"e2_{best_target}_r{r}"
        a = train_eval(name=tag, lora_cfg=f"{e1dir}/{best_target}.yaml",
                       save_dir=ck / tag, out=res / f"{tag}.json",
                       lr=best_lr, rank=r, epochs=1,
                       train_cfg="configs/train/full.yaml", dev_n=DEV_N, logdir=logs)
        e2_scores[str(r)] = a
    best_rank = max(RANKS, key=lambda r: e2_scores[str(r)])
    decisions["e2_scores"] = e2_scores
    decisions["best_rank"] = best_rank
    save_decisions()
    log(f"E2 scores: {e2_scores} -> best rank {best_rank}")

    best_adapter = ck / f"e2_{best_target}_r{best_rank}"
    decisions["best_adapter"] = str(best_adapter)
    save_decisions()

    # Derived config matching the best adapter's rank (eval re-applies structure
    # from a YAML, so its rank/alpha must match what was trained via --rank).
    import yaml  # noqa: PLC0415
    best_cfg_src = yaml.safe_load(Path(f"{e1dir}/{best_target}.yaml").read_text())
    best_cfg_src["rank"] = best_rank
    best_cfg_src["alpha"] = 2 * best_rank
    best_cfg_src["name"] = f"{mk}_best_{best_target}_r{best_rank}"
    best_cfg_path = res / "best_lora.yaml"
    best_cfg_path.write_text(yaml.safe_dump(best_cfg_src, sort_keys=False))

    # ---- Stage 5: final test@512 matrix -------------------------------------
    # base 0/few/instruct on test
    base_out = res / "final_base_test.json"
    if not done(base_out):
        run([
            sys.executable, "-u", "scripts/eval_base_truncation.py",
            "--lora-config", base_cfg,
            "--prompts", "zeroshot", "fewshot", "instruct",
            "--split", "test", "--num-problems", str(TEST_N),
            "--max-new-tokens", str(MNT), "--batch-size", "64",
            "--out", str(base_out),
        ], logs / "final_base_test.log")
    base_runs = json.loads(base_out.read_text())["runs"]
    decisions["final_base_test"] = {k: v["accuracy"] for k, v in base_runs.items()}

    # best LoRA on test
    lora_test = res / "final_lora_test.json"
    if not done(lora_test):
        run([
            sys.executable, "-u", "scripts/eval_gsm8k.py",
            "--lora-config", str(best_cfg_path),
            "--adapter-dir", str(best_adapter),
            "--split", "test", "--num-problems", str(TEST_N),
            "--base-prompt", "instruct", "--ft-prompt", "instruct", "--skip-base",
            "--batch-size", "64", "--max-new-tokens", str(MNT),
            "--no-save-completions", "--output", str(lora_test),
        ], logs / "final_lora_test.log")
    decisions["final_lora_test"] = json.loads(lora_test.read_text())["ft"]["accuracy"]

    # Full FT on test (optional)
    if not args.skip_full_ft:
        ft_dir = ck / "full_ft"
        ft_train = res / "full_ft_train.json"
        if not done(ft_train):
            run([
                sys.executable, "-u", "scripts/train_full_ft.py",
                "--model-config", base_cfg,
                "--data-config", "configs/data/gsm8k_train.yaml",
                "--train-config", "configs/train/full.yaml",
                "--train-prompt", TRAIN_PROMPT, "--num-epochs", "1",
                "--optimizer", "sgd", "--max-new-tokens", str(MNT),
                "--save-dir", str(ft_dir), "--output", str(ft_train),
            ], logs / "full_ft_train.log")
        ft_test = res / "final_full_ft_test.json"
        if not done(ft_test):
            run([
                sys.executable, "-u", "scripts/eval_gsm8k.py",
                "--lora-config", base_cfg, "--full-model-dir", str(ft_dir),
                "--split", "test", "--num-problems", str(TEST_N),
                "--base-prompt", "instruct", "--ft-prompt", "instruct", "--skip-base",
                "--batch-size", "64", "--max-new-tokens", str(MNT),
                "--no-save-completions", "--output", str(ft_test),
            ], logs / "final_full_ft_test.log")
        decisions["final_full_ft_test"] = json.loads(ft_test.read_text())["ft"]["accuracy"]

    decisions["wall_time_sec"] = round(time.time() - t0, 1)
    save_decisions()
    log(f"=== {mk} DONE in {decisions['wall_time_sec']/60:.1f} min ===")
    log(json.dumps(decisions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
