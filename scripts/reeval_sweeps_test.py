"""Re-score the E1 (target) and E2 (rank) sweep adapters on GSM8K-**test**.

The sweeps were originally selected on the held-out dev slice, but that slice is
part of GSM8K-train and is contaminated by pretraining (see headroom caveat), so
its absolute numbers are inflated. The trained adapters live on the shared volume,
so we just re-score them on the clean test split — no re-training.

For each model: E1 over {q,v,qv,attention,all_linear} at iso-budget rank (from the
E15b YAML), and E2 over rank {1,2,4,8,16,32,64} on the model's best target (rank
overridden via a derived YAML so eval re-applies the matching structure).

Writes results_headroom/<model>/test_sweeps/{e1_<t>,e2_<t>_r<r>}.json on the
volume. Idempotent.

Usage (on a pod):
    python scripts/reeval_sweeps_test.py --model qwen2_1.5b --best-target attention
    python scripts/reeval_sweeps_test.py --model qwen25_1.5b --best-target qv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEST_N = 1319
MNT = 512
TARGETS = ["q", "v", "qv", "attention", "all_linear"]
RANKS = [1, 2, 4, 8, 16, 32, 64]

E1DIR = {"qwen2_1.5b": "configs/E15b_qwen2", "qwen25_1.5b": "configs/E15b_qwen25"}
CK = "/workspace/lora-qwen/checkpoints/headroom"


def run(cmd: list[str], logfile: Path) -> None:
    print("[re] $ " + " ".join(str(c) for c in cmd), flush=True)
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


def score(lora_cfg: str, adapter: str, out: Path, log: Path) -> float:
    if not done(out):
        run([
            sys.executable, "-u", "scripts/eval_gsm8k.py",
            "--lora-config", lora_cfg, "--adapter-dir", adapter,
            "--split", "test", "--num-problems", str(TEST_N),
            "--base-prompt", "instruct", "--ft-prompt", "instruct", "--skip-base",
            "--batch-size", "64", "--max-new-tokens", str(MNT),
            "--no-save-completions", "--output", str(out),
        ], log)
    return json.loads(out.read_text())["ft"]["accuracy"]


def derived_cfg(base_yaml: Path, rank: int, dst: Path) -> Path:
    import yaml  # noqa: PLC0415
    src = yaml.safe_load(base_yaml.read_text())
    src["rank"] = rank
    src["alpha"] = 2 * rank
    dst.write_text(yaml.safe_dump(src, sort_keys=False))
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(E1DIR))
    ap.add_argument("--best-target", required=True, choices=TARGETS)
    ap.add_argument("--results-root", default="/workspace/lora-qwen/results_headroom")
    args = ap.parse_args()
    e1dir = E1DIR[args.model]
    res = Path(args.results_root) / args.model / "test_sweeps"
    res.mkdir(parents=True, exist_ok=True)
    logs = res / "logs"
    ck = f"{CK}/{args.model}"

    e1: dict[str, float] = {}
    for t in TARGETS:
        a = score(f"{e1dir}/{t}.yaml", f"{ck}/e1_{t}",
                  res / f"e1_{t}.json", logs / f"e1_{t}.log")
        e1[t] = a
        print(f"[re] {args.model} E1 {t}: test={a:.4f}", flush=True)

    e2: dict[str, float] = {}
    bt = args.best_target
    for r in RANKS:
        dcfg = derived_cfg(Path(f"{e1dir}/{bt}.yaml"), r, res / f"_cfg_{bt}_r{r}.yaml")
        a = score(str(dcfg), f"{ck}/e2_{bt}_r{r}",
                  res / f"e2_{bt}_r{r}.json", logs / f"e2_{bt}_r{r}.log")
        e2[str(r)] = a
        print(f"[re] {args.model} E2 {bt} r{r}: test={a:.4f}", flush=True)

    summary = {"model": args.model, "best_target": bt,
               "e1_test": e1, "e2_test": e2}
    (res / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[re] DONE", json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
