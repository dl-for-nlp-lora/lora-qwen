"""Produce the data for the 'where do FT gains come from' analysis.

For each model in the headroom study, on GSM8K-test (instruct @512):
  1. base + best-LoRA completions WITH per-example text saved (for the by-hand
     qualitative categorization of corrections vs regressions), and
  2. a batch=1 latency benchmark (tokens + wall time per sample, base vs FT).

Everything is written to the shared volume under results_headroom/<model>/.
Idempotent: existing valid JSONs are skipped.

Usage (on the pod):
    python scripts/run_ft_analysis.py --model qwen2_1.5b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEST_N = 1319
LAT_N = 150
MNT = 512

MODELS = {
    "qwen2_1.5b":  {
        "base": "configs/qwen2_1.5b/base.yaml",
        "ft": "results_headroom/qwen2_1.5b/best_lora.yaml",
        "adapter": "checkpoints/headroom/qwen2_1.5b/e2_attention_r1",
    },
    "qwen25_1.5b": {
        "base": "configs/qwen25_1.5b/base.yaml",
        "ft": "results_headroom/qwen25_1.5b/best_lora.yaml",
        "adapter": "checkpoints/headroom/qwen25_1.5b/e2_qv_r16",
    },
    "qwen3_1.7b": {
        "base": "configs/qwen3_1.7b/base.yaml",
        "ft": "results_headroom/ref_qwen3_1.7b/best_lora.yaml",
        "adapter": "checkpoints/headroom/ref_qwen3_1.7b/qv_r16",
    },
}


def run(cmd: list[str], logfile: Path) -> None:
    print("[fta] $ " + " ".join(str(c) for c in cmd), flush=True)
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
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--results-root", default="/workspace/lora-qwen/results_headroom")
    args = ap.parse_args()
    m = MODELS[args.model]
    res = Path(args.results_root) / args.model
    logs = res / "logs"

    # 1) base + FT completions (saved) on full test
    compl = res / "base_vs_ft_completions.json"
    if not done(compl):
        run([
            sys.executable, "-u", "scripts/eval_gsm8k.py",
            "--lora-config", m["ft"], "--adapter-dir", m["adapter"],
            "--split", "test", "--num-problems", str(TEST_N),
            "--base-prompt", "instruct", "--ft-prompt", "instruct",
            "--batch-size", "64", "--max-new-tokens", str(MNT),
            "--output", str(compl),
        ], logs / "completions.log")

    # 2) batch=1 latency benchmark
    lat = res / "latency.json"
    if not done(lat):
        run([
            sys.executable, "-u", "scripts/bench_latency.py",
            "--base-cfg", m["base"], "--ft-cfg", m["ft"], "--adapter", m["adapter"],
            "--n", str(LAT_N), "--prompt", "instruct", "--mnt", str(MNT),
            "--out", str(lat),
        ], logs / "latency.log")

    d = json.loads(lat.read_text())
    print(f"[fta] {args.model} DONE | base {d['base']['mean_gen_tokens']:.0f}tok/"
          f"{d['base']['mean_wall_sec']*1000:.0f}ms  ft "
          f"{d['ft']['mean_gen_tokens']:.0f}tok/{d['ft']['mean_wall_sec']*1000:.0f}ms  "
          f"speedup x{d['speedup_wall']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
