"""Generate iso-budget LoRA configs for the Qwen2-1.5B / Qwen2.5-1.5B headroom study.

Both backbones share identical dims (hidden=1536, 12 Q-heads x128 -> q/o_proj
1536->1536; 2 KV-heads -> k/v_proj 1536->256; 28 layers; intermediate=8960),
so the GQA-aware iso-budget ranks are the SAME for both models; only model_name
differs.

LoRA params for a target set: P = L * r * sum_m (in_m + out_m), L=28.
We solve r per target set for a common budget so E1 ("which matrix?") varies
*where* LoRA goes at (approximately) fixed trainable params, exactly as in
EXPERIMENTS.md. alpha = 2*r throughout (paper convention).

Writes:
  configs/qwen2_1.5b/base.yaml,   configs/qwen25_1.5b/base.yaml
  configs/E15b_qwen2/{q,v,qv,attention,all_linear}.yaml
  configs/E15b_qwen25/{q,v,qv,attention,all_linear}.yaml
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

L = 28
HIDDEN = 1536
KV = 256          # 2 kv-heads * 128
INTER = 8960

# (in + out) summed over the matrices in each target set, per layer.
PER_LAYER = {
    "q_proj": (HIDDEN + HIDDEN),
    "v_proj": (HIDDEN + KV),
    "k_proj": (HIDDEN + KV),
    "o_proj": (HIDDEN + HIDDEN),
    "gate_proj": (HIDDEN + INTER),
    "up_proj": (HIDDEN + INTER),
    "down_proj": (INTER + HIDDEN),
}

TARGET_SETS = {
    "q":          ["q_proj"],
    "v":          ["v_proj"],
    "qv":         ["q_proj", "v_proj"],
    "attention":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
}

BUDGET = 2_500_000   # ~2.5M trainable params target


def coef(modules: list[str]) -> int:
    return L * sum(PER_LAYER[m] for m in modules)


def solve_rank(modules: list[str], budget: int) -> int:
    c = coef(modules)
    r = max(1, round(budget / c))
    return r


def write_yaml(path: Path, model_name: str, name: str,
               targets: list[str] | None, rank: int, alpha: int,
               backend: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'name: "{name}"', f'model_name: "{model_name}"']
    if targets is None:
        lines.append("target_modules: null")
    else:
        lines.append("target_modules:")
        lines += [f"  - {t}" for t in targets]
    lines += [
        f"rank: {rank}",
        f"alpha: {alpha}",
        "dropout: 0.05",
        'dtype: "bfloat16"',
        'device: "auto"',
        f'backend: "{backend}"',
    ]
    path.write_text("\n".join(lines) + "\n")


MODELS = {
    "qwen2_1.5b":  "Qwen/Qwen2-1.5B",
    "qwen25_1.5b": "Qwen/Qwen2.5-1.5B",
}
E1_DIR = {"qwen2_1.5b": "E15b_qwen2", "qwen25_1.5b": "E15b_qwen25"}


def main() -> None:
    print(f"Budget target: {BUDGET:,} trainable params (L={L})\n")
    ranks = {}
    for key, mods in TARGET_SETS.items():
        r = solve_rank(mods, BUDGET)
        p = r * coef(mods)
        ranks[key] = r
        print(f"  {key:11s} targets={mods} -> r={r:3d} alpha={2*r:3d} "
              f"params={p:,} ({100*p/BUDGET-100:+.1f}% vs budget)")

    for mkey, mname in MODELS.items():
        write_yaml(REPO / "configs" / mkey / "base.yaml",
                   mname, f"{mkey}-base", None, 16, 32, "peft")
        for tkey, mods in TARGET_SETS.items():
            r = ranks[tkey]
            write_yaml(REPO / "configs" / E1_DIR[mkey] / f"{tkey}.yaml",
                       mname, f"{E1_DIR[mkey]}_{tkey}", mods, r, 2 * r, "custom")
    print("\nWrote base + E1 iso-budget configs for both models.")


if __name__ == "__main__":
    main()
