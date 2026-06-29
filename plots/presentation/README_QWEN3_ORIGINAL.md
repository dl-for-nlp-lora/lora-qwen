# Qwen3 original results — presentation slides 5–10

## Overview

This folder contains the presentation slides and data for the **first-cohort
Qwen3-1.7B experiments** (before the later methodology corrections). They are
kept for the LoRA-reproduction part of the talk (E1/E2 ablations); the
methodology-fixed numbers live in
[`../../analysis/headroom_summary.md`](../../analysis/headroom_summary.md). This
cohort used:
- **Dataset:** MetaMathQA (30k examples)
- **Token limit:** 256 tokens (later found to truncate some outputs)
- **Evaluation:** GSM8K test (1,319 problems)
- **Model:** Qwen3-1.7B-Base

---

## Generated Slides

### Slide 5: `slide5_experiment_design.png`
**Title:** "Experimental Design Overview"

**Content:**
- **Replicated from paper:**
  - E1: Which weight matrices? (§7.1)
  - E2: Optimal rank? (§7.2)
  - E3: Subspace analysis (§7.3)

- **Not replicated (with justification):**
  - §5.2-5.3: RoBERTa/DeBERTa on GLUE (encoder models)
  - §5.4: GPT-2 NLG benchmarks (outdated)
  - §5.5: GPT-3 175B (requires multi-node cluster)

- **Our contributions:**
  - E4: Headroom study (Qwen2 vs Qwen2.5)
  - Extended E1: Budget ablation (~1.6M, ~2.6M, ~4.4M)
  - Modern architecture: Qwen3-1.7B with GQA

**Use:** Overview slide showing what we reproduced vs what we didn't

---

### Slide 6: `slide6_e1_table.png`
**Title:** "E1 - Which Weight Matrices? (Original ~1.6M Budget)"

**Table shows:**
| Config | Targets | Rank | α | Trainable | Base | FT | Δ |
|--------|---------|------|---|-----------|------|----|---|
| E1a_q_proj | q | 14 | 28 | 1,605,632 | 0.472 | 0.695 | +0.224 |
| E1b_v_proj | v | 19 | 38 | 1,634,304 | 0.472 | 0.685 | +0.213 |
| E1c_qv_proj | N/A | - | - | - | 0.488 | 0.688 | +0.199 |
| E1d_attention | N/A | - | - | - | 0.488 | 0.698 | +0.209 |
| E1e_all_linear | N/A | - | - | - | 0.488 | 0.700 | +0.212 |

**Note:** Last 3 configs have different base accuracy (different random seed or data split)

**Key Finding:** All configs show similar gains (+0.20 to +0.22), target choice matters less than expected

---

### Slide 7: `slide7_e2_rank_sweep.png`
**Title:** "E2 - Rank Sweep Results (All Configurations)"

**Four panels:**
1. **All Linear (7 layers):** Best at r=1 (0.704)
2. **Attention Only (4 layers):** Best at r=16 (0.710)
3. **Q+V Proj Only:** Best at r=2 (0.708)
4. **Summary text:** Best rank and accuracy per target type

**Key Finding:** Low rank suffices; performance plateaus early; best overall: attention-only at r=16

---

### Slide 8: `slide8_all_linear_contribution.png`
**Title:** "Our Contribution - All Linear Fine-tuning"

**Two panels:**
1. **Accuracy comparison:** All-linear vs attention-only across ranks
   - All-linear best: r=1 (0.704)
   - Attention-only best: r=16 (0.710)

2. **Delta comparison:** LoRA gain for both methods
   - Both show similar gains (~0.21-0.24)

**Key Finding:** Adapting MLP layers (gate/up/down) is competitive with attention-only, challenging the paper's assumption to freeze MLP

---

### Slide 9: `slide9_budget_comparison.png`
**Title:** "Budget Ablation Study"

**Two panels:**
1. **Best per budget:**
   - ~1.6M: E1a_q_proj (0.695)
   - ~2.6M: E1a_q_proj_rebalanced (0.707)
   - ~4.4M: E4.4m_attention (0.716)

2. **All configs by parameter count:** Scatter plot showing all configurations

**Key Finding:** Higher budget (~4.4M) with attention-only achieves best results (0.716)

---

### Slide 10: `slide10_r32_comparison.png`
**Title:** "R32 Sweep - Higher Budget Impact"

**Two panels:**
1. **Rank-32 comparison across targets:**
   - q_proj: 0.712 (best)
   - attention: 0.704
   - qv_proj: 0.697
   - v_proj: 0.686
   - all_linear: 0.694

2. **Budget comparison at rank=32:**
   - Compares ~1.6M vs ~4.4M budget for same target types

**Key Finding:** At fixed rank=32, higher budget shows mixed results; q_proj benefits most from increased budget

---

## Data Files

### CSV Exports

| File | Description |
|------|-------------|
| `e1_original_table.csv` | E1 target sweep (~1.6M budget) |
| `e2_all_results.csv` | All E2 rank sweep results |
| `budget_comparison.csv` | Combined E1, E1_rebalanced, E_4.4m |
| `e_r32_sweep.csv` | R32 sweep results |
| `e2_attention_rank_sweep.csv` | E2 attention-only subset |

### Raw results location

All results from the repo's `results/` directory:
- `E1/` — original ~1.6M budget
- `E1_rebalanced_2.6m_params/` — ~2.6M budget
- `E_4.4m_params/` — ~4.4M budget
- `E2/` — rank sweep (all configs)
- `E_r32_sweep/` — fixed rank=32 sweep

---

## Key Numbers Summary

### E1 Results (~1.6M budget)
- **Best:** E1a_q_proj (0.695, Δ=+0.224)
- **Worst:** E1c_qv_proj (0.688, Δ=+0.199)
- **Range:** 0.007 (within noise)

### E2 Results (best per target)
- **All-linear:** r=1 → 0.704
- **Attention:** r=16 → 0.710
- **Q+V:** r=2 → 0.708

### Budget Ablation
- **~1.6M:** 0.695 (q_proj)
- **~2.6M:** 0.707 (q_proj rebalanced)
- **~4.4M:** 0.716 (attention)

### R32 Sweep
- **Best:** q_proj at 0.712
- **All configs within:** 0.026 (2 SE)

---

## Statistical Notes

- **n=1319** test problems
- **Binomial SE** at acc=0.70: ±0.013 (±17 problems)
- **2 SE threshold:** ±0.026
- Differences <2 SE are **not statistically distinguishable from noise**

**Implication:** Most E1/E2 differences are within 1-2 SE, supporting the LoRA paper's finding that "target choice and rank matter less than expected"

---

## Limitations (Discovered Later)

1. **256-token truncation:** Many solutions cut off before `#### N` format
2. **MetamathQA mismatch:** Training data doesn't match eval format (no system prompt)
3. **Qwen3 contamination:** Base model already pretrained on GSM8K (base acc ~0.47-0.49 zeroshot)
4. **Single seed:** All results from one random seed

**Corrected setup:** See `headroom_summary.md` and Qwen2/Qwen2.5 results for methodology-fixed experiments

---

## How to regenerate

```bash
# from the repo root
python plots/presentation/extract_qwen3_results.py   # 1. extract data from JSON results
python plots/presentation/plot_qwen3_original.py     # 2. generate all plots
```

Output files are saved to `plots/presentation/`.

---

## Presentation order

**Recommended slide sequence:**
1. **Slide 5** — experiment design overview (what we replicated vs didn't)
2. **Slide 6** — E1 table (original ~1.6M budget results)
3. **Slide 7** — E2 rank sweep (all configurations)
4. **Slide 8** — all-linear fine-tuning (our extension)
5. **Slide 9** — budget ablation (how params affect learning)
6. **Slide 10** — R32 comparison (higher budget impact)

---

## Related docs

- [`../../README.md`](../../README.md) — project overview
- [`../../EXPERIMENTS.md`](../../EXPERIMENTS.md) — full experimental plan
- [`../../analysis/headroom_summary.md`](../../analysis/headroom_summary.md) — methodology-fixed results
