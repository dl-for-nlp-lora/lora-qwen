"""Extract all Qwen3 original results for presentation - Simplified version"""

import json
import os
import csv
from pathlib import Path

RESULTS_BASE = Path("D:/DNLP/lora-qwen/results")
OUTPUT_DIR = RESULTS_BASE.parent / "plots" / "presentation"
OUTPUT_DIR.mkdir(exist_ok=True)

def safe_get(d, *keys, default=None):
    """Safely get nested dictionary values."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d

def extract_folder(folder_name):
    """Extract results from a folder."""
    folder = RESULTS_BASE / folder_name
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return []
    
    results = []
    for f in sorted(folder.iterdir()):
        if f.suffix != '.json':
            continue
        
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                d = json.load(fp)
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
            continue
        
        base_acc = safe_get(d, 'base', 'accuracy')
        ft_acc = safe_get(d, 'ft', 'accuracy')
        lora = d.get('lora', {})
        rank = lora.get('rank')
        targets = lora.get('target_modules', [])
        target_name = '+'.join([t.replace('_proj','') for t in targets]) if targets else 'N/A'
        trainable = lora.get('expected_trainable_params')
        
        delta = None
        if base_acc is not None and ft_acc is not None:
            delta = ft_acc - base_acc
        
        results.append({
            'file': f.name,
            'base': base_acc,
            'ft': ft_acc,
            'delta': delta,
            'rank': rank,
            'targets': target_name,
            'trainable': trainable
        })
    
    return results

print("=" * 80)
print("QWEN3 ORIGINAL RESULTS (256 tokens, MetamathQA)")
print("=" * 80)

# Extract all folders
print("\nExtracting results...")
e1 = extract_folder("E1")
e1_rebal = extract_folder("E1_rebalanced_2.6m_params")
e44m = extract_folder("E_4.4m_params")
e2 = extract_folder("E2")
e_r32 = extract_folder("E_r32_sweep")

# Print summaries
def print_results(name, results):
    print(f"\n### {name} ###")
    for r in results:
        if r['delta'] is not None:
            print(f"{r['file']}: t={r['targets']}, r={r['rank']}, base={r['base']:.4f}, ft={r['ft']:.4f}, d={r['delta']:+.4f}")
        else:
            print(f"{r['file']}: INCOMPLETE")

print_results("E1 (~1.6M)", e1)
print_results("E1_rebalanced (~2.6M)", e1_rebal)
print_results("E_4.4m_params (~4.4M)", e44m)
print_results("E2 (all configs)", e2)
print_results("E_r32_sweep", e_r32)

# Write CSVs
print("\n\n### Writing CSV files ###")

# E1 table
with open(OUTPUT_DIR / "e1_original_table.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Config', 'Targets', 'Rank', 'Alpha', 'Trainable', 'Base Acc', 'FT Acc', 'Delta'])
    for r in e1:
        if r['delta'] is not None:
            alpha = r['rank'] * 2 if r['rank'] else 'N/A'
            writer.writerow([
                r['file'].replace('.json', '').replace('E1_', ''),
                r['targets'],
                r['rank'],
                alpha,
                r['trainable'],
                f"{r['base']:.4f}",
                f"{r['ft']:.4f}",
                f"{r['delta']:+.4f}"
            ])
print("[OK] e1_original_table.csv")

# E2 all results
with open(OUTPUT_DIR / "e2_all_results.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Config', 'Targets', 'Rank', 'Base Acc', 'FT Acc', 'Delta'])
    for r in e2:
        if r['delta'] is not None:
            writer.writerow([
                r['file'].replace('.json', ''),
                r['targets'],
                r['rank'],
                f"{r['base']:.4f}",
                f"{r['ft']:.4f}",
                f"{r['delta']:+.4f}"
            ])
print("[OK] e2_all_results.csv")

# Budget comparison
with open(OUTPUT_DIR / "budget_comparison.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Budget', 'Config', 'Targets', 'Rank', 'Trainable', 'Base Acc', 'FT Acc', 'Delta'])
    for r in e1:
        if r['delta'] is not None:
            writer.writerow(['~1.6M', r['file'].replace('.json', ''), r['targets'], r['rank'], r['trainable'], f"{r['base']:.4f}", f"{r['ft']:.4f}", f"{r['delta']:+.4f}"])
    for r in e1_rebal:
        if r['delta'] is not None:
            writer.writerow(['~2.6M', r['file'].replace('.json', ''), r['targets'], r['rank'], r['trainable'], f"{r['base']:.4f}", f"{r['ft']:.4f}", f"{r['delta']:+.4f}"])
    for r in e44m:
        if r['delta'] is not None:
            writer.writerow(['~4.4M', r['file'].replace('.json', ''), r['targets'], r['rank'], r['trainable'], f"{r['base']:.4f}", f"{r['ft']:.4f}", f"{r['delta']:+.4f}"])
print("[OK] budget_comparison.csv")

# R32 sweep
with open(OUTPUT_DIR / "e_r32_sweep.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Config', 'Targets', 'Rank', 'Trainable', 'Base Acc', 'FT Acc', 'Delta'])
    for r in e_r32:
        if r['delta'] is not None:
            writer.writerow([r['file'].replace('.json', ''), r['targets'], r['rank'], r['trainable'], f"{r['base']:.4f}", f"{r['ft']:.4f}", f"{r['delta']:+.4f}"])
print("[OK] e_r32_sweep.csv")

# E2 attention-only subset for plotting
e2_attn = [r for r in e2 if r['targets'] in ['q+k+v+o', 'q', 'v', 'q+v']]
with open(OUTPUT_DIR / "e2_attention_rank_sweep.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Config', 'Targets', 'Rank', 'Base Acc', 'FT Acc', 'Delta'])
    for r in e2_attn:
        if r['delta'] is not None:
            writer.writerow([r['file'].replace('.json', ''), r['targets'], r['rank'], f"{r['base']:.4f}", f"{r['ft']:.4f}", f"{r['delta']:+.4f}"])
print("[OK] e2_attention_rank_sweep.csv")

print("\n[OK] All CSV files saved to:", OUTPUT_DIR)
