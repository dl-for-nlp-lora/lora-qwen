"""
Presentation Plots for Qwen3 Original Results
Slides 5-9: E1, E2, Budget Comparison, R32 Sweep
"""

import json
import csv
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("D:/DNLP/lora-qwen/plots/presentation")
DATA_DIR = Path("D:/DNLP/lora-qwen/results")

# Professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

# Color palette
COLORS = {
    'base': '#2E86AB',
    'lora': '#A23B72',
    'full_ft': '#F18F01',
    'q': '#4ECDC4',
    'v': '#FF6B6B',
    'qv': '#95E1D3',
    'attention': '#F38181',
    'all_linear': '#AA96DA',
    'grid': '#E5E5E5'
}


def load_csv(filename):
    """Load CSV file and return list of dicts."""
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filename} not found")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_slide5_experiment_design():
    """
    Slide 5: Experiment Design - What we replicated vs what we didn't
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.axis('off')
    
    # Paper experiments we replicated
    replicated = [
        "E1: Which weight matrices? (§7.1, Tab. 5)",
        "  - Iso-budget target sweep (~1.6M params)",
        "  - Configs: q, v, q+v, attention, all-linear",
        "",
        "E2: Optimal rank? (§7.2, Tab. 6)",
        "  - Rank sweep: r ∈ {1, 2, 4, 8, 16, 32, 64}",
        "  - Best target from E1",
        "",
        "E3: Subspace analysis (§7.3, Fig. 3-4)",
        "  - Grassmann similarity heatmaps",
        "  - Cross-seed consistency",
        "  - Amplification factors"
    ]
    
    # Paper experiments we didn't replicate
    not_replicated = [
        "§5.2-5.3: RoBERTa/DeBERTa on GLUE",
        "  - Encoder models, classification heads",
        "  - Not meaningful for decoder-only LoRA impl",
        "",
        "§5.4: GPT-2 on NLG (E2E/WebNLG/DART)",
        "  - BLEU/METEOR on outdated benchmarks",
        "  - GSM8K accuracy more relevant for 2024 LMs",
        "",
        "§5.5: GPT-3 175B scale-up",
        "  - Requires multi-node cluster (350GB weights)",
        "",
        "§7.2 subspace on 96 layers",
        "  - Qwen3-1.7B has 28 layers (we run same analysis)"
    ]
    
    # Our contributions
    contributions = [
        "E4: Headroom study (our novel contribution)",
        "  - Qwen2-1.5B vs Qwen2.5-1.5B (controlled pair)",
        "  - How LoRA gain depends on base knowledge",
        "",
        "Extended E1: Budget ablation",
        "  - ~1.6M, ~2.6M, ~4.4M param budgets",
        "  - All-linear at higher ranks",
        "",
        "Modern architecture reproduction",
        "  - Qwen3-1.7B (GQA, decoder-only)",
        "  - MetaMathQA/GSM8K (2024 benchmarks)"
    ]
    
    # Create text boxes
    ax.text(0.02, 0.95, "REPLICATED FROM PAPER", transform=ax.transAxes, 
            fontsize=14, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor=COLORS['q'], alpha=0.3))
    ax.text(0.05, 0.90, '\n'.join(replicated), transform=ax.transAxes,
            fontsize=10, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['q'], linewidth=1.5))
    
    ax.text(0.52, 0.95, "NOT REPLICATED (with justification)", transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='#FFB6B6', alpha=0.3))
    ax.text(0.55, 0.90, '\n'.join(not_replicated), transform=ax.transAxes,
            fontsize=9, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#FF6B6B', linewidth=1.5))
    
    ax.text(0.02, 0.48, "OUR CONTRIBUTIONS", transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor=COLORS['lora'], alpha=0.3))
    ax.text(0.05, 0.43, '\n'.join(contributions), transform=ax.transAxes,
            fontsize=10, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['lora'], linewidth=1.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Slide 5: Experimental Design Overview', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide5_experiment_design.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: slide5_experiment_design.png")


def plot_slide6_e1_table():
    """
    Slide 6: E1 Results Table (Original ~1.6M budget)
    """
    data = load_csv('e1_original_table.csv')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.axis('off')
    
    # Table data
    columns = ['Config', 'Targets', 'Rank', 'α', 'Trainable', 'Base', 'FT', 'Δ']
    table_data = []
    
    for row in data:
        try:
            trainable = f"{int(row['Trainable']):,}" if row['Trainable'] else 'N/A'
            rank = row['Rank'] if row['Rank'] else 'N/A'
            alpha = row['Alpha'] if row['Alpha'] else 'N/A'
            table_data.append([
                row['Config'],
                row['Targets'],
                rank,
                alpha,
                trainable,
                row['Base Acc'],
                row['FT Acc'],
                row['Delta']
            ])
        except Exception as e:
            print(f"Skipping row {row['Config']}: {e}")
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=columns, loc='center',
                     cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor(COLORS['base'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')
        table[(0, i)].set_edgecolor('black')
        table[(0, i)].set_linewidth(1.5)
    
    # Style data rows
    colors = [COLORS['q'], COLORS['v'], COLORS['qv'], COLORS['attention'], COLORS['all_linear']]
    for i in range(1, len(table_data) + 1):
        color = colors[(i-1) % len(colors)]
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color if j == 1 else 'white')
            table[(i, j)].set_edgecolor('black')
            table[(i, j)].set_linewidth(1)
            if j in [5, 6, 7]:  # Accuracy columns
                table[(i, j)].set_text_props(fontweight='bold')
    
    ax.set_title('Slide 6: E1 - Which Weight Matrices? (Original ~1.6M Budget)\n'
                'Qwen3-1.7B-Base, 256 tokens, MetamathQA',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide6_e1_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: slide6_e1_table.png")


def plot_slide7_e2_rank_sweep():
    """
    Slide 7: E2 - All Rank Sweep Results
    """
    data = load_csv('e2_all_results.csv')
    
    # Group by target type
    targets = {}
    for row in data:
        target = row['Targets']
        if target not in targets:
            targets[target] = []
        try:
            rank = int(row['Rank'])
            ft_acc = float(row['FT Acc'])
            base_acc = float(row['Base Acc'])
            delta = float(row['Delta'])
            targets[target].append({
                'rank': rank,
                'ft': ft_acc,
                'base': base_acc,
                'delta': delta,
                'config': row['Config']
            })
        except:
            pass
    
    # Sort by rank
    for target in targets:
        targets[target].sort(key=lambda x: x['rank'])
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color mapping for target types
    target_colors = {
        'q+k+v+o+gate+up+down': COLORS['all_linear'],
        'q+k+v+o': COLORS['attention'],
        'q+v': COLORS['qv'],
        'q': COLORS['q'],
        'v': COLORS['v']
    }
    
    # Plot 1: All-linear
    ax = axes[0, 0]
    target = 'q+k+v+o+gate+up+down'
    if target in targets:
        ranks = [d['rank'] for d in targets[target]]
        ft_accs = [d['ft'] for d in targets[target]]
        base_accs = [d['base'] for d in targets[target]]
        
        ax.plot(ranks, ft_accs, 'o-', color=COLORS['all_linear'], linewidth=2, markersize=8, label='LoRA FT')
        ax.axhline(y=np.mean(base_accs), color='gray', linestyle='--', linewidth=2, label='Base')
        
        for r, ft, base in zip(ranks, ft_accs, base_accs):
            ax.annotate(f'{ft:.3f}', xy=(r, ft), ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Rank')
        ax.set_ylabel('GSM8K Accuracy')
        ax.set_title('All Linear (7 layers)\nBest: r=1 (0.704)', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
    
    # Plot 2: Attention only
    ax = axes[0, 1]
    target = 'q+k+v+o'
    if target in targets:
        ranks = [d['rank'] for d in targets[target]]
        ft_accs = [d['ft'] for d in targets[target]]
        base_accs = [d['base'] for d in targets[target]]
        
        ax.plot(ranks, ft_accs, 'o-', color=COLORS['attention'], linewidth=2, markersize=8, label='LoRA FT')
        ax.axhline(y=np.mean(base_accs), color='gray', linestyle='--', linewidth=2, label='Base')
        
        for r, ft in zip(ranks, ft_accs):
            ax.annotate(f'{ft:.3f}', xy=(r, ft), ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Rank')
        ax.set_ylabel('GSM8K Accuracy')
        ax.set_title('Attention Only (4 layers)\nBest: r=16 (0.710)', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
    
    # Plot 3: Q+V only
    ax = axes[1, 0]
    target = 'q+v'
    if target in targets:
        ranks = [d['rank'] for d in targets[target]]
        ft_accs = [d['ft'] for d in targets[target]]
        base_accs = [d['base'] for d in targets[target]]
        
        ax.plot(ranks, ft_accs, 'o-', color=COLORS['qv'], linewidth=2, markersize=8, label='LoRA FT')
        ax.axhline(y=np.mean(base_accs), color='gray', linestyle='--', linewidth=2, label='Base')
        
        for r, ft in zip(ranks, ft_accs):
            ax.annotate(f'{ft:.3f}', xy=(r, ft), ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Rank')
        ax.set_ylabel('GSM8K Accuracy')
        ax.set_title('Q+V Proj Only\nBest: r=2 (0.708)', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
    
    # Plot 4: Summary - Best per target
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "E2 RANK SWEEP SUMMARY\n\n"
    for target, results in targets.items():
        if results:
            best = max(results, key=lambda x: x['ft'])
            target_short = target.replace('+', ' + ').replace('gate+up+down', 'MLP')
            summary_text += f"{target_short}:\n"
            summary_text += f"  Best rank: {best['rank']}, FT: {best['ft']:.4f}, Δ: {best['delta']:+.4f}\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            family='monospace', va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', linewidth=1))
    
    plt.suptitle('Slide 7: E2 - Rank Sweep Results (All Configurations)\n'
                'Qwen3-1.7B-Base, 256 tokens, MetamathQA',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide7_e2_rank_sweep.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: slide7_e2_rank_sweep.png")


def plot_slide8_all_linear_contribution():
    """
    Slide 8: Our Contribution - All Linear Fine-tuning
    Compare all-linear vs attention-only across different ranks
    """
    e2_data = load_csv('e2_all_results.csv')
    
    # Extract all-linear and attention configs
    all_linear = []
    attention = []
    
    for row in e2_data:
        try:
            rank = int(row['Rank'])
            ft_acc = float(row['FT Acc'])
            base_acc = float(row['Base Acc'])
            delta = float(row['Delta'])
            
            if row['Targets'] == 'q+k+v+o+gate+up+down':
                all_linear.append({'rank': rank, 'ft': ft_acc, 'base': base_acc, 'delta': delta})
            elif row['Targets'] == 'q+k+v+o':
                attention.append({'rank': rank, 'ft': ft_acc, 'base': base_acc, 'delta': delta})
        except:
            pass
    
    # Sort by rank
    all_linear.sort(key=lambda x: x['rank'])
    attention.sort(key=lambda x: x['rank'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy comparison
    ax = axes[0]
    
    all_linear_ranks = [d['rank'] for d in all_linear]
    all_linear_accs = [d['ft'] for d in all_linear]
    attention_ranks = [d['rank'] for d in attention]
    attention_accs = [d['ft'] for d in attention]
    
    ax.plot(all_linear_ranks, all_linear_accs, 's-', color=COLORS['all_linear'], 
            linewidth=2.5, markersize=10, label='All Linear (7 layers)', markeredgecolor='black')
    ax.plot(attention_ranks, attention_accs, 'o-', color=COLORS['attention'], 
            linewidth=2.5, markersize=10, label='Attention Only (4 layers)', markeredgecolor='black')
    
    # Annotate best points
    best_al = max(all_linear, key=lambda x: x['ft'])
    best_attn = max(attention, key=lambda x: x['ft'])
    
    ax.annotate(f'Best: r={best_al["rank"]}\n({best_al["ft"]:.3f})',
               xy=(best_al['rank'], best_al['ft']),
               xytext=(best_al['rank']+5, best_al['ft']+0.01),
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=COLORS['all_linear'], alpha=0.7))
    
    ax.annotate(f'Best: r={best_attn["rank"]}\n({best_attn["ft"]:.3f})',
               xy=(best_attn['rank'], best_attn['ft']),
               xytext=(best_attn['rank']+5, best_attn['ft']-0.02),
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=COLORS['attention'], alpha=0.7))
    
    ax.set_xlabel('Rank', fontsize=11)
    ax.set_ylabel('GSM8K Test Accuracy', fontsize=11)
    ax.set_title('All Linear vs Attention Only\n'
                'All-linear competitive at low rank',
                fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Delta comparison
    ax = axes[1]
    
    all_linear_deltas = [d['delta'] for d in all_linear]
    attention_deltas = [d['delta'] for d in attention]
    
    ax.plot(all_linear_ranks, all_linear_deltas, 's-', color=COLORS['all_linear'], 
            linewidth=2.5, markersize=10, label='All Linear', markeredgecolor='black')
    ax.plot(attention_ranks, attention_deltas, 'o-', color=COLORS['attention'], 
            linewidth=2.5, markersize=10, label='Attention Only', markeredgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    
    ax.set_xlabel('Rank', fontsize=11)
    ax.set_ylabel('LoRA Gain (Δ Accuracy)', fontsize=11)
    ax.set_title('Adaptation Gain Comparison\n'
                'Both show similar gains (~0.21-0.24)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.suptitle('Slide 8: Our Contribution - All Linear Fine-tuning\n'
                'Key Finding: Adapting MLP layers (gate/up/down) is competitive with attention-only',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide8_all_linear_contribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: slide8_all_linear_contribution.png")


def plot_slide9_budget_comparison():
    """
    Slide 9: Budget Ablation - How parameter budget affects learning
    Compare ~1.6M, ~2.6M, ~4.4M budgets
    """
    budget_data = load_csv('budget_comparison.csv')
    
    # Group by budget
    budgets = {'~1.6M': [], '~2.6M': [], '~4.4M': []}
    for row in budget_data:
        budget = row['Budget']
        if budget in budgets:
            try:
                ft_acc = float(row['FT Acc'])
                delta = float(row['Delta'])
                budgets[budget].append({
                    'config': row['Config'],
                    'targets': row['Targets'],
                    'ft': ft_acc,
                    'delta': delta,
                    'trainable': int(row['Trainable'])
                })
            except:
                pass
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Best per budget
    ax = axes[0]
    
    budget_names = list(budgets.keys())
    best_ft = []
    best_configs = []
    
    for budget in budget_names:
        if budgets[budget]:
            best = max(budgets[budget], key=lambda x: x['ft'])
            best_ft.append(best['ft'])
            best_configs.append(f"{best['config']}\n({best['trainable']:,} params)")
        else:
            best_ft.append(0)
            best_configs.append('N/A')
    
    bars = ax.bar(budget_names, best_ft, color=[COLORS['q'], COLORS['v'], COLORS['qv']],
                  edgecolor='black', linewidth=1.5)
    
    # Annotate bars
    for i, (budget, ft, config) in enumerate(zip(budget_names, best_ft, best_configs)):
        ax.annotate(f'{ft:.4f}\n{config}',
                   xy=(i, ft),
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Best GSM8K Accuracy', fontsize=11)
    ax.set_title('Best Configuration per Budget', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.75)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: All configs per budget (scatter)
    ax = axes[1]
    
    colors_map = {'~1.6M': COLORS['q'], '~2.6M': COLORS['v'], '~4.4M': COLORS['qv']}
    
    for budget, configs in budgets.items():
        if configs:
            trainable = [c['trainable'] for c in configs]
            ft_accs = [c['ft'] for c in configs]
            ax.scatter(trainable, ft_accs, s=150, c=colors_map[budget], 
                      edgecolors='black', linewidth=1.5, label=budget, alpha=0.7)
            
            # Annotate points
            for c in configs:
                ax.annotate(c['config'].replace('E1_', '').replace('E4.4m_', '').replace('.json', '')[:10],
                           xy=(c['trainable'], c['ft']),
                           ha='center', va='bottom',
                           fontsize=7)
    
    ax.set_xlabel('Trainable Parameters', fontsize=11)
    ax.set_ylabel('GSM8K Accuracy', fontsize=11)
    ax.set_title('All Configurations by Parameter Count', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.suptitle('Slide 9: Budget Ablation Study\n'
                'Key Finding: Higher budget (~4.4M) with attention-only achieves best results (0.716)',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide9_budget_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: slide9_budget_comparison.png")


def plot_slide10_r32_comparison():
    """
    Slide 10: R32 Sweep - Higher budget impact
    Compare all configs at rank=32
    """
    r32_data = load_csv('e_r32_sweep.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bar chart comparison
    ax = axes[0]
    
    configs = []
    ft_accs = []
    deltas = []
    targets = []
    
    for row in r32_data:
        try:
            configs.append(row['Config'].replace('E_r32_', '').replace('.json', ''))
            ft_accs.append(float(row['FT Acc']))
            deltas.append(float(row['Delta']))
            targets.append(row['Targets'])
        except:
            pass
    
    colors = [COLORS.get('all_linear') if 'all' in t else 
              COLORS.get('attention') if 'q+k+v+o' in t else
              COLORS.get('qv') if 'q+v' in t else
              COLORS.get('q') if t == 'q' else
              COLORS.get('v')
              for t in targets]
    
    bars = ax.bar(range(len(configs)), ft_accs, color=colors, edgecolor='black', linewidth=1.5)
    
    # Annotate bars
    for i, (cfg, ft, delta) in enumerate(zip(configs, ft_accs, deltas)):
        ax.annotate(f'{ft:.4f}\n(Δ={delta:+.3f})',
                   xy=(i, ft),
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=9, rotation=15)
    ax.set_ylabel('GSM8K Accuracy', fontsize=11)
    ax.set_title('Rank-32 Comparison Across Target Modules\n'
                'Best: q_proj (0.712)',
                fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.75)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Comparison with E1/E2 at similar ranks
    ax = axes[1]
    
    # Get r=32 data from E2 for comparison
    e2_data = load_csv('e2_all_results.csv')
    e2_r32 = []
    for row in e2_data:
        try:
            if int(row['Rank']) == 32:
                target = row['Targets']
                if 'all' in target:
                    e2_r32.append({'target': 'all_linear_1.6M', 'ft': float(row['FT Acc']), 'budget': '1.6M'})
                elif 'q+k+v+o' in target:
                    e2_r32.append({'target': 'attention_1.6M', 'ft': float(row['FT Acc']), 'budget': '1.6M'})
        except:
            pass
    
    # Add R32 sweep data
    for row in r32_data:
        try:
            target = row['Targets']
            ft = float(row['FT Acc'])
            if 'all' in target:
                e2_r32.append({'target': 'all_linear_4.4M', 'ft': ft, 'budget': '4.4M'})
            elif 'q+k+v+o' in target:
                e2_r32.append({'target': 'attention_4.4M', 'ft': ft, 'budget': '4.4M'})
            elif target == 'q':
                e2_r32.append({'target': 'q_proj_4.4M', 'ft': ft, 'budget': '4.4M'})
        except:
            pass
    
    # Plot
    labels = [d['target'] for d in e2_r32]
    values = [d['ft'] for d in e2_r32]
    colors = [COLORS['all_linear'] if 'all' in l else 
              COLORS['attention'] if 'attention' in l else
              COLORS['q']
              for l in labels]
    
    ax.bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=1.5)
    
    for i, (label, val) in enumerate(zip(labels, values)):
        ax.annotate(f'{val:.4f}',
                   xy=(i, val),
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('GSM8K Accuracy', fontsize=11)
    ax.set_title('Rank-32: Budget Comparison\n'
                'Higher budget improves q_proj significantly',
                fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.75)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Slide 10: R32 Sweep - Higher Budget Impact\n'
                'Key Finding: At fixed rank=32, higher budget (~4.4M) shows mixed results vs ~1.6M',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'slide10_r32_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: slide10_r32_comparison.png")


def main():
    """Generate all presentation plots."""
    print("=" * 80)
    print("GENERATING QWEN3 ORIGINAL RESULTS PRESENTATION PLOTS")
    print("=" * 80)
    print()
    
    plot_slide5_experiment_design()
    plot_slide6_e1_table()
    plot_slide7_e2_rank_sweep()
    plot_slide8_all_linear_contribution()
    plot_slide9_budget_comparison()
    plot_slide10_r32_comparison()
    
    print()
    print("=" * 80)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Output directory:", OUTPUT_DIR)
    print()
    print("Generated files:")
    for f in sorted(OUTPUT_DIR.glob('slide*.png')):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
