#!/usr/bin/env python3
"""
Generate minimalistic, professional publication figures

Style:
- Single gradient colors (blues/grays)
- Clean, lightweight text
- Minimal styling
- Academic appearance
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

# Professional minimalist style
sns.set_style("ticks")
sns.set_context("paper")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal'
})

print("="*70)
print("GENERATING MINIMALIST PROFESSIONAL FIGURES")
print("="*70)

Path('visualizations').mkdir(exist_ok=True)

# Load data from results directory
print("\nLoading experiment data...")

results_dir = Path(os.getenv('RESULTS_DIR', 'results'))

# Find latest experiment results
toxic_dirs = sorted(results_dir.glob('experiment_*'))
gender_dirs = sorted(results_dir.glob('gender_experiment_*'))
race_dirs = sorted(results_dir.glob('race_experiment_*'))

if not (toxic_dirs and gender_dirs and race_dirs):
    print("ERROR: Experiment results not found in results/")
    print("Please run experiments first:")
    print("  python scripts/run_full_experiment.py")
    print("  python scripts/run_gender_experiment.py")
    print("  python scripts/run_race_experiment.py")
    sys.exit(1)

# Use most recent experiments
toxic_file = list(toxic_dirs[-1].glob('pruned_neurons_*.json'))[0]
gender_file = list(gender_dirs[-1].glob('pruned_neurons_*.json'))[0]
race_file = list(race_dirs[-1].glob('pruned_neurons_*.json'))[0]

print(f"  Toxic: {toxic_file.parent.name}")
print(f"  Gender: {gender_file.parent.name}")
print(f"  Race: {race_file.parent.name}")

with open(toxic_file, 'r') as f:
    toxic_neurons = set((n['layer'], n['index']) for n in json.load(f)['neurons'])
with open(gender_file, 'r') as f:
    gender_neurons = set((n['layer'], n['index']) for n in json.load(f)['neurons'])
with open(race_file, 'r') as f:
    race_neurons = set((n['layer'], n['index']) for n in json.load(f)['neurons'])

toxic_gender = toxic_neurons & gender_neurons
toxic_race = toxic_neurons & race_neurons
gender_race = gender_neurons & race_neurons
triple = toxic_neurons & gender_neurons & race_neurons

print(f"✓ Data loaded")

# =============================================================================
# Figure 1: Multi-Bias Comparison (Single Gradient)
# =============================================================================
print("\n1. Multi-bias comparison (minimal style)...")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
fig.subplots_adjust(wspace=0.35, bottom=0.15, top=0.88)

# Single gradient: blues
colors = ['#1565c0', '#1976d2', '#1e88e5']  # Dark to light blue gradient
experiments = ['Toxicity', 'Gender', 'Race']

# Panel A: Bias Reductions
ax = axes[0]
reductions = [17.3, 14.1, 11.3]
bars = ax.bar(experiments, reductions, color=colors, alpha=0.75, edgecolor='#0d47a1', linewidth=0.8)
ax.set_ylabel('Bias Reduction (%)', fontsize=10)
ax.set_title('(a) Bias Suppression', fontsize=11, pad=8)
ax.set_ylim(0, 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

for bar, val in zip(bars, reductions):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9)

# Panel B: PPL Changes
ax = axes[1]
ppl_changes = [0.80, 0.95, 2.57]
bars = ax.bar(experiments, ppl_changes, color=colors, alpha=0.75, edgecolor='#0d47a1', linewidth=0.8)
ax.set_ylabel('Perplexity Increase (%)', fontsize=10)
ax.set_title('(b) General Capability', fontsize=11, pad=8)
ax.axhline(y=5, color='#757575', linestyle='--', linewidth=1, alpha=0.5)
ax.text(2.05, 5.3, '5% threshold', fontsize=8, ha='right', color='#757575')
ax.set_ylim(0, 6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

for bar, val in zip(bars, ppl_changes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)

# Panel C: Differential Signals
ax = axes[2]
differentials = [27.2, 39.9, 37.6]
bars = ax.bar(experiments, differentials, color=colors, alpha=0.75, edgecolor='#0d47a1', linewidth=0.8)
ax.set_ylabel('Differential Signal (%)', fontsize=10)
ax.set_title('(c) Behavior-Specificity', fontsize=11, pad=8)
ax.axhline(y=15, color='#757575', linestyle='--', linewidth=1, alpha=0.5)
ax.text(2.05, 15.8, '15% threshold', fontsize=8, ha='right', color='#757575')
ax.set_ylim(0, 44)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

for bar, val in zip(bars, differentials):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.2,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9)

plt.savefig('visualizations/fig1_multibias.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: fig1_multibias.png")
plt.close()

# =============================================================================
# Figure 2: Three-Way Venn (Minimal)
# =============================================================================
print("\n2. Three-way Venn breakdown (minimal)...")

fig, ax = plt.subplots(figsize=(8, 5))

# Calculate regions
toxic_only = len(toxic_neurons - gender_neurons - race_neurons)
gender_only = len(gender_neurons - toxic_neurons - race_neurons)
race_only = len(race_neurons - toxic_neurons - gender_neurons)
tox_gen_only = len((toxic_neurons & gender_neurons) - race_neurons)
tox_race_only = len((toxic_neurons & race_neurons) - gender_neurons)
gen_race_only = len((gender_neurons & race_neurons) - toxic_neurons)

regions = [
    ('Triple Overlap\n(All Three)', len(triple)),
    ('Gender ∩ Race', gen_race_only),
    ('Toxic ∩ Gender', tox_gen_only),
    ('Toxic ∩ Race', tox_race_only),
    ('Race Only', race_only),
    ('Gender Only', gender_only),
    ('Toxic Only', toxic_only)
]

labels = [r[0] for r in regions]
values = [r[1] for r in regions]

# Monochromatic gradient (dark to light gray/blue)
colors_gradient = ['#263238', '#37474f', '#546e7a', '#78909c', '#90a4ae', '#b0bec5', '#cfd8dc']

y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, values, color=colors_gradient, alpha=0.8, edgecolor='#37474f', linewidth=0.8)

ax.set_xlabel('Neuron Count', fontsize=10)
ax.set_title('Three-Way Circuit Overlap Regions', fontsize=11, pad=10)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0, max(values) * 1.18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Simple value labels
for bar, val in zip(bars, values):
    width = bar.get_width()
    ax.text(width + 1.5, bar.get_y() + bar.get_height()/2.,
            f'{val}', ha='left', va='center', fontsize=9)

# Subtle annotation
ax.text(0.98, 0.02, 'Enrichment: 1.05M×, p<0.001',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=8, style='italic', color='#546e7a')

plt.savefig('visualizations/fig2_threeway_venn.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: fig2_threeway_venn.png")
plt.close()

# =============================================================================
# Figure 3: Correlation Heatmap (Minimal Gradient)
# =============================================================================
print("\n3. Correlation heatmap (minimal)...")

fig, ax = plt.subplots(figsize=(5.5, 5))

corr_matrix = np.array([
    [1.000, 0.602, 0.370],
    [0.602, 1.000, 0.442],
    [0.370, 0.442, 1.000]
])

labels = ['Toxicity', 'Gender', 'Race']

# Single gradient: white to dark blue
cmap = sns.light_palette("#1565c0", as_cmap=True)

im = ax.imshow(corr_matrix, cmap=cmap, vmin=0.3, vmax=1.0, aspect='auto')

ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(3))
ax.set_xticklabels(labels, fontsize=10)
ax.set_yticklabels(labels, fontsize=10)

# Annotate values (only upper triangle to avoid redundancy)
for i in range(3):
    for j in range(i, 3):
        color = 'white' if corr_matrix[i, j] > 0.65 else '#263238'
        ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
               ha="center", va="center", color=color, fontsize=10)

# Minimal grid
for i in range(4):
    ax.axhline(i-0.5, color='white', linewidth=1.5)
    ax.axvline(i-0.5, color='white', linewidth=1.5)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Correlation', fontsize=9, rotation=270, labelpad=15)
cbar.ax.tick_params(labelsize=8)

ax.set_title('Differential Score Correlation Matrix\n(458,752 neurons)',
             fontsize=11, pad=10)

plt.savefig('visualizations/fig3_correlation.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: fig3_correlation.png")
plt.close()

# =============================================================================
# Figure 4: Layer Distribution (Minimal)
# =============================================================================
print("\n4. Layer distribution (minimal)...")

fig, ax = plt.subplots(figsize=(10, 4))

toxic_by_layer = Counter(int(layer.split('.')[2]) for layer, idx in toxic_neurons)
gender_by_layer = Counter(int(layer.split('.')[2]) for layer, idx in gender_neurons)
race_by_layer = Counter(int(layer.split('.')[2]) for layer, idx in race_neurons)

layers = range(32)
toxic_counts = [toxic_by_layer.get(i, 0) for i in layers]
gender_counts = [gender_by_layer.get(i, 0) for i in layers]
race_counts = [race_by_layer.get(i, 0) for i in layers]

x = np.arange(32)
width = 0.27

# Monochromatic gradient (dark to light)
color_toxic = '#37474f'
color_gender = '#546e7a'
color_race = '#78909c'

bars1 = ax.bar(x - width, toxic_counts, width, label='Toxicity',
               color=color_toxic, alpha=0.8, edgecolor='#263238', linewidth=0.5)
bars2 = ax.bar(x, gender_counts, width, label='Gender',
               color=color_gender, alpha=0.8, edgecolor='#263238', linewidth=0.5)
bars3 = ax.bar(x + width, race_counts, width, label='Race',
               color=color_race, alpha=0.8, edgecolor='#263238', linewidth=0.5)

ax.set_xlabel('Layer Number', fontsize=10)
ax.set_ylabel('Pruned Neurons', fontsize=10)
ax.set_title('Neuron Distribution Across Layers', fontsize=11, pad=8)
ax.set_xticks(x[::2])  # Every other tick to reduce crowding
ax.set_xticklabels([str(i) for i in range(0, 32, 2)], fontsize=8)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9, edgecolor='#90a4ae')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Subtle late-layer annotation
ax.axvspan(21.5, 31.5, alpha=0.05, color='#546e7a', zorder=0)
ax.text(26.5, max(max(toxic_counts), max(gender_counts), max(race_counts)) - 2,
        '70% late-layer', ha='center', fontsize=8, style='italic', color='#546e7a')

plt.savefig('visualizations/fig4_layer_dist.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: fig4_layer_dist.png")
plt.close()

# =============================================================================
# Figure 5: Pairwise Overlaps (Minimal Gradient)
# =============================================================================
print("\n5. Pairwise overlaps (minimal)...")

fig, ax = plt.subplots(figsize=(7, 4))

pairs = ['Toxic–Gender', 'Toxic–Race', 'Gender–Race']
overlaps = [len(toxic_gender), len(toxic_race), len(gender_race)]
percentages = [11, 6, 15]

# Gradient: dark to light
colors_grad = ['#455a64', '#607d8b', '#90a4ae']

bars = ax.bar(pairs, overlaps, color=colors_grad, alpha=0.8, edgecolor='#37474f', linewidth=0.8)

ax.set_ylabel('Overlapping Neurons', fontsize=10)
ax.set_title('Pairwise Circuit Overlaps', fontsize=11, pad=8)
ax.set_ylim(0, max(overlaps) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

for bar, val, pct in zip(bars, overlaps, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.4,
            f'{val}\n({pct}%)', ha='center', va='bottom', fontsize=9, linespacing=1.2)

plt.savefig('visualizations/fig5_pairwise.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: fig5_pairwise.png")
plt.close()

# =============================================================================
# Figure 6: Summary 2-Panel (Correlation + Overlaps)
# =============================================================================
print("\n6. Summary panel (correlation + overlaps)...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(wspace=0.35)

# Left: Correlation heatmap
ax = axes[0]

corr_matrix = np.array([
    [1.000, 0.602, 0.370],
    [0.602, 1.000, 0.442],
    [0.370, 0.442, 1.000]
])

cmap = sns.light_palette("#1565c0", as_cmap=True)
im = ax.imshow(corr_matrix, cmap=cmap, vmin=0.3, vmax=1.0, aspect='auto')

ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(3))
ax.set_xticklabels(['Toxicity', 'Gender', 'Race'], fontsize=9)
ax.set_yticklabels(['Toxicity', 'Gender', 'Race'], fontsize=9)

# Annotate
for i in range(3):
    for j in range(i, 3):
        color = 'white' if corr_matrix[i, j] > 0.65 else '#263238'
        ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
               ha="center", va="center", color=color, fontsize=9)

for i in range(4):
    ax.axhline(i-0.5, color='white', linewidth=1)
    ax.axvline(i-0.5, color='white', linewidth=1)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('r', fontsize=9, rotation=0, labelpad=8)
cbar.ax.tick_params(labelsize=8)

ax.set_title('(a) Correlation Matrix', fontsize=10, pad=8)

# Right: Overlaps
ax = axes[1]

pairs_short = ['T–G', 'T–R', 'G–R']
bars = ax.bar(pairs_short, overlaps, color=colors_grad, alpha=0.8, edgecolor='#37474f', linewidth=0.8)

ax.set_ylabel('Overlap (%)', fontsize=10)
ax.set_title('(b) Pairwise Overlaps', fontsize=10, pad=8)
ax.set_ylim(0, 18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{pct}%', ha='center', va='bottom', fontsize=9)

plt.savefig('visualizations/fig6_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: fig6_summary.png")
plt.close()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("✅ MINIMALIST FIGURES COMPLETE")
print("="*70)
print("\nGenerated files (300 DPI, minimal style):")
print("  fig1_multibias.png - Multi-bias comparison (3 panels)")
print("  fig2_threeway_venn.png - Overlap regions")
print("  fig3_correlation.png - Correlation heatmap")
print("  fig4_layer_dist.png - Layer distribution")
print("  fig5_pairwise.png - Pairwise overlaps")
print("  fig6_summary.png - Summary panel (2-panel)")
print("\nStyle: Single gradient, clean text, minimal design")
print("="*70)

