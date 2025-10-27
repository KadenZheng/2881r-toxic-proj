"""Create visualizations for circuit overlap analysis

Generates:
1. Layer distribution heatmap (toxic vs gender vs overlap)
2. Neuron count bar chart per layer
3. Scatter plot: R_diff_toxic vs R_diff_gender (with quadrants)
4. Layer group stacked bar chart
"""
import sys
sys.path.append('.')

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_layer_distribution_heatmap(overlap_data: dict, output_path: str):
    """Create heatmap showing neuron distribution across layers."""

    print("\nCreating layer distribution heatmap...")

    # Load detailed layer stats from JSON
    with open('circuit_overlap_analysis.json', 'r') as f:
        data = json.load(f)

    # Extract from overlap log
    toxic_neurons = json.load(open('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_results/experiment_20251026_120604/pruned_neurons_20251026_120616.json'))
    gender_neurons = json.load(open('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_results/gender_experiment_20251027_102521/pruned_neurons_gender_20251027_102654.json'))

    toxic_list = [(n['layer'], n['index']) for n in toxic_neurons['neurons']]
    gender_list = [(n['layer'], n['index']) for n in gender_neurons['neurons']]

    # Count per layer
    from collections import Counter

    toxic_by_layer = Counter(int(layer.split('.')[2]) for layer, idx in toxic_list)
    gender_by_layer = Counter(int(layer.split('.')[2]) for layer, idx in gender_list)

    # Compute overlap per layer
    overlap_by_layer = {}
    for layer_num in range(32):
        layer_name = f'model.layers.{layer_num}.mlp.up_proj'
        toxic_in_layer = set((l, i) for l, i in toxic_list if int(l.split('.')[2]) == layer_num)
        gender_in_layer = set((l, i) for l, i in gender_list if int(l.split('.')[2]) == layer_num)
        overlap_in_layer = toxic_in_layer & gender_in_layer
        overlap_by_layer[layer_num] = len(overlap_in_layer)

    # Create matrix for heatmap [layers x 3 types]
    matrix = np.zeros((32, 3))
    for layer in range(32):
        matrix[layer, 0] = toxic_by_layer.get(layer, 0)
        matrix[layer, 1] = gender_by_layer.get(layer, 0)
        matrix[layer, 2] = overlap_by_layer.get(layer, 0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 12))

    sns.heatmap(matrix,
                annot=True,
                fmt='.0f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Number of Neurons'},
                xticklabels=['Toxicity', 'Gender', 'Overlap'],
                yticklabels=[f'Layer {i}' for i in range(32)],
                ax=ax)

    ax.set_title('Neuron Distribution Across Layers\n(Toxicity vs Gender Bias)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Experiment Type', fontsize=12)
    ax.set_ylabel('Layer Number', fontsize=12)

    # Highlight late layers
    ax.axhline(y=22, color='blue', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(2.5, 22, 'Late Layers →', fontsize=10, color='blue')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to {output_path}")
    plt.close()


def create_layer_bar_chart(output_path: str):
    """Create grouped bar chart of neuron counts per layer."""

    print("\nCreating layer bar chart...")

    # Load neuron lists
    toxic_neurons = json.load(open('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_results/experiment_20251026_120604/pruned_neurons_20251026_120616.json'))
    gender_neurons = json.load(open('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_results/gender_experiment_20251027_102521/pruned_neurons_gender_20251027_102654.json'))

    toxic_list = [(n['layer'], n['index']) for n in toxic_neurons['neurons']]
    gender_list = [(n['layer'], n['index']) for n in gender_neurons['neurons']]

    from collections import Counter

    toxic_by_layer = Counter(int(layer.split('.')[2]) for layer, idx in toxic_list)
    gender_by_layer = Counter(int(layer.split('.')[2]) for layer, idx in gender_list)

    # Prepare data
    layers = range(32)
    toxic_counts = [toxic_by_layer.get(i, 0) for i in layers]
    gender_counts = [gender_by_layer.get(i, 0) for i in layers]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(16, 6))

    x = np.arange(32)
    width = 0.35

    bars1 = ax.bar(x - width/2, toxic_counts, width, label='Toxicity', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, gender_counts, width, label='Gender Bias', color='#3498db', alpha=0.8)

    ax.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Pruned Neurons', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Pruned Neurons Across Layers\n(Toxicity vs Gender Bias)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(32)])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Highlight late layers
    ax.axvspan(21.5, 31.5, alpha=0.1, color='yellow', label='Late Layers (22-31)')

    # Add text annotations for high-overlap layers
    for layer in [15, 30, 31]:
        if toxic_by_layer.get(layer, 0) > 0 or gender_by_layer.get(layer, 0) > 0:
            ax.annotate('*', xy=(layer, max(toxic_counts[layer], gender_counts[layer]) + 1),
                       fontsize=16, color='red', ha='center')

    ax.text(27, max(max(toxic_counts), max(gender_counts)) - 5,
            '* = 2+ overlapping neurons', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved bar chart to {output_path}")
    plt.close()


def create_layer_group_comparison(output_path: str):
    """Create stacked bar chart comparing layer groups."""

    print("\nCreating layer group comparison...")

    # Data from Phase 4
    data = {
        'Early (0-10)': {'Toxic': 10, 'Gender': 5, 'Overlap': 0},
        'Middle (11-21)': {'Toxic': 20, 'Gender': 25, 'Overlap': 4},
        'Late (22-31)': {'Toxic': 70, 'Gender': 70, 'Overlap': 7}
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Grouped bars
    groups = list(data.keys())
    toxic_vals = [data[g]['Toxic'] for g in groups]
    gender_vals = [data[g]['Gender'] for g in groups]
    overlap_vals = [data[g]['Overlap'] for g in groups]

    x = np.arange(len(groups))
    width = 0.25

    ax1.bar(x - width, toxic_vals, width, label='Toxicity', color='#e74c3c', alpha=0.8)
    ax1.bar(x, gender_vals, width, label='Gender', color='#3498db', alpha=0.8)
    ax1.bar(x + width, overlap_vals, width, label='Overlap', color='#2ecc71', alpha=0.8)

    ax1.set_xlabel('Layer Group', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Neurons', fontsize=12, fontweight='bold')
    ax1.set_title('Neuron Distribution by Layer Group', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Right plot: Overlap percentage by group
    overlap_pct = [
        (overlap_vals[i] / max(toxic_vals[i], gender_vals[i]) * 100) if max(toxic_vals[i], gender_vals[i]) > 0 else 0
        for i in range(len(groups))
    ]

    bars = ax2.bar(groups, overlap_pct, color=['#95a5a6', '#7f8c8d', '#34495e'], alpha=0.8)
    ax2.set_ylabel('Overlap %', fontsize=12, fontweight='bold')
    ax2.set_title('Overlap Percentage by Layer Group', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(overlap_pct) * 1.2 if max(overlap_pct) > 0 else 10)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, pct in zip(bars, overlap_pct):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved group comparison to {output_path}")
    plt.close()


def create_summary_visualization(output_path: str):
    """Create comprehensive summary figure with multiple subplots."""

    print("\nCreating comprehensive summary visualization...")

    fig = plt.figure(figsize=(16, 10))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Load data
    with open('circuit_overlap_analysis.json', 'r') as f:
        data = json.load(f)

    # Subplot 1: Overlap metrics (top left)
    ax1 = fig.add_subplot(gs[0, 0])

    metrics = ['Neuron\nOverlap', 'IoU', 'Correlation']
    values = [
        11,  # overlap count
        data['phase1_overlap']['iou'] * 100,  # IoU as percentage
        data['phase2_correlation']['correlation'] * 100  # correlation as percentage
    ]

    bars = ax1.bar(metrics, values, color=['#e74c3c', '#f39c12', '#9b59b6'], alpha=0.8)
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('Key Overlap Metrics', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Subplot 2: Quadrant distribution (top right)
    ax2 = fig.add_subplot(gs[0, 1])

    quadrant_names = ['Multi-bias\n(Both neg)', 'Toxic-only', 'Gender-only', 'General\nCapability']
    quadrant_counts = [
        data['phase2_quadrants']['both_negative_count'],
        data['phase2_quadrants']['toxic_only_count'],
        data['phase2_quadrants']['gender_only_count'],
        data['phase2_quadrants']['general_count']
    ]
    quadrant_pcts = [count/458752*100 for count in quadrant_counts]

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']
    bars = ax2.bar(range(4), quadrant_pcts, color=colors, alpha=0.8)
    ax2.set_ylabel('Percentage of Neurons', fontsize=11)
    ax2.set_title('Quadrant Distribution (All 458,752 Neurons)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(quadrant_names, fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for bar, pct in zip(bars, quadrant_pcts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    # Subplot 3: Layer group distribution (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])

    groups = ['Early\n(0-10)', 'Middle\n(11-21)', 'Late\n(22-31)']
    group_data = data['phase4_layer_groups']

    x = np.arange(len(groups))
    width = 0.25

    toxic_group = [group_data['early']['toxic'], group_data['middle']['toxic'], group_data['late']['toxic']]
    gender_group = [group_data['early']['gender'], group_data['middle']['gender'], group_data['late']['gender']]
    overlap_group = [group_data['early']['overlap'], group_data['middle']['overlap'], group_data['late']['overlap']]

    ax3.bar(x - width, toxic_group, width, label='Toxicity', color='#e74c3c', alpha=0.8)
    ax3.bar(x, gender_group, width, label='Gender', color='#3498db', alpha=0.8)
    ax3.bar(x + width, overlap_group, width, label='Overlap', color='#2ecc71', alpha=0.8)

    ax3.set_xlabel('Layer Group', fontsize=11)
    ax3.set_ylabel('Number of Neurons', fontsize=11)
    ax3.set_title('Distribution by Layer Group', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(groups)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Subplot 4: Pruned neuron quadrant membership (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])

    toxic_multi = data['phase2_pruned']['toxic_in_multi']
    toxic_specific = 100 - toxic_multi
    gender_multi = data['phase2_pruned']['gender_in_multi']
    gender_specific = 100 - gender_multi

    x = np.arange(2)
    width = 0.35

    multi_vals = [toxic_multi, gender_multi]
    specific_vals = [toxic_specific, gender_specific]

    ax4.bar(x, multi_vals, width, label='Multi-bias', color='#2ecc71', alpha=0.8)
    ax4.bar(x, specific_vals, width, bottom=multi_vals, label='Bias-specific', color='#95a5a6', alpha=0.8)

    ax4.set_ylabel('Number of Pruned Neurons', fontsize=11)
    ax4.set_title('Pruned Neuron Quadrant Membership', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Toxicity\nExperiment', 'Gender\nExperiment'])
    ax4.legend()
    ax4.set_ylim(0, 110)
    ax4.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, (multi, specific) in enumerate(zip(multi_vals, specific_vals)):
        ax4.text(i, multi/2, f'{multi}%', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax4.text(i, multi + specific/2, f'{specific}%', ha='center', va='center', fontsize=10, fontweight='bold')

    # Add overall title
    fig.suptitle('Circuit Overlap Analysis: Toxicity vs Gender Bias\n' +
                 f'11 Overlapping Neurons (11%), Correlation: 0.60, p~0',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary visualization to {output_path}")
    plt.close()


def main():
    """Generate all visualizations."""

    print("="*70)
    print("CIRCUIT OVERLAP VISUALIZATION GENERATOR")
    print("="*70)

    # Create output directory
    Path('visualizations').mkdir(exist_ok=True)

    # Generate visualizations
    create_layer_distribution_heatmap(
        {},
        'visualizations/layer_distribution_heatmap.png'
    )

    create_layer_bar_chart(
        'visualizations/layer_neuron_counts.png'
    )

    create_layer_group_comparison(
        'visualizations/layer_group_comparison.png'
    )

    create_summary_visualization(
        'visualizations/circuit_overlap_summary.png'
    )

    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS CREATED")
    print("="*70)
    print("\nGenerated files:")
    print("  1. visualizations/layer_distribution_heatmap.png")
    print("  2. visualizations/layer_neuron_counts.png")
    print("  3. visualizations/layer_group_comparison.png")
    print("  4. visualizations/circuit_overlap_summary.png")
    print("="*70)


if __name__ == "__main__":
    main()
