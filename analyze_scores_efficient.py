"""Memory-efficient attribution score analysis for code review

Analyzes scores by loading one file at a time and extracting key statistics.
"""
import sys
sys.path.append('.')

import torch
import numpy as np
import json
from pathlib import Path

def analyze_file_streaming(filepath, name):
    """Analyze single file with minimal memory"""
    print(f"\n{'='*70}")
    print(f"Analyzing: {name}")
    print(f"File: {filepath}")
    print(f"{'='*70}")

    # Load scores
    print("Loading scores...")
    scores = torch.load(filepath, map_location='cpu')

    print(f"✓ Loaded {len(scores)} layers")

    # Collect statistics without keeping everything in memory
    stats = {
        'name': name,
        'file': str(filepath),
        'n_layers': len(scores),
        'n_params': 0,
        'total_relevance': 0.0,
        'n_nan': 0,
        'n_inf': 0,
        'n_zero': 0,
        'n_negative': 0,
        'up_proj_layers': [],
        'up_proj_stats': {}
    }

    # Process each layer
    for layer_name, layer_scores in scores.items():
        n_params = layer_scores.numel()
        stats['n_params'] += n_params
        stats['total_relevance'] += layer_scores.sum().item()
        stats['n_nan'] += torch.isnan(layer_scores).sum().item()
        stats['n_inf'] += torch.isinf(layer_scores).sum().item()
        stats['n_zero'] += (layer_scores == 0).sum().item()
        stats['n_negative'] += (layer_scores < 0).sum().item()

        # Track up_proj layers specifically
        if 'up_proj' in layer_name:
            stats['up_proj_layers'].append(layer_name)

    print(f"\nBasic Statistics:")
    print(f"  Total parameters: {stats['n_params']:,}")
    print(f"  Total relevance: {stats['total_relevance']:.2e}")
    print(f"  Average relevance/param: {stats['total_relevance']/stats['n_params']:.2e}")

    print(f"\nNumerical Issues:")
    print(f"  NaN: {stats['n_nan']}")
    print(f"  Inf: {stats['n_inf']}")
    print(f"  Zero: {stats['n_zero']:,} ({100*stats['n_zero']/stats['n_params']:.2f}%)")
    print(f"  Negative: {stats['n_negative']:,} ({100*stats['n_negative']/stats['n_params']:.4f}%)")

    print(f"\nup_proj Layers: {len(stats['up_proj_layers'])}")

    # Analyze first up_proj layer in detail
    if stats['up_proj_layers']:
        first_up_proj = stats['up_proj_layers'][0]
        layer_scores = scores[first_up_proj]

        print(f"\nDetailed analysis of {first_up_proj}:")
        print(f"  Shape: {layer_scores.shape}")
        print(f"  Min: {layer_scores.min().item():.2e}")
        print(f"  Max: {layer_scores.max().item():.2e}")
        print(f"  Mean: {layer_scores.mean().item():.2e}")
        print(f"  Std: {layer_scores.std().item():.2e}")

        stats['up_proj_stats'] = {
            'shape': list(layer_scores.shape),
            'min': float(layer_scores.min().item()),
            'max': float(layer_scores.max().item()),
            'mean': float(layer_scores.mean().item()),
            'std': float(layer_scores.std().item())
        }

    # Clear from memory
    del scores
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return stats


def compare_files(stats_list):
    """Compare statistics across all files"""
    print(f"\n{'='*70}")
    print("Cross-File Comparison")
    print(f"{'='*70}")

    print(f"\nTotal Relevance Comparison:")
    for stat in stats_list:
        print(f"  {stat['name']:20s}: {stat['total_relevance']:.2e}")

    print(f"\nPer-Sample Relevance (assuming 128 C4, 93 toxic):")
    for stat in stats_list:
        if 'General' in stat['name']:
            per_sample = stat['total_relevance'] / 128
        else:
            per_sample = stat['total_relevance'] / 93
        print(f"  {stat['name']:20s}: {per_sample:.2e}")

    # Check for negative values (should be ZERO since we use abs())
    print(f"\nNegative Values Check (should all be 0):")
    for stat in stats_list:
        if stat['n_negative'] > 0:
            print(f"  ❌ {stat['name']}: {stat['n_negative']:,} negative values!")
        else:
            print(f"  ✓ {stat['name']}: 0 negative values")


def main():
    print("="*70)
    print("MEMORY-EFFICIENT ATTRIBUTION SCORE ANALYSIS")
    print("="*70)

    files = [
        ('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed0.pt', 'General Seed 0'),
        ('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed1.pt', 'General Seed 1'),
        ('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed2.pt', 'General Seed 2'),
        ('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_toxic.pt', 'Toxic'),
    ]

    stats_list = []

    for filepath, name in files:
        stat = analyze_file_streaming(filepath, name)
        stats_list.append(stat)

    # Compare files
    compare_files(stats_list)

    # Save report
    report = {
        'files': stats_list,
        'summary': {
            'all_negative_counts_zero': all(s['n_negative'] == 0 for s in stats_list),
            'all_nan_counts_zero': all(s['n_nan'] == 0 for s in stats_list),
            'all_inf_counts_zero': all(s['n_inf'] == 0 for s in stats_list),
        }
    }

    with open('code_review_stats.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Report saved to code_review_stats.json")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if report['summary']['all_negative_counts_zero']:
        print("✓ No negative values (abs() working correctly)")
    else:
        print("❌ WARNING: Found negative values!")

    if report['summary']['all_nan_counts_zero']:
        print("✓ No NaN values")
    else:
        print("⚠️ WARNING: Found NaN values!")

    if report['summary']['all_inf_counts_zero']:
        print("✓ No Inf values")
    else:
        print("⚠️ WARNING: Found Inf values!")


if __name__ == "__main__":
    main()
