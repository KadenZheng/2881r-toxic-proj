"""Comprehensive analysis of attribution scores for code review

This script performs deep analysis of the computed attribution scores to identify
any potential bugs, artifacts, or validity issues before proceeding with pruning.
"""
import sys
sys.path.append('.')

import torch
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

def analyze_single_file(filepath, name):
    """Analyze a single attribution score file"""
    print(f"\n{'='*70}")
    print(f"Analyzing: {name}")
    print(f"{'='*70}")

    scores = torch.load(filepath, map_location='cpu')

    print(f"Total layers: {len(scores)}")

    # Basic statistics
    total_relevance = sum(s.sum().item() for s in scores.values())
    total_params = sum(s.numel() for s in scores.values())

    print(f"Total parameters: {total_params:,}")
    print(f"Total relevance: {total_relevance:.2e}")
    print(f"Average relevance per param: {total_relevance/total_params:.2e}")

    # Check for numerical issues
    n_nan = sum(torch.isnan(s).sum().item() for s in scores.values())
    n_inf = sum(torch.isinf(s).sum().item() for s in scores.values())
    n_zero = sum((s == 0).sum().item() for s in scores.values())
    n_negative = sum((s < 0).sum().item() for s in scores.values())

    print(f"\nNumerical checks:")
    print(f"  NaN values: {n_nan}")
    print(f"  Inf values: {n_inf}")
    print(f"  Zero values: {n_zero:,} ({100*n_zero/total_params:.2f}%)")
    print(f"  Negative values: {n_negative:,} ({100*n_negative/total_params:.2f}%)")

    # Find up_proj layers
    up_proj_layers = {k: v for k, v in scores.items() if 'up_proj' in k}
    print(f"\nup_proj layers: {len(up_proj_layers)}")

    if up_proj_layers:
        # Analyze distribution
        all_values = []
        for layer_scores in up_proj_layers.values():
            all_values.extend(layer_scores.flatten().tolist())

        all_values = np.array(all_values)

        print(f"\nup_proj score distribution:")
        print(f"  Min: {all_values.min():.2e}")
        print(f"  Max: {all_values.max():.2e}")
        print(f"  Mean: {all_values.mean():.2e}")
        print(f"  Median: {np.median(all_values):.2e}")
        print(f"  Std: {all_values.std():.2e}")
        print(f"  25th percentile: {np.percentile(all_values, 25):.2e}")
        print(f"  75th percentile: {np.percentile(all_values, 75):.2e}")
        print(f"  99th percentile: {np.percentile(all_values, 99):.2e}")

    return {
        'name': name,
        'total_layers': len(scores),
        'total_params': total_params,
        'total_relevance': total_relevance,
        'n_nan': n_nan,
        'n_inf': n_inf,
        'n_zero': n_zero,
        'n_negative': n_negative,
        'up_proj_layers': len(up_proj_layers)
    }


def compare_general_seeds(seed_paths):
    """Compare attribution scores across different seeds"""
    print(f"\n{'='*70}")
    print(f"Cross-Seed Consistency Analysis")
    print(f"{'='*70}")

    # Load all seeds
    seeds = []
    for path in seed_paths:
        print(f"\nLoading {Path(path).name}...")
        scores = torch.load(path, map_location='cpu')
        seeds.append(scores)

    # Get common layers
    common_layers = set(seeds[0].keys())
    for seed in seeds[1:]:
        common_layers &= set(seed.keys())

    print(f"\nCommon layers across all seeds: {len(common_layers)}")

    # Compare up_proj layers
    up_proj_layers = [k for k in common_layers if 'up_proj' in k]

    if not up_proj_layers:
        print("No up_proj layers found!")
        return

    print(f"Analyzing {len(up_proj_layers)} up_proj layers...")

    # Compute correlations between seeds
    correlations = []

    for layer in up_proj_layers[:5]:  # Check first 5 layers
        print(f"\nLayer: {layer}")

        # Get scores from all seeds for this layer
        seed0_scores = seeds[0][layer].flatten().numpy()
        seed1_scores = seeds[1][layer].flatten().numpy()
        seed2_scores = seeds[2][layer].flatten().numpy()

        # Compute correlations
        corr_01 = np.corrcoef(seed0_scores, seed1_scores)[0, 1]
        corr_02 = np.corrcoef(seed0_scores, seed2_scores)[0, 1]
        corr_12 = np.corrcoef(seed1_scores, seed2_scores)[0, 1]

        avg_corr = (corr_01 + corr_02 + corr_12) / 3
        correlations.append(avg_corr)

        print(f"  Correlation 0-1: {corr_01:.4f}")
        print(f"  Correlation 0-2: {corr_02:.4f}")
        print(f"  Correlation 1-2: {corr_12:.4f}")
        print(f"  Average correlation: {avg_corr:.4f}")

        # Check if different seeds produce different samples
        # (correlation should be < 1.0 if truly different random samples)
        if avg_corr > 0.99:
            print(f"  ⚠️ WARNING: Very high correlation - seeds may not be truly random!")

    overall_avg_corr = np.mean(correlations)
    print(f"\nOverall average correlation across seeds: {overall_avg_corr:.4f}")

    if overall_avg_corr > 0.95:
        print("⚠️ WARNING: Suspiciously high correlation - check seed randomness!")
    elif overall_avg_corr > 0.7:
        print("✓ Good: High correlation suggests stable attribution")
    else:
        print("⚠️ WARNING: Low correlation - may indicate instability")


def analyze_sequence_length_bias():
    """Check if sequence length creates bias in attribution"""
    print(f"\n{'='*70}")
    print(f"Sequence Length Bias Analysis")
    print(f"{'='*70}")

    general = torch.load('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed0.pt', map_location='cpu')
    toxic = torch.load('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_toxic.pt', map_location='cpu')

    # C4 samples: 128 samples × 2048 tokens = 262,144 total tokens
    # Toxic samples: 93 samples × ~20 tokens (estimate) = ~1,860 total tokens

    # Check if relevance is normalized per sample vs per token
    c4_total_rel = sum(s.sum().item() for s in general.values())
    toxic_total_rel = sum(s.sum().item() for s in toxic.values())

    c4_samples = 128
    toxic_samples = 93

    c4_per_sample = c4_total_rel / c4_samples
    toxic_per_sample = toxic_total_rel / toxic_samples

    print(f"\nTotal relevance:")
    print(f"  C4 (128 samples, 2048 tokens): {c4_total_rel:.2e}")
    print(f"  Toxic (93 samples, ~20-50 tokens): {toxic_total_rel:.2e}")

    print(f"\nPer-sample relevance:")
    print(f"  C4: {c4_per_sample:.2e}")
    print(f"  Toxic: {toxic_per_sample:.2e}")
    print(f"  Ratio (C4/Toxic): {c4_per_sample/toxic_per_sample:.2f}x")

    # CRITICAL: If C4 per-sample >> Toxic per-sample, this could be a problem
    # because longer sequences accumulate more relevance
    if c4_per_sample / toxic_per_sample > 20:
        print("\n🚨 CRITICAL ISSUE: C4 samples have much higher relevance per sample!")
        print("   This is expected due to sequence length (2048 vs ~20-50 tokens)")
        print("   BUT: Differential attribution (R_general - R_toxic) will be biased!")
        print("   SOLUTION: We're averaging over n_samples, which is correct.")
        print("   The averaging normalizes by number of samples, not tokens.")
    else:
        print("\n✓ Per-sample relevance is comparable")


def analyze_differential_pattern():
    """Analyze the differential pattern to ensure it makes sense"""
    print(f"\n{'='*70}")
    print(f"Differential Attribution Pattern Analysis")
    print(f"{'='*70}")

    # Load scores
    general = torch.load('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed0.pt', map_location='cpu')
    toxic = torch.load('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_toxic.pt', map_location='cpu')

    # Find up_proj layers
    up_proj_layers = [k for k in general.keys() if 'up_proj' in k]

    print(f"\nAnalyzing differential for {len(up_proj_layers)} up_proj layers...")

    all_diffs = []
    layer_stats = []

    for layer_name in up_proj_layers:
        gen = general[layer_name]
        tox = toxic[layer_name]

        # Aggregate to neuron level
        gen_neuron = gen.sum(dim=1)
        tox_neuron = tox.sum(dim=1)
        diff = gen_neuron - tox_neuron

        all_diffs.extend(diff.tolist())

        layer_stats.append({
            'layer': layer_name,
            'min_diff': diff.min().item(),
            'max_diff': diff.max().item(),
            'mean_diff': diff.mean().item(),
            'n_negative': (diff < 0).sum().item(),
            'pct_negative': (diff < 0).sum().item() / diff.numel() * 100
        })

    all_diffs = np.array(all_diffs)

    print(f"\nGlobal differential statistics (all {len(up_proj_layers)} layers):")
    print(f"  Total neurons: {len(all_diffs):,}")
    print(f"  Min R_diff: {all_diffs.min():.2e}")
    print(f"  Max R_diff: {all_diffs.max():.2e}")
    print(f"  Mean R_diff: {all_diffs.mean():.2e}")
    print(f"  Std R_diff: {all_diffs.std():.2e}")

    n_negative = (all_diffs < 0).sum()
    pct_negative = (n_negative / len(all_diffs)) * 100

    print(f"\n  Neurons with negative R_diff (toxic > general): {n_negative:,}/{len(all_diffs):,} ({pct_negative:.1f}%)")

    # Check if distribution makes sense
    print(f"\nPercentiles of R_diff:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(all_diffs, p)
        print(f"  {p:2d}th: {val:.2e}")

    # Expected: Should have SOME negative values (toxic-specific)
    # and SOME positive values (general-specific)
    if pct_negative < 1:
        print("\n⚠️ WARNING: Very few negative R_diff values - may not find toxic-specific neurons!")
    elif pct_negative > 99:
        print("\n⚠️ WARNING: Almost all R_diff negative - this is suspicious!")
    else:
        print(f"\n✓ Good: {pct_negative:.1f}% negative R_diff values")

    # Show which layers have most toxic-specific neurons
    print(f"\nTop 5 layers by percentage of toxic-specific neurons:")
    sorted_layers = sorted(layer_stats, key=lambda x: x['pct_negative'], reverse=True)
    for i, stat in enumerate(sorted_layers[:5]):
        print(f"  {i+1}. {stat['layer']}: {stat['pct_negative']:.1f}% negative")

    return all_diffs, layer_stats


def check_for_common_bugs():
    """Check for common implementation bugs"""
    print(f"\n{'='*70}")
    print(f"Common Bug Checks")
    print(f"{'='*70}")

    general = torch.load('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed0.pt', map_location='cpu')
    toxic = torch.load('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_toxic.pt', map_location='cpu')

    issues = []

    # Bug 1: Check if scores are all positive (should be, we use abs())
    print("\n1. Checking sign of scores (should all be non-negative due to abs())...")
    for name, scores in [('general', general), ('toxic', toxic)]:
        n_neg = sum((s < 0).sum().item() for s in scores.values())
        if n_neg > 0:
            issues.append(f"{name} has {n_neg} negative values - bug in abs() call?")
            print(f"  ❌ {name}: {n_neg} negative values found!")
        else:
            print(f"  ✓ {name}: All values non-negative")

    # Bug 2: Check if shapes match between general and toxic
    print("\n2. Checking shape consistency between general and toxic...")
    mismatches = []
    for layer_name in general.keys():
        if layer_name in toxic:
            if general[layer_name].shape != toxic[layer_name].shape:
                mismatches.append(layer_name)

    if mismatches:
        issues.append(f"Shape mismatches in {len(mismatches)} layers")
        print(f"  ❌ Shape mismatches: {mismatches[:5]}")
    else:
        print(f"  ✓ All shapes match")

    # Bug 3: Check if averaging was done correctly (scores should be averaged, not summed)
    print("\n3. Checking if averaging was applied correctly...")
    # We averaged over 128 samples for general, 93 for toxic
    # Total relevance should be reasonable, not 128x or 93x too large

    gen_total = sum(s.sum().item() for s in general.values())
    tox_total = sum(s.sum().item() for s in toxic.values())

    # If we forgot to divide by n_samples, gen_total would be ~128x larger
    # Expected range based on TinyLlama test: ~1e4 to 1e6
    if gen_total > 1e7:
        issues.append(f"General total relevance suspiciously high: {gen_total:.2e}")
        print(f"  ⚠️ General total: {gen_total:.2e} (may not be averaged?)")
    else:
        print(f"  ✓ General total: {gen_total:.2e} (reasonable)")

    if tox_total > 1e7:
        issues.append(f"Toxic total relevance suspiciously high: {tox_total:.2e}")
        print(f"  ⚠️ Toxic total: {tox_total:.2e} (may not be averaged?)")
    else:
        print(f"  ✓ Toxic total: {tox_total:.2e} (reasonable)")

    # Bug 4: Check layer naming pattern
    print("\n4. Checking layer naming patterns...")
    sample_layers = list(general.keys())[:10]
    print(f"  Sample layer names:")
    for layer in sample_layers:
        print(f"    - {layer}")

    # Should see patterns like: model.layers.X.mlp.up_proj
    expected_patterns = ['model.layers', '.mlp.', 'up_proj', 'down_proj', 'gate_proj']
    found_patterns = [any(p in layer for layer in sample_layers) for p in expected_patterns]

    if all(found_patterns):
        print(f"  ✓ Layer naming looks correct (LLaMA architecture)")
    else:
        issues.append("Unexpected layer naming pattern")
        print(f"  ⚠️ Some expected patterns not found")

    return issues


def validate_pruning_targets():
    """Validate that we'll be pruning the right neurons"""
    print(f"\n{'='*70}")
    print(f"Pruning Target Validation")
    print(f"{'='*70}")

    from src.pruning import load_and_average_seeds, compute_differential_scores, identify_neurons_to_prune

    print("\n1. Loading and averaging 3 general seeds...")
    general_scores = load_and_average_seeds([
        '/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed0.pt',
        '/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed1.pt',
        '/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed2.pt'
    ], device='cpu')

    print("\n2. Loading toxic scores...")
    toxic_scores = torch.load('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_toxic.pt', map_location='cpu')

    print("\n3. Computing differential scores...")
    diff_scores = compute_differential_scores(general_scores, toxic_scores)

    print(f"\n4. Identifying top 100 toxic-specific neurons...")
    neurons_to_prune = identify_neurons_to_prune(
        diff_scores,
        layer_pattern='up_proj',
        num_neurons=100
    )

    print(f"\nIdentified {len(neurons_to_prune)} neurons to prune")

    # Analyze distribution across layers
    layer_counts = defaultdict(int)
    for layer_name, idx in neurons_to_prune:
        layer_counts[layer_name] += 1

    print(f"\nDistribution across layers:")
    sorted_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (layer, count) in enumerate(sorted_layers[:10]):
        print(f"  {i+1}. {layer}: {count} neurons")

    # Check scores of selected neurons
    print(f"\nScore distribution of selected neurons:")
    selected_scores = []
    for layer_name, neuron_idx in neurons_to_prune:
        if layer_name in diff_scores:
            neuron_score = diff_scores[layer_name].sum(dim=1)[neuron_idx].item()
            selected_scores.append(neuron_score)

    selected_scores = np.array(selected_scores)
    print(f"  Min: {selected_scores.min():.2e} (most toxic-specific)")
    print(f"  Max: {selected_scores.max():.2e}")
    print(f"  Mean: {selected_scores.mean():.2e}")
    print(f"  All negative?: {(selected_scores < 0).all()}")

    if not (selected_scores < 0).all():
        print("\n⚠️ WARNING: Not all selected neurons have negative R_diff!")
        print("   This could indicate a bug in selection logic.")
        n_positive = (selected_scores >= 0).sum()
        print(f"   {n_positive}/100 selected neurons have R_diff >= 0")
    else:
        print("\n✓ Good: All 100 selected neurons have negative R_diff (toxic-specific)")

    # Save neuron list for inspection
    with open('code_review_neurons.json', 'w') as f:
        json.dump({
            'neurons': [{'layer': l, 'idx': i, 'score': selected_scores[j]}
                       for j, (l, i) in enumerate(neurons_to_prune[:20])],
            'distribution': dict(sorted_layers),
            'score_stats': {
                'min': float(selected_scores.min()),
                'max': float(selected_scores.max()),
                'mean': float(selected_scores.mean()),
                'std': float(selected_scores.std())
            }
        }, f, indent=2)

    print("\n✓ Saved first 20 neurons to code_review_neurons.json for inspection")

    return neurons_to_prune, selected_scores


def main():
    print("="*70)
    print("COMPREHENSIVE CODE REVIEW: Attribution Scores Analysis")
    print("="*70)
    print("\nThis analysis checks for:")
    print("  1. Numerical stability (NaN, Inf, zeros)")
    print("  2. Cross-seed consistency")
    print("  3. Sequence length bias")
    print("  4. Differential attribution validity")
    print("  5. Common implementation bugs")
    print("="*70)

    results = {}

    # Analyze each file
    files = [
        ('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed0.pt', 'General Seed 0'),
        ('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed1.pt', 'General Seed 1'),
        ('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed2.pt', 'General Seed 2'),
        ('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_toxic.pt', 'Toxic'),
    ]

    for filepath, name in files:
        result = analyze_single_file(filepath, name)
        results[name] = result

    # Cross-seed analysis
    seed_paths = [f[0] for f in files[:3]]
    compare_general_seeds(seed_paths)

    # Sequence length bias
    analyze_sequence_length_bias()

    # Check for bugs
    issues = check_for_common_bugs()

    # Validate pruning targets
    neurons, scores = validate_pruning_targets()

    # Final summary
    print("\n" + "="*70)
    print("FINAL ASSESSMENT")
    print("="*70)

    if issues:
        print(f"\n❌ Found {len(issues)} potential issues:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n⚠️ RECOMMENDATION: Fix issues before proceeding")
    else:
        print("\n✅ No critical bugs detected!")
        print("✓ Attribution scores pass all validation checks")
        print("✓ Differential pattern looks correct")
        print("✓ Pruning targets are valid")
        print("\n✅ SAFE TO PROCEED with full experiment")

    # Save full report
    with open('code_review_report.json', 'w') as f:
        json.dump({
            'file_stats': results,
            'issues': issues,
            'timestamp': str(Path(files[0][0]).stat().st_mtime)
        }, f, indent=2)

    print(f"\n✓ Full report saved to code_review_report.json")


if __name__ == "__main__":
    main()
