"""Analyze circuit overlap between toxicity and gender bias experiments

This script investigates whether toxic and gender-biased behaviors share
common neural circuits or are processed independently.

Research Question: Are there "universal bias neurons" that control multiple
harmful behaviors?

Phases:
1. Simple Overlap Analysis - Exact neuron matches, statistical significance
2. Differential Correlation - Correlation of R_diff scores across all neurons
4. Layer Distribution - Per-layer overlap patterns
"""
import sys
sys.path.append('.')

import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set

from src.pruning import load_and_average_seeds, compute_differential_scores, aggregate_to_neuron_level


# ============================================================================
# PHASE 1: SIMPLE OVERLAP ANALYSIS
# ============================================================================

def load_neuron_sets(
    toxic_path: str,
    gender_path: str
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Load pruned neuron sets from both experiments.

    Args:
        toxic_path: Path to toxicity experiment neuron JSON
        gender_path: Path to gender bias experiment neuron JSON

    Returns:
        Tuple of (toxic_neurons, gender_neurons) as lists of (layer, index) tuples
    """
    print("Loading pruned neuron sets...")

    # Load toxicity neurons
    with open(toxic_path, 'r') as f:
        toxic_data = json.load(f)

    toxic_neurons = [(n['layer'], n['index']) for n in toxic_data['neurons']]

    # Load gender neurons
    with open(gender_path, 'r') as f:
        gender_data = json.load(f)

    gender_neurons = [(n['layer'], n['index']) for n in gender_data['neurons']]

    print(f"✓ Loaded toxicity neurons: {len(toxic_neurons)}")
    print(f"✓ Loaded gender neurons: {len(gender_neurons)}")

    return toxic_neurons, gender_neurons


def compute_simple_overlap(
    toxic_neurons: List[Tuple[str, int]],
    gender_neurons: List[Tuple[str, int]]
) -> Dict:
    """
    Compute basic overlap statistics between neuron sets.

    Args:
        toxic_neurons: List of (layer, index) tuples from toxicity experiment
        gender_neurons: List of (layer, index) tuples from gender experiment

    Returns:
        Dictionary with overlap statistics
    """
    print("\nComputing overlap statistics...")

    # Convert to sets for intersection/union operations
    toxic_set = set(toxic_neurons)
    gender_set = set(gender_neurons)

    # Compute overlap
    overlap = toxic_set & gender_set
    union = toxic_set | gender_set

    # IoU (Jaccard similarity)
    iou = len(overlap) / len(union) if len(union) > 0 else 0

    # Overlap as percentage of each set
    overlap_pct_toxic = len(overlap) / len(toxic_set) * 100 if len(toxic_set) > 0 else 0
    overlap_pct_gender = len(overlap) / len(gender_set) * 100 if len(gender_set) > 0 else 0

    stats = {
        'toxic_total': len(toxic_set),
        'gender_total': len(gender_set),
        'overlap_count': len(overlap),
        'union_count': len(union),
        'iou': iou,
        'overlap_pct_toxic': overlap_pct_toxic,
        'overlap_pct_gender': overlap_pct_gender,
        'overlapping_neurons': sorted(list(overlap))
    }

    print(f"✓ Overlap computed: {len(overlap)} neurons")

    return stats


def hypergeometric_significance_test(
    overlap_count: int,
    toxic_size: int,
    gender_size: int,
    total_neurons: int = 458752
) -> Dict:
    """
    Test statistical significance of overlap using hypergeometric distribution.

    Args:
        overlap_count: Number of overlapping neurons
        toxic_size: Size of toxic neuron set
        gender_size: Size of gender neuron set
        total_neurons: Total up_proj neurons in model (32 × 14,336)

    Returns:
        Dictionary with significance statistics
    """
    print("\nComputing statistical significance...")

    # Random expectation
    expected_overlap = (toxic_size * gender_size) / total_neurons

    # Hypergeometric test
    try:
        from scipy.stats import hypergeom

        # P(overlap >= k | random selection)
        p_value = 1 - hypergeom.cdf(overlap_count - 1, total_neurons, toxic_size, gender_size)

    except ImportError:
        print("  ⚠️ scipy not available, skipping p-value calculation")
        p_value = None

    stats = {
        'expected_overlap': expected_overlap,
        'observed_overlap': overlap_count,
        'enrichment': overlap_count / expected_overlap if expected_overlap > 0 else float('inf'),
        'p_value': p_value
    }

    print(f"✓ Expected (random): {expected_overlap:.2f} neurons")
    print(f"✓ Observed: {overlap_count} neurons")

    if p_value is not None:
        print(f"✓ P-value: {p_value:.2e}")

    return stats


def generate_phase1_report(overlap_stats: Dict, significance_stats: Dict) -> str:
    """Generate Phase 1 report."""

    report = []
    report.append("="*70)
    report.append("PHASE 1: NEURON SET OVERLAP ANALYSIS")
    report.append("="*70)

    report.append("\nNeuron Sets:")
    report.append(f"  Toxicity: {overlap_stats['toxic_total']} neurons")
    report.append(f"  Gender: {overlap_stats['gender_total']} neurons")

    report.append("\nOverlap:")
    report.append(f"  Exact matches: {overlap_stats['overlap_count']} neurons")
    report.append(f"  Overlap % (of toxic set): {overlap_stats['overlap_pct_toxic']:.1f}%")
    report.append(f"  Overlap % (of gender set): {overlap_stats['overlap_pct_gender']:.1f}%")
    report.append(f"  IoU (Jaccard): {overlap_stats['iou']:.4f}")

    report.append("\nStatistical Significance:")
    report.append(f"  Random expectation: {significance_stats['expected_overlap']:.3f} neurons")
    report.append(f"  Observed: {significance_stats['observed_overlap']} neurons")
    report.append(f"  Enrichment: {significance_stats['enrichment']:.1f}×")

    if significance_stats['p_value'] is not None:
        report.append(f"  P-value: {significance_stats['p_value']:.2e}")

        if significance_stats['p_value'] < 0.01:
            report.append("  ✅ SIGNIFICANT (p < 0.01)")
        elif significance_stats['p_value'] < 0.05:
            report.append("  ✓ Marginally significant (p < 0.05)")
        else:
            report.append("  ⚠️ Not significant (p >= 0.05)")

    if overlap_stats['overlap_count'] > 0:
        report.append("\nOverlapping Neurons:")
        for layer, idx in overlap_stats['overlapping_neurons'][:20]:
            layer_num = int(layer.split('.')[2])
            report.append(f"  Layer {layer_num}: neuron {idx}")

        if len(overlap_stats['overlapping_neurons']) > 20:
            report.append(f"  ... and {len(overlap_stats['overlapping_neurons'])-20} more")

    return "\n".join(report)


# ============================================================================
# PHASE 2: DIFFERENTIAL CORRELATION ANALYSIS
# ============================================================================

def load_attributions_sequential(
    general_paths: List[str],
    toxic_path: str,
    stereotype_path: str
) -> Tuple[Dict, Dict]:
    """
    Load attribution scores and compute differentials sequentially to avoid OOM.

    Args:
        general_paths: Paths to 3 general seed files
        toxic_path: Path to toxic attribution file
        stereotype_path: Path to stereotype attribution file

    Returns:
        Tuple of (toxic_diff, gender_diff) dictionaries
    """
    print("\nLoading attribution scores (sequential to avoid OOM)...")

    # Step 1: Load and average general scores
    print("  1. Loading and averaging general scores...")
    general = load_and_average_seeds(general_paths, device='cpu')
    print(f"     ✓ General averaged: {len(general)} layers")

    # Step 2: Load toxic and compute differential
    print("  2. Loading toxic scores...")
    toxic = torch.load(toxic_path, map_location='cpu')
    print(f"     ✓ Toxic loaded: {len(toxic)} layers")

    print("  3. Computing toxic differential...")
    toxic_diff = compute_differential_scores(general, toxic)
    print(f"     ✓ Toxic differential: {len(toxic_diff)} layers")

    # Clear toxic from memory
    del toxic
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 3: Load stereotype and compute differential
    print("  4. Loading stereotype scores...")
    stereotype = torch.load(stereotype_path, map_location='cpu')
    print(f"     ✓ Stereotype loaded: {len(stereotype)} layers")

    print("  5. Computing gender differential...")
    gender_diff = compute_differential_scores(general, stereotype)
    print(f"     ✓ Gender differential: {len(gender_diff)} layers")

    # Clear stereotype from memory
    del stereotype
    del general
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return toxic_diff, gender_diff


def extract_all_neuron_diffs(
    toxic_diff: Dict,
    gender_diff: Dict
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int]]]:
    """
    Extract neuron-level differential scores for all up_proj neurons.

    Args:
        toxic_diff: Toxic differential scores
        gender_diff: Gender differential scores

    Returns:
        Tuple of (toxic_neuron_scores, gender_neuron_scores, neuron_ids)
        Arrays have length 458,752 (32 layers × 14,336 neurons)
    """
    print("\nExtracting neuron-level differential scores...")

    # Get up_proj layers
    up_proj_layers = sorted([k for k in toxic_diff.keys() if 'up_proj' in k])

    print(f"  Found {len(up_proj_layers)} up_proj layers")

    all_toxic_scores = []
    all_gender_scores = []
    neuron_ids = []

    for layer in up_proj_layers:
        # Aggregate to neuron level
        toxic_neurons = aggregate_to_neuron_level(toxic_diff[layer])
        gender_neurons = aggregate_to_neuron_level(gender_diff[layer])

        # Collect scores
        for idx in range(len(toxic_neurons)):
            all_toxic_scores.append(toxic_neurons[idx].item())
            all_gender_scores.append(gender_neurons[idx].item())
            neuron_ids.append((layer, idx))

    all_toxic_scores = np.array(all_toxic_scores)
    all_gender_scores = np.array(all_gender_scores)

    print(f"✓ Extracted {len(neuron_ids):,} neuron differential scores")

    return all_toxic_scores, all_gender_scores, neuron_ids


def correlation_analysis(
    toxic_scores: np.ndarray,
    gender_scores: np.ndarray
) -> Dict:
    """
    Compute correlation between toxic and gender differential scores.

    Args:
        toxic_scores: R_diff_toxic for all neurons
        gender_scores: R_diff_gender for all neurons

    Returns:
        Dictionary with correlation statistics
    """
    print("\nComputing correlation analysis...")

    # Pearson correlation
    correlation = np.corrcoef(toxic_scores, gender_scores)[0, 1]

    print(f"✓ Pearson correlation: {correlation:.4f}")

    # Interpretation
    if abs(correlation) > 0.5:
        interpretation = "HIGH - Shared bias mechanism"
    elif abs(correlation) > 0.3:
        interpretation = "MODERATE - Partially related"
    else:
        interpretation = "LOW - Independent circuits"

    stats = {
        'correlation': correlation,
        'interpretation': interpretation
    }

    return stats


def quadrant_classification(
    toxic_scores: np.ndarray,
    gender_scores: np.ndarray,
    neuron_ids: List[Tuple[str, int]]
) -> Dict:
    """
    Classify all neurons into quadrants based on differential scores.

    Quadrants:
    - Both negative: Multi-bias neurons (toxic AND gender-biased)
    - Toxic negative, gender positive: Toxic-specific
    - Toxic positive, gender negative: Gender-specific
    - Both positive: General-capability

    Args:
        toxic_scores: R_diff_toxic for all neurons
        gender_scores: R_diff_gender for all neurons
        neuron_ids: Neuron identifiers (layer, index)

    Returns:
        Dictionary with quadrant classifications
    """
    print("\nClassifying neurons into quadrants...")

    # Classify each neuron
    both_negative = []
    toxic_only = []
    gender_only = []
    general_capability = []

    for i, (nid, toxic, gender) in enumerate(zip(neuron_ids, toxic_scores, gender_scores)):
        if toxic < 0 and gender < 0:
            both_negative.append((nid, toxic, gender))
        elif toxic < 0 and gender >= 0:
            toxic_only.append((nid, toxic, gender))
        elif toxic >= 0 and gender < 0:
            gender_only.append((nid, toxic, gender))
        else:  # both >= 0
            general_capability.append((nid, toxic, gender))

    total = len(neuron_ids)

    print(f"✓ Classified {total:,} neurons:")
    print(f"  Both negative (multi-bias): {len(both_negative):,} ({100*len(both_negative)/total:.1f}%)")
    print(f"  Toxic-specific: {len(toxic_only):,} ({100*len(toxic_only)/total:.1f}%)")
    print(f"  Gender-specific: {len(gender_only):,} ({100*len(gender_only)/total:.1f}%)")
    print(f"  General-capability: {len(general_capability):,} ({100*len(general_capability)/total:.1f}%)")

    quadrants = {
        'both_negative': both_negative,
        'toxic_only': toxic_only,
        'gender_only': gender_only,
        'general_capability': general_capability,
        'total': total
    }

    return quadrants


def analyze_pruned_quadrants(
    toxic_neurons: List[Tuple[str, int]],
    gender_neurons: List[Tuple[str, int]],
    quadrants: Dict
) -> Dict:
    """
    Analyze which quadrants the pruned neurons fell into.

    Critical question: Did we prune multi-bias neurons or bias-specific neurons?

    Args:
        toxic_neurons: Pruned neurons from toxicity experiment
        gender_neurons: Pruned neurons from gender experiment
        quadrants: Quadrant classification from quadrant_classification()

    Returns:
        Dictionary with pruned neuron quadrant membership
    """
    print("\nAnalyzing pruned neuron quadrants...")

    toxic_set = set(toxic_neurons)
    gender_set = set(gender_neurons)

    # Get neuron IDs from each quadrant
    both_neg_ids = set(nid for nid, _, _ in quadrants['both_negative'])
    toxic_only_ids = set(nid for nid, _, _ in quadrants['toxic_only'])
    gender_only_ids = set(nid for nid, _, _ in quadrants['gender_only'])

    # Check which quadrant toxic-pruned neurons are in
    toxic_in_multi = toxic_set & both_neg_ids
    toxic_in_toxic_only = toxic_set & toxic_only_ids
    toxic_in_gender_only = toxic_set & gender_only_ids

    # Check which quadrant gender-pruned neurons are in
    gender_in_multi = gender_set & both_neg_ids
    gender_in_gender_only = gender_set & gender_only_ids
    gender_in_toxic_only = gender_set & toxic_only_ids

    print(f"✓ Toxic-pruned neurons by quadrant:")
    print(f"    Multi-bias: {len(toxic_in_multi)}/100 ({100*len(toxic_in_multi)/100:.1f}%)")
    print(f"    Toxic-specific: {len(toxic_in_toxic_only)}/100")
    print(f"    Gender-specific: {len(toxic_in_gender_only)}/100 (unexpected!)")

    print(f"\n✓ Gender-pruned neurons by quadrant:")
    print(f"    Multi-bias: {len(gender_in_multi)}/100 ({100*len(gender_in_multi)/100:.1f}%)")
    print(f"    Gender-specific: {len(gender_in_gender_only)}/100")
    print(f"    Toxic-specific: {len(gender_in_toxic_only)}/100 (unexpected!)")

    stats = {
        'toxic_in_multi': len(toxic_in_multi),
        'toxic_in_toxic_only': len(toxic_in_toxic_only),
        'gender_in_multi': len(gender_in_multi),
        'gender_in_gender_only': len(gender_in_gender_only)
    }

    return stats


def generate_phase2_report(
    correlation_stats: Dict,
    quadrants: Dict,
    pruned_stats: Dict
) -> str:
    """Generate Phase 2 report."""

    report = []
    report.append("\n" + "="*70)
    report.append("PHASE 2: DIFFERENTIAL CORRELATION ANALYSIS")
    report.append("="*70)

    report.append("\nCorrelation Analysis:")
    report.append(f"  Pearson correlation (R_diff_toxic vs R_diff_gender): {correlation_stats['correlation']:.4f}")
    report.append(f"  Interpretation: {correlation_stats['interpretation']}")

    report.append("\nQuadrant Distribution (all 458,752 neurons):")
    total = quadrants['total']
    report.append(f"  Both negative (multi-bias): {len(quadrants['both_negative']):,} ({100*len(quadrants['both_negative'])/total:.1f}%)")
    report.append(f"  Toxic-specific: {len(quadrants['toxic_only']):,} ({100*len(quadrants['toxic_only'])/total:.1f}%)")
    report.append(f"  Gender-specific: {len(quadrants['gender_only']):,} ({100*len(quadrants['gender_only'])/total:.1f}%)")
    report.append(f"  General-capability: {len(quadrants['general_capability']):,} ({100*len(quadrants['general_capability'])/total:.1f}%)")

    report.append("\nPruned Neurons by Quadrant:")
    report.append(f"  Toxic-pruned in multi-bias: {pruned_stats['toxic_in_multi']}/100")
    report.append(f"  Gender-pruned in multi-bias: {pruned_stats['gender_in_multi']}/100")

    # Top multi-bias neurons
    both_neg_sorted = sorted(quadrants['both_negative'], key=lambda x: x[1] + x[2])  # Most negative in both
    report.append("\nTop 10 Multi-Bias Neurons (most negative in both dimensions):")
    for rank, (nid, toxic, gender) in enumerate(both_neg_sorted[:10], 1):
        layer_num = int(nid[0].split('.')[2])
        report.append(f"  {rank}. Layer {layer_num}, neuron {nid[1]}: toxic={toxic:.3f}, gender={gender:.3f}")

    return "\n".join(report)


# ============================================================================
# PHASE 4: LAYER DISTRIBUTION ANALYSIS
# ============================================================================

def layer_distribution_analysis(
    toxic_neurons: List[Tuple[str, int]],
    gender_neurons: List[Tuple[str, int]]
) -> Dict:
    """
    Analyze distribution of pruned neurons across layers.

    Args:
        toxic_neurons: Pruned neurons from toxicity
        gender_neurons: Pruned neurons from gender

    Returns:
        Dictionary with per-layer statistics
    """
    print("\nAnalyzing layer distribution...")

    # Count neurons per layer
    toxic_by_layer = Counter(layer for layer, idx in toxic_neurons)
    gender_by_layer = Counter(layer for layer, idx in gender_neurons)

    # All layers that have any pruned neurons
    all_layers = set(list(toxic_by_layer.keys()) + list(gender_by_layer.keys()))

    layer_stats = {}
    for layer_name in sorted(all_layers):
        layer_num = int(layer_name.split('.')[2])

        # Get neuron sets for this layer
        toxic_in_layer = set((layer, idx) for layer, idx in toxic_neurons if layer == layer_name)
        gender_in_layer = set((layer, idx) for layer, idx in gender_neurons if layer == layer_name)
        overlap_in_layer = toxic_in_layer & gender_in_layer

        layer_stats[layer_num] = {
            'toxic_count': len(toxic_in_layer),
            'gender_count': len(gender_in_layer),
            'overlap_count': len(overlap_in_layer),
            'overlap_neurons': list(overlap_in_layer)
        }

    print(f"✓ Analyzed {len(layer_stats)} layers with pruned neurons")

    return layer_stats


def compute_overlap_by_layer(layer_stats: Dict) -> Dict:
    """Compute overlap statistics by layer group."""

    early_toxic = sum(s['toxic_count'] for layer, s in layer_stats.items() if layer < 11)
    early_gender = sum(s['gender_count'] for layer, s in layer_stats.items() if layer < 11)
    early_overlap = sum(s['overlap_count'] for layer, s in layer_stats.items() if layer < 11)

    middle_toxic = sum(s['toxic_count'] for layer, s in layer_stats.items() if 11 <= layer < 22)
    middle_gender = sum(s['gender_count'] for layer, s in layer_stats.items() if 11 <= layer < 22)
    middle_overlap = sum(s['overlap_count'] for layer, s in layer_stats.items() if 11 <= layer < 22)

    late_toxic = sum(s['toxic_count'] for layer, s in layer_stats.items() if layer >= 22)
    late_gender = sum(s['gender_count'] for layer, s in layer_stats.items() if layer >= 22)
    late_overlap = sum(s['overlap_count'] for layer, s in layer_stats.items() if layer >= 22)

    groups = {
        'early': {'toxic': early_toxic, 'gender': early_gender, 'overlap': early_overlap},
        'middle': {'toxic': middle_toxic, 'gender': middle_gender, 'overlap': middle_overlap},
        'late': {'toxic': late_toxic, 'gender': late_gender, 'overlap': late_overlap}
    }

    return groups


def identify_hotspot_layers(layer_stats: Dict, top_n: int = 5) -> List[Tuple[int, Dict]]:
    """Identify layers with most overlap."""

    sorted_layers = sorted(
        layer_stats.items(),
        key=lambda x: x[1]['overlap_count'],
        reverse=True
    )

    return sorted_layers[:top_n]


def generate_phase4_report(
    layer_stats: Dict,
    layer_groups: Dict,
    hotspots: List[Tuple[int, Dict]]
) -> str:
    """Generate Phase 4 report."""

    report = []
    report.append("\n" + "="*70)
    report.append("PHASE 4: LAYER DISTRIBUTION ANALYSIS")
    report.append("="*70)

    report.append("\nLayer Group Concentrations:")
    for group_name in ['early', 'middle', 'late']:
        group = layer_groups[group_name]
        range_str = {'early': '0-10', 'middle': '11-21', 'late': '22-31'}[group_name]

        report.append(f"\n  {group_name.capitalize()} layers ({range_str}):")
        report.append(f"    Toxic: {group['toxic']} neurons")
        report.append(f"    Gender: {group['gender']} neurons")
        report.append(f"    Overlap: {group['overlap']} neurons")

    report.append("\nHotspot Layers (most overlap):")
    if hotspots[0][1]['overlap_count'] > 0:
        for rank, (layer_num, stats) in enumerate(hotspots, 1):
            if stats['overlap_count'] > 0:
                report.append(f"  {rank}. Layer {layer_num}: {stats['overlap_count']} overlaps "
                            f"({stats['toxic_count']} toxic, {stats['gender_count']} gender)")
    else:
        report.append("  No layers with multiple overlaps")

    report.append("\nComplete Layer-by-Layer Distribution:")
    for layer_num in sorted(layer_stats.keys()):
        stats = layer_stats[layer_num]
        if stats['toxic_count'] > 0 or stats['gender_count'] > 0:
            report.append(f"  Layer {layer_num:2d}: {stats['toxic_count']:2d} toxic, "
                        f"{stats['gender_count']:2d} gender, {stats['overlap_count']:2d} overlap")

    return "\n".join(report)


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    """Run complete circuit overlap analysis."""

    print("="*70)
    print("CIRCUIT OVERLAP ANALYSIS: Toxicity vs Gender Bias")
    print("="*70)
    print("\nResearch Question: Are there universal bias neurons?")
    print("="*70)

    # File paths
    toxic_neurons_path = '/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_results/experiment_20251026_120604/pruned_neurons_20251026_120616.json'
    gender_neurons_path = '/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_results/gender_experiment_20251027_102521/pruned_neurons_gender_20251027_102654.json'

    general_paths = [
        '/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed0.pt',
        '/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed1.pt',
        '/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed2.pt'
    ]
    toxic_path = '/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_toxic.pt'
    stereotype_path = '/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_stereotype.pt'

    # ========================================================================
    # PHASE 1: Simple Overlap
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 1: SIMPLE OVERLAP ANALYSIS")
    print("="*70)

    toxic_neurons, gender_neurons = load_neuron_sets(toxic_neurons_path, gender_neurons_path)

    overlap_stats = compute_simple_overlap(toxic_neurons, gender_neurons)

    significance_stats = hypergeometric_significance_test(
        overlap_stats['overlap_count'],
        overlap_stats['toxic_total'],
        overlap_stats['gender_total']
    )

    phase1_report = generate_phase1_report(overlap_stats, significance_stats)
    print("\n" + phase1_report)

    # ========================================================================
    # PHASE 2: Differential Correlation
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 2: DIFFERENTIAL CORRELATION ANALYSIS")
    print("="*70)

    toxic_diff, gender_diff = load_attributions_sequential(
        general_paths,
        toxic_path,
        stereotype_path
    )

    toxic_scores, gender_scores, neuron_ids = extract_all_neuron_diffs(
        toxic_diff,
        gender_diff
    )

    correlation_stats = correlation_analysis(toxic_scores, gender_scores)

    quadrants = quadrant_classification(toxic_scores, gender_scores, neuron_ids)

    pruned_stats = analyze_pruned_quadrants(toxic_neurons, gender_neurons, quadrants)

    phase2_report = generate_phase2_report(correlation_stats, quadrants, pruned_stats)
    print(phase2_report)

    # ========================================================================
    # PHASE 4: Layer Distribution
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 4: LAYER DISTRIBUTION ANALYSIS")
    print("="*70)

    layer_stats = layer_distribution_analysis(toxic_neurons, gender_neurons)

    layer_groups = compute_overlap_by_layer(layer_stats)

    hotspots = identify_hotspot_layers(layer_stats, top_n=10)

    phase4_report = generate_phase4_report(layer_stats, layer_groups, hotspots)
    print(phase4_report)

    # ========================================================================
    # FINAL INTERPRETATION
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL INTERPRETATION")
    print("="*70)

    overlap_count = overlap_stats['overlap_count']
    iou = overlap_stats['iou']
    correlation = correlation_stats['correlation']

    print(f"\nKey Findings:")
    print(f"  Neuron overlap: {overlap_count}/100 (IoU: {iou:.3f})")
    print(f"  Correlation: {correlation:.3f}")
    print(f"  Multi-bias neurons: {len(quadrants['both_negative']):,} ({100*len(quadrants['both_negative'])/quadrants['total']:.1f}%)")

    # Interpret based on research thresholds
    if overlap_count >= 30 and correlation > 0.5:
        interpretation = "UNIVERSAL BIAS MECHANISM"
        evidence = "High overlap + high correlation → shared circuits"
    elif overlap_count >= 10 and correlation > 0.3:
        interpretation = "PARTIALLY SHARED MECHANISMS"
        evidence = "Moderate overlap + moderate correlation → some shared circuits"
    elif overlap_count < 5 or correlation < 0.2:
        interpretation = "INDEPENDENT BIAS CIRCUITS"
        evidence = "Low overlap + low correlation → separate implementations"
    else:
        interpretation = "MIXED/UNCLEAR"
        evidence = "Metrics don't clearly indicate shared vs independent"

    print(f"\nConclusion: {interpretation}")
    print(f"Evidence: {evidence}")

    if overlap_count > 0 and significance_stats.get('p_value'):
        if significance_stats['p_value'] < 0.01:
            print(f"Statistical significance: YES (p={significance_stats['p_value']:.2e})")
        else:
            print(f"Statistical significance: NO (p={significance_stats['p_value']:.2e})")

    # Save complete report
    output_path = Path('circuit_overlap_analysis.json')
    output_data = {
        'phase1_overlap': overlap_stats,
        'phase1_significance': significance_stats,
        'phase2_correlation': correlation_stats,
        'phase2_quadrants': {
            'both_negative_count': len(quadrants['both_negative']),
            'toxic_only_count': len(quadrants['toxic_only']),
            'gender_only_count': len(quadrants['gender_only']),
            'general_count': len(quadrants['general_capability'])
        },
        'phase2_pruned': pruned_stats,
        'phase4_layer_groups': layer_groups,
        'interpretation': interpretation
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Complete analysis saved to {output_path}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
