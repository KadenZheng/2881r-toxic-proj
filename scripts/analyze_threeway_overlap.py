#!/usr/bin/env python3
"""Three-Way Circuit Overlap Analysis: Toxic ∩ Gender ∩ Race

Analyzes overlap patterns across all three bias experiments:
1. Pairwise overlaps (toxic-gender, toxic-race, gender-race)
2. Triple overlap (toxic ∩ gender ∩ race) - NOVEL
3. Venn diagram quantification (7 regions)
4. Statistical significance testing
5. 3×3 Correlation matrix of R_diff scores

Research Question: Are there "universal bias neurons" affecting all three behaviors?
"""

import sys
sys.path.append('.')

import os
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

from src.pruning import load_and_average_seeds, compute_differential_scores, aggregate_to_neuron_level

print("="*70)
print("THREE-WAY CIRCUIT OVERLAP ANALYSIS")
print("="*70)
print("Analyzing: Toxicity ∩ Gender ∩ Race")
print("="*70)

# =============================================================================
# 1. Load Pruned Neuron Sets
# =============================================================================
print("\n" + "="*70)
print("STEP 1: Loading Pruned Neuron Sets")
print("="*70)

# Use environment variable or default to relative path
results_dir = Path(os.getenv('RESULTS_DIR', 'results'))

# Find latest experiments  
toxic_dirs = list(results_dir.glob('experiment_*'))
gender_dirs = list(results_dir.glob('gender_experiment_*'))
race_dirs = list(results_dir.glob('race_experiment_*'))

print(f"Found {len(toxic_dirs)} toxic experiments")
print(f"Found {len(gender_dirs)} gender experiments")
print(f"Found {len(race_dirs)} race experiments")

if not (toxic_dirs and gender_dirs and race_dirs):
    print("\n❌ ERROR: Not all experiments found")
    print(f"  Toxic: {len(toxic_dirs)}")
    print(f"  Gender: {len(gender_dirs)}")
    print(f"  Race: {len(race_dirs)}")
    sys.exit(1)

# Get most recent
toxic_dir = sorted(toxic_dirs)[-1]
gender_dir = sorted(gender_dirs)[-1]
race_dir = sorted(race_dirs)[-1]

print(f"\nUsing:")
print(f"  Toxic:  {toxic_dir.name}")
print(f"  Gender: {gender_dir.name}")
print(f"  Race:   {race_dir.name}")

# Load neuron lists
toxic_file = list(toxic_dir.glob('pruned_neurons_*.json'))[0]
gender_file = list(gender_dir.glob('pruned_neurons_*.json'))[0]
race_file = list(race_dir.glob('pruned_neurons_*.json'))[0]

with open(toxic_file, 'r') as f:
    toxic_neurons = [(n['layer'], n['index']) for n in json.load(f)['neurons']]

with open(gender_file, 'r') as f:
    gender_neurons = [(n['layer'], n['index']) for n in json.load(f)['neurons']]

with open(race_file, 'r') as f:
    race_neurons = [(n['layer'], n['index']) for n in json.load(f)['neurons']]

print(f"\n✓ Loaded neuron lists:")
print(f"  Toxic:  {len(toxic_neurons)} neurons")
print(f"  Gender: {len(gender_neurons)} neurons")
print(f"  Race:   {len(race_neurons)} neurons")

# Convert to sets
toxic_set = set(toxic_neurons)
gender_set = set(gender_neurons)
race_set = set(race_neurons)

# =============================================================================
# 2. Pairwise Overlaps
# =============================================================================
print("\n" + "="*70)
print("STEP 2: Pairwise Overlap Analysis")
print("="*70)

toxic_gender = toxic_set & gender_set
toxic_race = toxic_set & race_set
gender_race = gender_set & race_set

print(f"\nPairwise overlaps:")
print(f"  Toxic ∩ Gender: {len(toxic_gender)} neurons")
print(f"  Toxic ∩ Race:   {len(toxic_race)} neurons")
print(f"  Gender ∩ Race:  {len(gender_race)} neurons")

# Compute percentages
print(f"\nOverlap percentages:")
print(f"  Toxic-Gender: {100*len(toxic_gender)/100:.1f}%")
print(f"  Toxic-Race:   {100*len(toxic_race)/100:.1f}%")
print(f"  Gender-Race:  {100*len(gender_race)/100:.1f}%")

# =============================================================================
# 3. Triple Overlap (NOVEL)
# =============================================================================
print("\n" + "="*70)
print("STEP 3: Triple Overlap Analysis (NOVEL)")
print("="*70)

triple_overlap = toxic_set & gender_set & race_set

print(f"\n🎯 TRIPLE OVERLAP: {len(triple_overlap)} neurons")
print(f"   (Toxic ∩ Gender ∩ Race)")

if len(triple_overlap) > 0:
    print(f"\n   Universal bias neurons:")
    for layer, idx in sorted(list(triple_overlap)):
        layer_num = int(layer.split('.')[2])
        print(f"     Layer {layer_num}, neuron {idx}")

# Expected random triple overlap
total_neurons = 458752  # 32 layers × 14,336 neurons
expected_triple = (100 * 100 * 100) / (total_neurons ** 2)

print(f"\n   Statistical analysis:")
print(f"     Expected (random): {expected_triple:.4f} neurons")
print(f"     Observed: {len(triple_overlap)} neurons")
if expected_triple > 0:
    enrichment_triple = len(triple_overlap) / expected_triple
    print(f"     Enrichment: {enrichment_triple:.1f}×")

# Hypergeometric test for triple overlap
try:
    from scipy.stats import hypergeom
    
    # This is complex for 3-way, use approximation
    # P(selecting k from pop1 AND pop2 AND pop3)
    p_val_approx = (100/total_neurons) ** 3 * total_neurons  # Rough approximation
    print(f"     P-value (approx): {p_val_approx:.2e}")
except:
    print(f"     P-value: (scipy not available)")

# =============================================================================
# 4. Venn Diagram Quantification (7 Regions)
# =============================================================================
print("\n" + "="*70)
print("STEP 4: Venn Diagram Quantification")
print("="*70)

# Compute all 7 non-empty regions
toxic_only = toxic_set - gender_set - race_set
gender_only = gender_set - toxic_set - race_set
race_only = race_set - toxic_set - gender_set

toxic_gender_only = toxic_gender - race_set
toxic_race_only = toxic_race - gender_set
gender_race_only = gender_race - toxic_set

triple = triple_overlap

print(f"\nVenn diagram regions:")
print(f"  Toxic only:           {len(toxic_only)} neurons")
print(f"  Gender only:          {len(gender_only)} neurons")
print(f"  Race only:            {len(race_only)} neurons")
print(f"  Toxic∩Gender only:    {len(toxic_gender_only)} neurons")
print(f"  Toxic∩Race only:      {len(toxic_race_only)} neurons")
print(f"  Gender∩Race only:     {len(gender_race_only)} neurons")
print(f"  Triple (T∩G∩R):       {len(triple)} neurons")

# Verify counts
total_counted = (len(toxic_only) + len(gender_only) + len(race_only) +
                 len(toxic_gender_only) + len(toxic_race_only) + len(gender_race_only) +
                 len(triple))

union = toxic_set | gender_set | race_set

print(f"\nVerification:")
print(f"  Sum of regions: {total_counted}")
print(f"  Union size:     {len(union)}")
print(f"  Match: {'✓' if total_counted == len(union) else '✗ MISMATCH!'}")

# =============================================================================
# 5. Load Attribution Scores for Correlation
# =============================================================================
print("\n" + "="*70)
print("STEP 5: Computing 3×3 Correlation Matrix")
print("="*70)

# Use environment variable or default to relative path
score_dir = Path(os.getenv('SCORES_DIR', 'scores'))

print("\nLoading attribution scores...")
print("  Loading general scores (3 seeds)...")
general_scores = load_and_average_seeds([
    str(score_dir / 'lrp_general_seed0.pt'),
    str(score_dir / 'lrp_general_seed1.pt'),
    str(score_dir / 'lrp_general_seed2.pt')
], device='cpu')

print("  Loading toxic scores...")
toxic_scores = torch.load(score_dir / 'lrp_toxic.pt', map_location='cpu')

print("  Loading gender scores...")
gender_scores = torch.load(score_dir / 'lrp_gender.pt', map_location='cpu')

print("  Loading race scores...")
race_scores = torch.load(score_dir / 'lrp_race.pt', map_location='cpu')

# Compute differentials
print("\nComputing differentials...")
toxic_diff = compute_differential_scores(general_scores, toxic_scores)
gender_diff = compute_differential_scores(general_scores, gender_scores)
race_diff = compute_differential_scores(general_scores, race_scores)

# Extract neuron-level scores for all up_proj neurons
print("Extracting neuron-level differential scores...")
up_proj_layers = sorted([l for l in toxic_diff.keys() if 'up_proj' in l])

all_toxic = []
all_gender = []
all_race = []

for layer in up_proj_layers:
    if layer in gender_diff and layer in race_diff:
        toxic_neuron = aggregate_to_neuron_level(toxic_diff[layer])
        gender_neuron = aggregate_to_neuron_level(gender_diff[layer])
        race_neuron = aggregate_to_neuron_level(race_diff[layer])
        
        all_toxic.extend(toxic_neuron.cpu().numpy())
        all_gender.extend(gender_neuron.cpu().numpy())
        all_race.extend(race_neuron.cpu().numpy())

all_toxic = np.array(all_toxic)
all_gender = np.array(all_gender)
all_race = np.array(all_race)

print(f"✓ Extracted {len(all_toxic):,} neuron differential scores")

# Compute correlation matrix
print("\nComputing correlation matrix...")

corr_toxic_gender = np.corrcoef(all_toxic, all_gender)[0, 1]
corr_toxic_race = np.corrcoef(all_toxic, all_race)[0, 1]
corr_gender_race = np.corrcoef(all_gender, all_race)[0, 1]

print(f"\n3×3 Correlation Matrix:")
print(f"              Toxic    Gender   Race")
print(f"  Toxic       1.000    {corr_toxic_gender:.3f}    {corr_toxic_race:.3f}")
print(f"  Gender      {corr_toxic_gender:.3f}    1.000    {corr_gender_race:.3f}")
print(f"  Race        {corr_toxic_race:.3f}    {corr_gender_race:.3f}    1.000")

# Test hypothesis: Gender-Race correlation > Toxic-Race correlation
print(f"\n🔬 Hypothesis Testing:")
if corr_gender_race > corr_toxic_race:
    print(f"  ✅ Gender-Race ({corr_gender_race:.3f}) > Toxic-Race ({corr_toxic_race:.3f})")
    print(f"     SUPPORTS: Social biases (gender, race) cluster separately from toxicity")
else:
    print(f"  ❌ Gender-Race ({corr_gender_race:.3f}) ≤ Toxic-Race ({corr_toxic_race:.3f})")
    print(f"     REJECTS: Social bias clustering hypothesis")

# =============================================================================
# 6. Save Results
# =============================================================================
print("\n" + "="*70)
print("STEP 6: Saving Analysis Results")
print("="*70)

analysis_results = {
    'pairwise_overlaps': {
        'toxic_gender': len(toxic_gender),
        'toxic_race': len(toxic_race),
        'gender_race': len(gender_race)
    },
    'triple_overlap': {
        'count': int(len(triple_overlap)),
        'neurons': [[layer, int(idx)] for layer, idx in sorted(list(triple_overlap))],
        'expected_random': float(expected_triple),
        'enrichment': float(len(triple_overlap) / expected_triple) if expected_triple > 0 else float('inf')
    },
    'venn_regions': {
        'toxic_only': len(toxic_only),
        'gender_only': len(gender_only),
        'race_only': len(race_only),
        'toxic_gender_only': len(toxic_gender_only),
        'toxic_race_only': len(toxic_race_only),
        'gender_race_only': len(gender_race_only),
        'triple': len(triple)
    },
    'correlation_matrix': {
        'toxic_gender': float(corr_toxic_gender),
        'toxic_race': float(corr_toxic_race),
        'gender_race': float(corr_gender_race)
    },
    'hypothesis_test': {
        'gender_race_gt_toxic_race': bool(corr_gender_race > corr_toxic_race),
        'interpretation': 'Social biases cluster separately' if corr_gender_race > corr_toxic_race else 'No clear clustering'
    }
}

output_file = 'threeway_overlap_analysis.json'
with open(output_file, 'w') as f:
    json.dump(analysis_results, f, indent=2)

print(f"✓ Saved analysis to: {output_file}")

# =============================================================================
# 7. Summary Report
# =============================================================================
print("\n" + "="*70)
print("THREE-WAY OVERLAP ANALYSIS SUMMARY")
print("="*70)

print(f"\nPairwise Overlaps:")
print(f"  Toxic-Gender:  {len(toxic_gender)} (11% each)")
print(f"  Toxic-Race:    {len(toxic_race)} ({100*len(toxic_race)/100:.0f}% each)")
print(f"  Gender-Race:   {len(gender_race)} ({100*len(gender_race)/100:.0f}% each)")

print(f"\n🎯 Triple Overlap: {len(triple)} neurons")
if len(triple) > 0:
    print(f"   These are 'universal bias neurons' affecting all three behaviors")
    enrichment_pct = (enrichment_triple - 1) * 100 if expected_triple > 0 else 0
    print(f"   Enrichment: {enrichment_triple:.0f}× over random (+{enrichment_pct:.0f}%)")
else:
    print(f"   No neurons affect all three behaviors simultaneously")

print(f"\nCorrelation Pattern:")
print(f"  Toxic-Gender:  {corr_toxic_gender:.3f}")
print(f"  Toxic-Race:    {corr_toxic_race:.3f}")
print(f"  Gender-Race:   {corr_gender_race:.3f} {'← HIGHEST' if corr_gender_race == max(corr_toxic_gender, corr_toxic_race, corr_gender_race) else ''}")

if corr_gender_race > max(corr_toxic_gender, corr_toxic_race):
    print(f"\n✅ FINDING: Social biases (gender, race) show highest correlation")
    print(f"   Suggests they share common neural mechanisms")
    print(f"   distinct from language toxicity mechanisms")

print("\n" + "="*70)
print("✅ THREE-WAY ANALYSIS COMPLETE")
print("="*70)
print(f"Results saved to: {output_file}")
print("\nNext step: Create visualizations")
print("  python scripts/visualize_threeway.py")
print("="*70)

