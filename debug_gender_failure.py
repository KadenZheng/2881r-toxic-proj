"""Debug why gender bias experiment failed catastrophically"""
import sys
sys.path.append('.')

import torch
import numpy as np
from src.pruning import load_and_average_seeds, compute_differential_scores, aggregate_to_neuron_level

print("="*80)
print("DEBUGGING GENDER BIAS EXPERIMENT FAILURE")
print("="*80)

# Load scores
print("\n1. Loading attribution scores...")
general = load_and_average_seeds([
    '/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed0.pt',
    '/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed1.pt',
    '/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_general_seed2.pt'
], device='cpu')

gender = torch.load('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_gender.pt', map_location='cpu')

print(f"General layers: {len(general)}")
print(f"Gender layers: {len(gender)}")

# Check total relevance
gen_total = sum(s.sum().item() for s in general.values())
gender_total = sum(s.sum().item() for s in gender.values())

print(f"\nTotal relevance:")
print(f"  General: {gen_total:.2e}")
print(f"  Gender: {gender_total:.2e}")
print(f"  Ratio (Gender/General): {gender_total/gen_total:.2f}")

# Compute differential
print("\n2. Computing differential...")
diff = compute_differential_scores(general, gender)

# Analyze up_proj layers
up_proj_layers = [k for k in diff.keys() if 'up_proj' in k]
print(f"\nup_proj layers: {len(up_proj_layers)}")

# Get all neuron-level differential scores
all_diffs = []
for layer in up_proj_layers:
    neuron_diffs = aggregate_to_neuron_level(diff[layer])
    all_diffs.extend(neuron_diffs.tolist())

all_diffs = np.array(all_diffs)

print(f"\nDifferential statistics (all {len(all_diffs):,} neurons):")
print(f"  Min: {all_diffs.min():.2e}")
print(f"  Max: {all_diffs.max():.2e}")
print(f"  Mean: {all_diffs.mean():.2e}")
print(f"  Median: {np.median(all_diffs):.2e}")

n_negative = (all_diffs < 0).sum()
n_positive = (all_diffs > 0).sum()
pct_negative = 100 * n_negative / len(all_diffs)

print(f"\nDistribution:")
print(f"  Negative (gender > general): {n_negative:,} ({pct_negative:.1f}%)")
print(f"  Positive (general > gender): {n_positive:,} ({100-pct_negative:.1f}%)")

# Critical check: Compare to toxicity differential
print("\n3. Comparing to toxicity differential...")
toxic = torch.load('/n/netscratch/kempner_krajan_lab/Lab/kzheng/sparc3_scores/lrp_toxic.pt', map_location='cpu')

toxic_total = sum(s.sum().item() for s in toxic.values())
print(f"Toxic total relevance: {toxic_total:.2e}")

diff_toxic = compute_differential_scores(general, toxic)
all_toxic_diffs = []
for layer in up_proj_layers:
    neuron_diffs = aggregate_to_neuron_level(diff_toxic[layer])
    all_toxic_diffs.extend(neuron_diffs.tolist())

all_toxic_diffs = np.array(all_toxic_diffs)

print(f"\nToxicity differential:")
print(f"  Negative: {(all_toxic_diffs < 0).sum():,} ({100*(all_toxic_diffs < 0).sum()/len(all_toxic_diffs):.1f}%)")
print(f"  Mean: {all_toxic_diffs.mean():.2e}")
print(f"  Median: {np.median(all_toxic_diffs):.2e}")

print(f"\nGender differential:")
print(f"  Negative: {n_negative:,} ({pct_negative:.1f}%)")
print(f"  Mean: {all_diffs.mean():.2e}")
print(f"  Median: {np.median(all_diffs):.2e}")

# CRITICAL DIAGNOSIS
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if pct_negative > 80:
    print("\n🚨 PROBLEM: >80% of neurons have negative R_diff!")
    print("   This means gender scores are HIGHER than general for most neurons")
    print("   Pruning removes neurons important for GENERAL performance!")
    print("   CAUSE: Gender attribution may have accumulated more relevance than expected")
elif pct_negative < 10:
    print("\n🚨 PROBLEM: <10% of neurons have negative R_diff!")
    print("   Almost no gender-specific neurons found")
    print("   Can't identify which neurons cause gender bias")
else:
    print(f"\n✓ Distribution looks reasonable ({pct_negative:.1f}% negative)")

# Check if gender relevance is abnormally high
if gender_total > gen_total:
    print(f"\n🚨 PROBLEM: Gender relevance ({gender_total:.2e}) > General ({gen_total:.2e})")
    print("   Gender scores are higher than general - this is backwards!")
    print("   CAUSE: 93 gender samples may have produced unusually high attributions")
    print("   FIX: Need to investigate why gender attribution is so high")

print("\n" + "="*80)
