"""Validate differential signal before proceeding to pruning

CRITICAL SAFETY CHECK: Ensures R_stereotype is significantly lower than R_general
to avoid pruning general-capability neurons (like BOLD failure).

Usage:
    python scripts/validate_differential.py \\
        --general_scores scores/lrp_general_seed*.pt \\
        --behavior_scores scores/lrp_stereotype.pt \\
        --min_differential 15.0
"""
import sys
sys.path.append('.')

import argparse
import torch
from pathlib import Path

from src.pruning import load_and_average_seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate differential signal strength'
    )

    parser.add_argument(
        '--general_scores',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to general attribution scores'
    )
    parser.add_argument(
        '--behavior_scores',
        type=str,
        required=True,
        help='Path to behavior-specific attribution scores (stereotype, toxic, etc.)'
    )
    parser.add_argument(
        '--min_differential',
        type=float,
        default=15.0,
        help='Minimum required differential percentage (default: 15%%)'
    )
    parser.add_argument(
        '--behavior_name',
        type=str,
        default='behavior',
        help='Name of behavior for reporting (e.g., "stereotype", "toxic")'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*70)
    print("DIFFERENTIAL SIGNAL VALIDATION")
    print("="*70)
    print(f"Behavior: {args.behavior_name}")
    print(f"Minimum required differential: {args.min_differential}%")
    print("="*70)

    # Load general scores
    print("\n1. Loading general attribution scores...")
    if len(args.general_scores) > 1:
        print(f"   Averaging {len(args.general_scores)} seed files...")
        general_scores = load_and_average_seeds(args.general_scores, device='cpu')
    else:
        general_scores = torch.load(args.general_scores[0], map_location='cpu')
        print(f"   ✓ Loaded {len(general_scores)} layers")

    # Load behavior scores
    print(f"\n2. Loading {args.behavior_name} attribution scores...")
    behavior_scores = torch.load(args.behavior_scores, map_location='cpu')
    print(f"   ✓ Loaded {len(behavior_scores)} layers")

    # Compute total relevance
    print("\n3. Computing total relevance...")

    general_total = sum(s.sum().item() for s in general_scores.values())
    behavior_total = sum(s.sum().item() for s in behavior_scores.values())

    print(f"   General total relevance: {general_total:.2e}")
    print(f"   {args.behavior_name.capitalize()} total relevance: {behavior_total:.2e}")

    # Compute differential
    diff_abs = general_total - behavior_total
    diff_pct = (diff_abs / general_total) * 100

    print(f"\n4. Differential Analysis:")
    print(f"   Absolute difference: {diff_abs:.2e}")
    print(f"   Percentage difference: {diff_pct:.2f}%")
    print(f"   Ratio ({args.behavior_name}/general): {behavior_total/general_total:.3f}")

    # Historical comparison
    print(f"\n5. Comparison with Known Results:")
    print(f"   Toxicity (successful): 4.32e+04 / 5.76e+04 = 0.750 (25% diff) ✅")
    print(f"   BOLD gender (failed): 5.69e+04 / 5.76e+04 = 0.988 (1.2% diff) ❌")
    print(f"   Current ({args.behavior_name}): {behavior_total:.2e} / {general_total:.2e} = {behavior_total/general_total:.3f} ({diff_pct:.1f}% diff)")

    # Decision
    print("\n" + "="*70)
    print("VALIDATION DECISION")
    print("="*70)

    if diff_pct >= args.min_differential:
        print(f"\n✅ PASS: Differential {diff_pct:.2f}% >= {args.min_differential}% threshold")
        print(f"\nDifferential signal is STRONG ENOUGH to distinguish")
        print(f"{args.behavior_name}-specific circuits from general circuits.")
        print("\n🎯 SAFE TO PROCEED with pruning experiment")
        print("\nNext step:")
        print(f"  sbatch slurm/run_gender_experiment.sbatch")
        return 0
    else:
        print(f"\n❌ FAIL: Differential {diff_pct:.2f}% < {args.min_differential}% threshold")
        print(f"\nDifferential signal is TOO WEAK - {args.behavior_name} prompts activate")
        print(f"too many of the same circuits as general text.")
        print("\n⚠️  DO NOT PROCEED with pruning - will damage model!")
        print("\nRecommended actions:")
        print("  1. Try different prompts (WinoBias, custom generation)")
        print("  2. Use more stereotype-specific prompts")
        print("  3. Check if prompts are triggering stereotypes vs facts")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
