#!/usr/bin/env python3
"""
Prepare racial stereotype prompts from StereoSet

Loads StereoSet race domain and saves 93 prompts for attribution computation.
"""

import sys
sys.path.append('.')

import argparse
from pathlib import Path

from src.data_prep import load_stereoset_race_prompts

def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare racial stereotype prompts from StereoSet'
    )
    
    parser.add_argument(
        '--max_prompts',
        type=int,
        default=93,
        help='Number of prompts to select (default: 93 to match toxic/gender)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/race_bias_prompts.pkl',
        help='Output path for selected prompts'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("STEREOSET RACIAL BIAS PROMPT PREPARATION")
    print("="*70)
    print(f"Target count: {args.max_prompts}")
    print(f"Output: {args.output}")
    print("="*70)
    
    # Load prompts
    prompts = load_stereoset_race_prompts(
        max_prompts=args.max_prompts,
        save_path=args.output
    )
    
    print("\n" + "="*70)
    print("✅ PROMPT PREPARATION COMPLETE")
    print("="*70)
    print(f"Selected prompts: {len(prompts)}")
    print(f"Saved to: {args.output}")
    print("\nSample prompts (first 5):")
    for i, prompt in enumerate(prompts[:5], 1):
        print(f"  {i}. {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    print("\nNext step: Validate differential signal")
    print(f"  python scripts/validate_racial_differential.py --samples {args.output}")


if __name__ == "__main__":
    main()

