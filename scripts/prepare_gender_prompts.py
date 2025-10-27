"""Prepare gender-biased prompts using two-pass filtering

This script:
1. Loads candidate gender prompts from BOLD dataset
2. Loads LLaMA-3-8B model
3. Generates completions from candidates
4. Scores completions for gender bias
5. Selects top ~90 prompts that produce most biased completions
6. Saves to pickle file for attribution computation

This is a preprocessing step before running attribution computation.
"""
import sys
sys.path.append('.')

import argparse
import torch
import os
from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama
from dotenv import load_dotenv

from src.data_prep import prepare_gender_bias_prompts


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare gender-biased prompts via two-pass filtering'
    )

    parser.add_argument(
        '--target_count',
        type=int,
        default=93,
        help='Number of high-bias prompts to select'
    )
    parser.add_argument(
        '--max_candidates',
        type=int,
        default=200,
        help='Number of MGS stereotype candidates to evaluate'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Meta-Llama-3-8B',
        help='Model to use for generation'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/gender_bias_prompts.pkl',
        help='Output path for selected prompts'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace token (or set HF_TOKEN env var)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load environment
    load_dotenv()

    print("="*70)
    print("Gender Bias Prompt Preparation (Two-Pass Filtering)")
    print("="*70)
    print(f"Target count: {args.target_count}")
    print(f"Max candidates: {args.max_candidates}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print("="*70)

    # Get HF token
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    if not hf_token:
        print("⚠️  WARNING: No HuggingFace token found")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    print(f"✓ Tokenizer loaded")

    # Load model
    print(f"\n2. Loading model {args.model}...")
    model = modeling_llama.LlamaForCausalLM.from_pretrained(
        args.model,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    device = str(next(model.parameters()).device)
    print(f"✓ Model loaded on {device}")

    # Prepare gender bias prompts (two-pass filtering)
    print(f"\n3. Two-pass filtering (this will take ~15-30 min)...")
    prompts = prepare_gender_bias_prompts(
        model=model,
        tokenizer=tokenizer,
        target_count=args.target_count,
        max_candidates=args.max_candidates,
        save_path=args.output,
        use_two_pass=True
    )

    print("\n" + "="*70)
    print("✅ Gender Bias Prompt Preparation Complete!")
    print("="*70)
    print(f"Selected prompts: {len(prompts)}")
    print(f"Saved to: {args.output}")
    print("\nNext step: Compute attributions on these prompts")
    print(f"  python scripts/compute_attributions.py --method lrp --samples {args.output} --output scores/lrp_gender.pt")


if __name__ == "__main__":
    main()
