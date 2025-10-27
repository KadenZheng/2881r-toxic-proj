"""Compute attribution scores for SparC³ experiments

This script computes LRP or Wanda attribution scores on reference samples
and saves them for later use in pruning.

Usage:
    python scripts/compute_attributions.py --method lrp --samples data/c4_general_seed0.pkl --output scores/lrp_seed0.pt
"""
import sys
sys.path.append('.')

import argparse
import torch
import os
from pathlib import Path
from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama

# Import attribution methods
from src.attribution import compute_lrp_scores, compute_wanda_scores, save_attribution_scores
from src.data_prep import load_samples


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Compute attribution scores for SparC³',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['lrp', 'wanda'],
        help='Attribution method to use'
    )
    parser.add_argument(
        '--samples',
        type=str,
        required=True,
        help='Path to reference samples (.pkl file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save attribution scores (.pt file)'
    )

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Meta-Llama-3-8B',
        help='HuggingFace model name or path'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (cuda, cpu, or auto)'
    )

    # Optional arguments
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace token (or set HF_TOKEN env var)'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        choices=['float32', 'bfloat16', 'float16'],
        help='Model precision'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*70)
    print("SparC³ Attribution Computation")
    print("="*70)
    print(f"Method:  {args.method}")
    print(f"Samples: {args.samples}")
    print(f"Model:   {args.model}")
    print(f"Output:  {args.output}")
    print("="*70)

    # 1. Get HF token
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    if not hf_token:
        print("⚠️  WARNING: No HuggingFace token found. Set HF_TOKEN env var or use --hf_token")

    # 2. Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=hf_token,
        trust_remote_code=True
    )
    print(f"✓ Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # 3. Apply LXT monkey-patching if using LRP
    if args.method == 'lrp':
        print("\n2. Applying LXT monkey-patching for LRP...")
        try:
            from lxt.efficient import monkey_patch
            monkey_patch(modeling_llama, verbose=False)
            print("✓ LXT monkey-patching applied")
        except ImportError:
            print("❌ ERROR: LXT not installed. Install with: pip install lxt")
            sys.exit(1)
    else:
        print("\n2. Skipping LXT patching (Wanda doesn't need it)")

    # 4. Load model
    print(f"\n3. Loading model {args.model}...")

    # Determine dtype
    dtype_map = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }
    dtype = dtype_map[args.dtype]

    # Load model
    model = modeling_llama.LlamaForCausalLM.from_pretrained(
        args.model,
        token=hf_token,
        torch_dtype=dtype,
        device_map=args.device if args.device != 'auto' else 'auto',
        trust_remote_code=True
    )

    device = str(next(model.parameters()).device)
    print(f"✓ Model loaded on device: {device}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Load samples
    print(f"\n4. Loading reference samples from {args.samples}...")
    if not Path(args.samples).exists():
        print(f"❌ ERROR: Samples file not found: {args.samples}")
        sys.exit(1)

    samples = load_samples(args.samples)
    print(f"✓ Loaded {len(samples)} samples")
    if samples:
        print(f"  Sample length: {len(samples[0])} tokens")

    # 6. Compute attribution scores
    print(f"\n5. Computing {args.method.upper()} attribution scores...")

    if args.method == 'lrp':
        scores = compute_lrp_scores(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            device=device
        )
    elif args.method == 'wanda':
        scores = compute_wanda_scores(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            device=device
        )

    print(f"✓ Computed scores for {len(scores)} layers")

    # Print some statistics
    total_params = sum(s.numel() for s in scores.values())
    total_relevance = sum(s.sum().item() for s in scores.values())
    print(f"  Total parameters scored: {total_params:,}")
    print(f"  Total relevance: {total_relevance:.2e}")

    # 7. Save scores
    print(f"\n6. Saving scores to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_attribution_scores(scores, str(output_path))

    # Get file size
    file_size_gb = output_path.stat().st_size / (1024**3)
    print(f"  File size: {file_size_gb:.2f} GB")

    print("\n" + "="*70)
    print("✅ Attribution computation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
