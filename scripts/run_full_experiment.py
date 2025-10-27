"""Run complete SparC³ toxicity suppression experiment

This script orchestrates the full pipeline:
1. Load attribution scores (general + toxic)
2. Compute differential and identify neurons to prune
3. Load model and prune neurons
4. Evaluate before/after (perplexity + toxicity)
5. Save results

Usage:
    python scripts/run_full_experiment.py --general_scores scores/lrp_general_avg.pt --toxic_scores scores/lrp_toxic.pt --output results/
"""
import sys
sys.path.append('.')

import argparse
import torch
import os
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama

# Import our modules
from src.pruning import (
    load_and_average_seeds,
    compute_differential_scores,
    identify_neurons_to_prune,
    prune_neurons,
    save_neuron_indices,
    save_pruned_model
)
from src.evaluation import run_full_evaluation
from src.data_prep import load_samples, load_wikitext2


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run complete SparC³ toxicity suppression experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Attribution scores
    parser.add_argument(
        '--general_scores',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to general attribution scores. If multiple, will average them.'
    )
    parser.add_argument(
        '--toxic_scores',
        type=str,
        required=True,
        help='Path to toxic attribution scores'
    )

    # Pruning parameters
    parser.add_argument(
        '--num_neurons',
        type=int,
        default=100,
        help='Number of neurons to prune'
    )
    parser.add_argument(
        '--layer_pattern',
        type=str,
        default='up_proj',
        help='Layer name pattern to filter for pruning'
    )

    # Evaluation data
    parser.add_argument(
        '--toxic_prompts',
        type=str,
        required=True,
        help='Path to toxic prompts pickle file'
    )

    # Model
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
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        choices=['float32', 'bfloat16', 'float16'],
        help='Model precision'
    )

    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save results'
    )
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='Save pruned model weights'
    )

    # Optional
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace token (or set HF_TOKEN env var)'
    )
    parser.add_argument(
        '--skip_baseline',
        action='store_true',
        help='Skip baseline evaluation (before pruning)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*70)
    print("SparC³ Toxicity Suppression Experiment")
    print("="*70)
    print(f"Timestamp: {timestamp}")
    print(f"Model: {args.model}")
    print(f"Neurons to prune: {args.num_neurons}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get HF token
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    if not hf_token:
        print("⚠️  WARNING: No HuggingFace token found")

    # ========================================================================
    # 1. Load attribution scores
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Loading Attribution Scores")
    print("="*70)

    # Load general scores (average if multiple)
    if len(args.general_scores) > 1:
        print(f"Loading and averaging {len(args.general_scores)} general score files...")
        general_scores = load_and_average_seeds(args.general_scores)
    else:
        print(f"Loading general scores from {args.general_scores[0]}...")
        general_scores = torch.load(args.general_scores[0])
        print(f"✓ Loaded {len(general_scores)} layers")

    # Load toxic scores
    print(f"\nLoading toxic scores from {args.toxic_scores}...")
    toxic_scores = torch.load(args.toxic_scores)
    print(f"✓ Loaded {len(toxic_scores)} layers")

    # ========================================================================
    # 2. Compute differential and identify neurons
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Computing Differential and Identifying Neurons")
    print("="*70)

    diff_scores = compute_differential_scores(general_scores, toxic_scores)

    neurons_to_prune = identify_neurons_to_prune(
        diff_scores,
        layer_pattern=args.layer_pattern,
        num_neurons=args.num_neurons
    )

    # Save neuron indices
    neuron_file = output_dir / f"pruned_neurons_{timestamp}.json"
    save_neuron_indices(
        neurons_to_prune,
        str(neuron_file),
        metadata={
            'timestamp': timestamp,
            'num_neurons': args.num_neurons,
            'layer_pattern': args.layer_pattern,
            'method': 'lrp_differential'
        }
    )

    # ========================================================================
    # 3. Load model
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Loading Model")
    print("="*70)

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    print(f"✓ Tokenizer loaded")

    print(f"\nLoading model {args.model}...")
    dtype_map = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }
    dtype = dtype_map[args.dtype]

    model = modeling_llama.LlamaForCausalLM.from_pretrained(
        args.model,
        token=hf_token,
        torch_dtype=dtype,
        device_map=args.device if args.device != 'auto' else 'auto'
    )

    device = str(next(model.parameters()).device)
    print(f"✓ Model loaded on {device}")

    # ========================================================================
    # 4. Evaluate baseline (before pruning)
    # ========================================================================
    results = {
        'timestamp': timestamp,
        'model': args.model,
        'num_neurons_pruned': args.num_neurons,
        'layer_pattern': args.layer_pattern
    }

    if not args.skip_baseline:
        print("\n" + "="*70)
        print("STEP 4: Baseline Evaluation (Before Pruning)")
        print("="*70)

        # Load evaluation data
        print("Loading evaluation data...")
        toxic_prompts = load_samples(args.toxic_prompts)
        wikitext = load_wikitext2(split='test')
        print(f"✓ Loaded {len(toxic_prompts)} toxic prompts")

        baseline_results = run_full_evaluation(
            model=model,
            tokenizer=tokenizer,
            wikitext_dataset=wikitext,
            toxic_prompts=toxic_prompts,
            output_file=str(output_dir / f"baseline_eval_{timestamp}.json"),
            verbose=True
        )

        results['baseline'] = baseline_results
    else:
        print("\n⚠️  Skipping baseline evaluation")

    # ========================================================================
    # 5. Prune neurons
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: Pruning Neurons")
    print("="*70)

    pruned_count = prune_neurons(model, neurons_to_prune, verbose=True)
    results['neurons_pruned_count'] = int(pruned_count)  # Ensure Python int for JSON

    # Save pruned model if requested
    if args.save_model:
        model_path = output_dir / f"pruned_model_{timestamp}"
        print(f"\nSaving pruned model to {model_path}...")
        save_pruned_model(model, str(model_path), tokenizer)

    # ========================================================================
    # 6. Evaluate after pruning
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: Evaluation After Pruning")
    print("="*70)

    # Reload evaluation data if not already loaded
    if args.skip_baseline:
        toxic_prompts = load_samples(args.toxic_prompts)
        wikitext = load_wikitext2(split='test')

    pruned_results = run_full_evaluation(
        model=model,
        tokenizer=tokenizer,
        wikitext_dataset=wikitext,
        toxic_prompts=toxic_prompts,
        output_file=str(output_dir / f"pruned_eval_{timestamp}.json"),
        verbose=True
    )

    results['pruned'] = pruned_results

    # ========================================================================
    # 7. Compute and display improvements
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    if not args.skip_baseline:
        baseline_ppl = results['baseline']['perplexity']
        baseline_tox = results['baseline']['toxicity_avg']
        pruned_ppl = results['pruned']['perplexity']
        pruned_tox = results['pruned']['toxicity_avg']

        ppl_change = float(((pruned_ppl - baseline_ppl) / baseline_ppl) * 100)
        tox_change = float(((pruned_tox - baseline_tox) / baseline_tox) * 100)

        print(f"\nPerplexity:")
        print(f"  Baseline: {baseline_ppl:.2f}")
        print(f"  Pruned:   {pruned_ppl:.2f}")
        print(f"  Change:   {ppl_change:+.2f}%")

        print(f"\nToxicity:")
        print(f"  Baseline: {baseline_tox:.4f}")
        print(f"  Pruned:   {pruned_tox:.4f}")
        print(f"  Change:   {tox_change:+.2f}%")

        results['improvements'] = {
            'perplexity_change_pct': float(ppl_change),
            'toxicity_change_pct': float(tox_change)
        }
    else:
        print(f"\nPruned Model Results:")
        print(f"  Perplexity: {results['pruned']['perplexity']:.2f}")
        print(f"  Toxicity:   {results['pruned']['toxicity_avg']:.4f}")

    # Save complete results
    results_file = output_dir / f"experiment_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Complete results saved to {results_file}")

    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
