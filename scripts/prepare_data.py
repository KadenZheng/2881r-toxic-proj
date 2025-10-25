"""Prepare all datasets for SparC³ experiments"""
import sys
sys.path.append('.')

from src.data_prep import prepare_c4_samples, prepare_toxic_prompts, load_wikitext2
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for SparC³')
    parser.add_argument('--c4_samples', type=int, default=128, help='Number of C4 samples')
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--toxic_count', type=int, default=93, help='Number of toxic prompts')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='Random seeds for C4')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    print("="*60)
    print("SparC³ Data Preparation")
    print("="*60)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        token=os.getenv('HF_TOKEN')
    )
    print(f"✓ Tokenizer loaded (vocab_size={tokenizer.vocab_size})")
    
    # Prepare C4 general samples for multiple seeds
    print(f"\n2. Preparing C4 general samples (n={args.c4_samples}, seq_len={args.seq_len})...")
    for seed in args.seeds:
        save_path = f"data/c4_general_seed{seed}.pkl"
        print(f"\n   Seed {seed}:")
        prepare_c4_samples(
            tokenizer,
            n_samples=args.c4_samples,
            seq_len=args.seq_len,
            seed=seed,
            save_path=save_path
        )
    
    # Prepare toxic prompts
    print(f"\n3. Preparing toxic prompts (target={args.toxic_count})...")
    prepare_toxic_prompts(
        tokenizer=tokenizer,
        min_toxicity=0.9,
        target_count=args.toxic_count,
        save_path="data/realtoxicityprompts_toxic.pkl"
    )
    
    # Load WikiText2 for caching
    print("\n4. Loading WikiText2 (for perplexity evaluation)...")
    wikitext_test = load_wikitext2(split='test')
    print(f"✓ WikiText2 test set: {len(wikitext_test)} examples")
    
    print("\n" + "="*60)
    print("✓ Data preparation complete!")
    print("="*60)
    print("\nGenerated files:")
    for seed in args.seeds:
        print(f"  - data/c4_general_seed{seed}.pkl")
    print(f"  - data/realtoxicityprompts_toxic.pkl")
    print(f"  - WikiText2 cached in HuggingFace cache")

if __name__ == "__main__":
    main()


