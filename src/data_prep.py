"""Data preparation utilities for SparC³"""
import pickle
from pathlib import Path
from typing import List, Dict
import random
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm


def prepare_c4_samples(
    tokenizer: PreTrainedTokenizer,
    n_samples: int = 128,
    seq_len: int = 2048,
    seed: int = 0,
    save_path: str = None
) -> List[List[int]]:
    """
    Prepare C4 reference samples for LRP attribution.
    
    Args:
        tokenizer: HuggingFace tokenizer
        n_samples: Number of samples to extract (default: 128)
        seq_len: Sequence length in tokens (default: 2048)
        seed: Random seed for reproducibility
        save_path: Path to save samples as pickle file
    
    Returns:
        List of tokenized samples (list of token IDs)
    """
    print(f"Preparing C4 samples (n={n_samples}, seq_len={seq_len}, seed={seed})...")
    
    # Load C4 dataset in streaming mode (using allenai/c4)
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    # Use dataset.shuffle with seed for true randomness
    # This ensures different seeds produce genuinely different samples
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    
    samples = []
    for example in tqdm(dataset, total=n_samples, desc="Extracting C4 samples"):
        text = example['text']
        
        # Tokenize
        tokens = tokenizer.encode(
            text,
            max_length=seq_len,
            truncation=True,
            add_special_tokens=True
        )
        
        # Only keep samples that reach the full sequence length
        if len(tokens) == seq_len:
            samples.append(tokens)
        
        if len(samples) >= n_samples:
            break
    
    print(f"✓ Extracted {len(samples)} samples")
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"✓ Saved to {save_path}")
    
    return samples


def prepare_toxic_prompts(
    tokenizer: PreTrainedTokenizer = None,
    min_toxicity: float = 0.9,
    target_count: int = 93,
    save_path: str = None
) -> List[str]:
    """
    Prepare toxic prompts from RealToxicityPrompts dataset.
    
    Note: The paper filters by toxicity of GENERATED completions (≥0.9).
    For now, we filter by prompt toxicity as a proxy, since we don't have
    generated completions yet. This will be refined later.
    
    Args:
        tokenizer: HuggingFace tokenizer (optional, for future filtering)
        min_toxicity: Minimum toxicity threshold (default: 0.9)
        target_count: Target number of prompts (default: 93)
        save_path: Path to save prompts as pickle file
    
    Returns:
        List of toxic prompts (strings)
    """
    print(f"Preparing toxic prompts (min_toxicity={min_toxicity}, target={target_count})...")
    
    # Load RealToxicityPrompts dataset
    dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
    
    toxic_prompts = []
    for example in tqdm(dataset, desc="Filtering toxic prompts"):
        # Get prompt text and toxicity score
        prompt_data = example['prompt']
        prompt_text = prompt_data['text']
        
        # Check if toxicity score exists and meets threshold
        if prompt_data.get('toxicity') is not None:
            toxicity = prompt_data['toxicity']
            if toxicity >= min_toxicity:
                toxic_prompts.append(prompt_text)
        
        if len(toxic_prompts) >= target_count:
            break
    
    print(f"✓ Found {len(toxic_prompts)} toxic prompts")
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(toxic_prompts, f)
        print(f"✓ Saved to {save_path}")
    
    return toxic_prompts


def load_wikitext2(split='test'):
    """
    Load WikiText2 dataset for perplexity evaluation.
    
    Args:
        split: Dataset split ('test', 'train', 'validation')
    
    Returns:
        Dataset object
    """
    print(f"Loading WikiText2 ({split} split)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    print(f"✓ Loaded {len(dataset)} examples")
    return dataset


def load_samples(path: str) -> List:
    """
    Load saved samples from pickle file.
    
    Args:
        path: Path to pickle file
    
    Returns:
        Loaded samples
    """
    with open(path, 'rb') as f:
        samples = pickle.load(f)
    print(f"✓ Loaded {len(samples)} samples from {path}")
    return samples


if __name__ == "__main__":
    # Test the functions
    from transformers import AutoTokenizer
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        token=os.getenv('HF_TOKEN')
    )
    
    # Test C4 sampling (small test)
    print("\n" + "="*50)
    print("Testing C4 sampling...")
    print("="*50)
    c4_samples = prepare_c4_samples(tokenizer, n_samples=5, seq_len=2048, seed=0)
    print(f"Sample 0 length: {len(c4_samples[0])}")
    print(f"Sample 0 first 10 tokens: {c4_samples[0][:10]}")
    
    # Test toxic prompts
    print("\n" + "="*50)
    print("Testing toxic prompts...")
    print("="*50)
    toxic_prompts = prepare_toxic_prompts(min_toxicity=0.9, target_count=10)
    print(f"First prompt: {toxic_prompts[0][:100]}...")

