"""Data preparation utilities for SparC³"""
import pickle
import torch
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


def load_bold_gender_candidates(max_prompts: int = 500) -> List[str]:
    """
    Load gender-related prompts from BOLD dataset as candidates for filtering.

    Args:
        max_prompts: Maximum number of candidate prompts to load

    Returns:
        List of gender-related prompt strings
    """
    print(f"Loading BOLD gender prompts (max {max_prompts})...")

    # Load BOLD dataset
    dataset = load_dataset("AlexaAI/bold", split="train")

    # Filter for gender domain
    gender_data = [row for row in dataset if row['domain'] == 'gender']

    print(f"✓ Found {len(gender_data)} gender entries in BOLD")

    # Extract prompts (each entry has multiple prompts)
    all_prompts = []
    for entry in gender_data:
        prompts = entry['prompts']
        # prompts is a list of strings
        if isinstance(prompts, list):
            all_prompts.extend(prompts)
        else:
            all_prompts.append(prompts)

        if len(all_prompts) >= max_prompts:
            break

    # Trim to max_prompts
    candidates = all_prompts[:max_prompts]

    print(f"✓ Extracted {len(candidates)} candidate gender prompts")

    return candidates


def load_mgs_stereotype_prompts(max_prompts: int = 200) -> List[str]:
    """
    Load gender stereotype prompts from MGS Stereotype-Elicitation-Prompt-Library.

    This dataset contains prompts specifically designed to elicit stereotypical
    completions (NOT biographical facts). Should create stronger differential
    signal than BOLD.

    Args:
        max_prompts: Maximum number of prompts to load (default: 200, all available)

    Returns:
        List of gender stereotype prompt strings
    """
    print(f"Loading MGS stereotype prompts (max {max_prompts})...")

    # Load MGS dataset
    dataset = load_dataset("wu981526092/Stereotype-Elicitation-Prompt-Library", split="train")

    # Filter for gender stereotypes
    gender_prompts = [
        ex['text'] for ex in dataset
        if ex['class'] == 'stereotype_gender'
    ]

    print(f"✓ Found {len(gender_prompts)} gender stereotype prompts in MGS")

    # Trim to max_prompts
    candidates = gender_prompts[:max_prompts]

    print(f"✓ Extracted {len(candidates)} stereotype-eliciting prompts")

    return candidates


def setup_gender_bias_classifier():
    """
    Setup ModernBERT gender bias classifier from HuggingFace.

    Returns:
        Function that scores text for gender bias (0-1 scale)
    """
    print("Loading ModernBERT gender bias classifier...")

    from transformers import pipeline

    # Load classifier with all scores
    classifier = pipeline(
        "text-classification",
        model="cirimus/modernbert-large-bias-type-classifier",
        return_all_scores=True
    )

    print("✓ Gender bias classifier loaded")

    def score_gender_bias(text: str) -> float:
        """
        Score a single text for gender bias.

        Args:
            text: Text to score

        Returns:
            Gender bias score (0-1, higher = more biased)
        """
        try:
            results = classifier(text)
            # results[0] is list of dicts with 'label' and 'score'
            # Find the 'Gender' label
            for item in results[0]:
                if 'gender' in item['label'].lower():
                    return item['score']

            # If no gender label found, return 0
            return 0.0
        except Exception as e:
            print(f"⚠️ WARNING: Bias scoring failed: {e}")
            return 0.0

    return score_gender_bias


def generate_and_filter_for_bias(
    model,
    tokenizer,
    candidate_prompts: List[str],
    bias_scorer,
    target_count: int = 93,
    max_new_tokens: int = 50,
    device: str = 'cuda'
) -> List[str]:
    """
    Two-pass approach: Generate completions and filter for highest bias.

    Pass 1: Generate completions from candidate prompts
    Pass 2: Score for gender bias and select most bias-inducing prompts

    Args:
        model: LLaMA model for generation
        tokenizer: HuggingFace tokenizer
        candidate_prompts: List of candidate prompts to evaluate
        bias_scorer: Function that scores text for bias (returns float)
        target_count: Number of high-bias prompts to select
        max_new_tokens: Tokens to generate per prompt
        device: Device for generation

    Returns:
        List of prompts that produce most gender-biased completions
    """
    print(f"\nTwo-pass filtering: {len(candidate_prompts)} candidates → {target_count} high-bias prompts")
    print(f"Pass 1: Generating completions...")

    model.eval()

    prompt_bias_scores = []

    for prompt in tqdm(candidate_prompts, desc="Generating & scoring"):
        # Generate completion
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    top_p=0.9
                )

                # Extract only the generated part (not prompt)
                prompt_length = inputs.input_ids.shape[1]
                generated_ids = outputs[0][prompt_length:]
                completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Score the completion for gender bias
                bias_score = bias_scorer(completion)

                prompt_bias_scores.append((prompt, bias_score))

            except Exception as e:
                print(f"⚠️ Generation failed for prompt: {e}")
                prompt_bias_scores.append((prompt, 0.0))

    print(f"✓ Generated and scored {len(prompt_bias_scores)} completions")

    # Pass 2: Sort by bias score and select top-N
    print(f"Pass 2: Selecting top {target_count} most bias-inducing prompts...")

    sorted_prompts = sorted(prompt_bias_scores, key=lambda x: x[1], reverse=True)
    selected = [prompt for prompt, score in sorted_prompts[:target_count]]

    avg_bias = sum(score for _, score in sorted_prompts[:target_count]) / target_count
    print(f"✓ Selected {len(selected)} prompts")
    print(f"  Average bias of selected: {avg_bias:.4f}")
    print(f"  Bias range: {sorted_prompts[0][1]:.4f} to {sorted_prompts[target_count-1][1]:.4f}")

    return selected


def prepare_gender_bias_prompts(
    model=None,
    tokenizer=None,
    target_count: int = 93,
    max_candidates: int = 500,
    save_path: str = None,
    use_two_pass: bool = True
) -> List[str]:
    """
    Prepare gender-biased prompts for attribution computation.

    Two-pass approach:
    1. Load BOLD gender candidate prompts
    2. Generate completions and score for bias
    3. Select prompts that produce most biased completions

    Args:
        model: LLaMA model for two-pass filtering (required if use_two_pass=True)
        tokenizer: HuggingFace tokenizer (required if use_two_pass=True)
        target_count: Number of high-bias prompts to select (default: 93)
        max_candidates: Number of BOLD candidates to consider (default: 500)
        save_path: Path to save selected prompts as pickle file
        use_two_pass: If True, use two-pass filtering (default: True)

    Returns:
        List of gender-biased prompt strings
    """
    print("="*70)
    print(f"Preparing Gender Stereotype Prompts (target: {target_count})")
    print("="*70)

    # Load candidate prompts from MGS Stereotype Library (NOT BOLD - learned from failure)
    candidates = load_mgs_stereotype_prompts(max_prompts=max_candidates)

    if not use_two_pass:
        # Simple approach: just use first target_count candidates
        selected = candidates[:target_count]
        print(f"✓ Selected first {len(selected)} candidates (no filtering)")
    else:
        # Two-pass filtering
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer required for two-pass filtering")

        # Setup bias scorer
        bias_scorer = setup_gender_bias_classifier()

        # Generate and filter
        device = str(next(model.parameters()).device)
        selected = generate_and_filter_for_bias(
            model,
            tokenizer,
            candidates,
            bias_scorer,
            target_count=target_count,
            device=device
        )

    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(selected, f)
        print(f"✓ Saved to {save_path}")

    return selected


def load_stereoset_race_prompts(
    max_prompts: int = 93,
    save_path: str = None
) -> List[str]:
    """
    Load racial stereotype prompts from StereoSet intrasentence dataset.

    Args:
        max_prompts: Number of prompts to select (default: 93 to match toxic/gender)
        save_path: Optional path to save selected prompts

    Returns:
        List of racial stereotype-eliciting prompt strings
    """
    print(f"Loading StereoSet race prompts (max {max_prompts})...")

    # Load StereoSet intrasentence dataset
    from datasets import load_dataset
    
    dataset = load_dataset("McGill-NLP/stereoset", "intrasentence")
    
    # Filter for race domain
    race_examples = [ex for ex in dataset['validation'] if ex['bias_type'] == 'race']
    
    print(f"✓ Found {len(race_examples)} race examples in StereoSet")
    
    # Extract prompts (context with BLANK marker)
    prompts = []
    for ex in race_examples:
        context = ex['context']
        # Use context before BLANK as prompt for completion
        if 'BLANK' in context:
            prompt = context.replace('BLANK', '').strip()
        else:
            prompt = context
        prompts.append(prompt)
    
    # Select first max_prompts (can add selection logic if needed)
    selected_prompts = prompts[:max_prompts]
    
    print(f"✓ Selected {len(selected_prompts)} racial stereotype prompts")
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(selected_prompts, f)
        print(f"✓ Saved to {save_path}")
    
    return selected_prompts


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

