"""Evaluation utilities for SparC³"""
import torch
import numpy as np
from typing import Dict, List, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm


def evaluate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    max_length: int = 2048,
    stride: int = 512,
    verbose: bool = True
) -> float:
    """
    Compute perplexity on a dataset using sliding window approach.

    Per paper methodology: use sliding window to compute perplexity on
    WikiText2 test set to measure general language modeling performance.

    Args:
        model: LLaMA model to evaluate
        tokenizer: HuggingFace tokenizer
        dataset: Dataset with 'text' field (e.g., WikiText2)
        max_length: Maximum sequence length for evaluation
        stride: Stride for sliding window
        verbose: If True, print progress

    Returns:
        Perplexity score (lower is better)

    Note:
        Implementation follows HuggingFace's standard perplexity evaluation:
        https://huggingface.co/docs/transformers/perplexity
    """
    if verbose:
        print(f"Evaluating perplexity (max_length={max_length}, stride={stride})...")

    model.eval()
    device = next(model.parameters()).device

    # Concatenate all text from dataset
    if hasattr(dataset, 'text'):
        # If dataset is a dict-like object with 'text' key
        texts = dataset['text']
    elif hasattr(dataset, '__iter__'):
        # If dataset is iterable with dict items
        texts = [item['text'] for item in dataset if item['text'].strip()]
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    full_text = '\n\n'.join(texts)

    # Tokenize full text
    encodings = tokenizer(full_text, return_tensors='pt')
    input_ids = encodings.input_ids

    if verbose:
        print(f"  Total tokens: {input_ids.size(1)}")

    # Compute negative log-likelihoods using sliding window
    nlls = []
    prev_end_loc = 0

    for begin_loc in tqdm(
        range(0, input_ids.size(1), stride),
        desc="Computing perplexity",
        disable=not verbose
    ):
        end_loc = min(begin_loc + max_length, input_ids.size(1))
        trg_len = end_loc - prev_end_loc  # Target length for this window

        input_ids_window = input_ids[:, begin_loc:end_loc].to(device)

        # Create target IDs (only compute loss on new tokens)
        target_ids = input_ids_window.clone()
        target_ids[:, :-trg_len] = -100  # Ignore already-seen tokens

        with torch.no_grad():
            outputs = model(input_ids_window, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == input_ids.size(1):
            break

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    ppl_value = ppl.item()

    if verbose:
        print(f"✓ Perplexity: {ppl_value:.2f}")

    return ppl_value


def evaluate_toxicity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    do_sample: bool = True,
    verbose: bool = True
) -> Tuple[float, List[float], List[str]]:
    """
    Generate completions for prompts and score toxicity with Detoxify.

    Per handoff decision: use temperature=0.7 for toxicity evaluation
    (not greedy decoding like repetition task).

    Args:
        model: LLaMA model to evaluate
        tokenizer: HuggingFace tokenizer
        prompts: List of prompt strings to complete
        max_new_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature (0.7 recommended)
        do_sample: If True, use sampling; if False, use greedy
        verbose: If True, print progress

    Returns:
        Tuple of (average_toxicity, per_sample_scores, completions)

    Raises:
        ImportError: If detoxify is not installed
    """
    try:
        from detoxify import Detoxify
    except ImportError:
        raise ImportError(
            "detoxify not installed. Install with: pip install detoxify"
        )

    if verbose:
        print(f"Evaluating toxicity on {len(prompts)} prompts...")
        print(f"  Generation: max_new_tokens={max_new_tokens}, "
              f"temperature={temperature}, do_sample={do_sample}")

    model.eval()
    device = next(model.parameters()).device

    # Initialize Detoxify model
    detoxify_model = Detoxify('original')

    toxicity_scores = []
    completions = []

    for prompt in tqdm(prompts, desc="Generating & scoring", disable=not verbose):
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        # Generate completion
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    top_p=0.9 if do_sample else None,  # Nucleus sampling
                )
            except Exception as e:
                if verbose:
                    print(f"⚠️  WARNING: Generation failed for prompt: {e}")
                toxicity_scores.append(0.0)
                completions.append("")
                continue

        # Decode completion (ONLY the generated part, not the prompt)
        prompt_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][prompt_length:]  # Skip prompt tokens
        completion_only = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Also keep full text for reference
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completions.append(full_text)

        # Score toxicity ONLY on the generated completion (not prompt)
        try:
            result = detoxify_model.predict(completion_only)
            toxicity = result['toxicity']
            toxicity_scores.append(toxicity)
        except Exception as e:
            if verbose:
                print(f"⚠️  WARNING: Toxicity scoring failed: {e}")
            toxicity_scores.append(0.0)

    # Compute average
    avg_toxicity = np.mean(toxicity_scores) if toxicity_scores else 0.0

    if verbose:
        print(f"✓ Average toxicity: {avg_toxicity:.4f}")
        print(f"  Min: {min(toxicity_scores):.4f}, Max: {max(toxicity_scores):.4f}")
        print(f"  Std: {np.std(toxicity_scores):.4f}")

    return avg_toxicity, toxicity_scores, completions


def evaluate_gender_bias(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    do_sample: bool = True,
    verbose: bool = True
) -> Tuple[float, List[float], List[str]]:
    """
    Generate completions for prompts and score gender bias with ModernBERT classifier.

    Analogous to evaluate_toxicity() but for gender bias detection.

    Args:
        model: LLaMA model to evaluate
        tokenizer: HuggingFace tokenizer
        prompts: List of prompt strings to complete
        max_new_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature (0.7 recommended)
        do_sample: If True, use sampling; if False, use greedy
        verbose: If True, print progress

    Returns:
        Tuple of (average_bias, per_sample_scores, completions)
    """
    if verbose:
        print(f"Evaluating gender bias on {len(prompts)} prompts...")
        print(f"  Generation: max_new_tokens={max_new_tokens}, "
              f"temperature={temperature}, do_sample={do_sample}")

    model.eval()
    device = next(model.parameters()).device

    # Initialize gender bias classifier
    print("  Loading ModernBERT gender bias classifier...")
    from src.data_prep import setup_gender_bias_classifier
    bias_scorer = setup_gender_bias_classifier()

    bias_scores = []
    completions = []

    for prompt in tqdm(prompts, desc="Generating & scoring", disable=not verbose):
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        # Generate completion
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    top_p=0.9 if do_sample else None,  # Nucleus sampling
                )
            except Exception as e:
                if verbose:
                    print(f"⚠️  WARNING: Generation failed for prompt: {e}")
                bias_scores.append(0.0)
                completions.append("")
                continue

        # Decode completion (ONLY the generated part, not the prompt)
        prompt_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][prompt_length:]  # Skip prompt tokens
        completion_only = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Also keep full text for reference
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completions.append(full_text)

        # Score gender bias ONLY on the generated completion (not prompt)
        try:
            bias_score = bias_scorer(completion_only)
            bias_scores.append(bias_score)
        except Exception as e:
            if verbose:
                print(f"⚠️  WARNING: Bias scoring failed: {e}")
            bias_scores.append(0.0)

    # Compute average
    avg_bias = np.mean(bias_scores) if bias_scores else 0.0

    if verbose:
        print(f"✓ Average gender bias: {avg_bias:.4f}")
        print(f"  Min: {min(bias_scores):.4f}, Max: {max(bias_scores):.4f}")
        print(f"  Std: {np.std(bias_scores):.4f}")

    return avg_bias, bias_scores, completions


def evaluate_racial_bias(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    do_sample: bool = True,
    verbose: bool = True
) -> Tuple[float, List[float], List[str]]:
    """
    Generate completions for prompts and score racial bias with ModernBERT classifier.

    Analogous to evaluate_toxicity() and evaluate_gender_bias() but for racial bias detection.

    Args:
        model: LLaMA model to evaluate
        tokenizer: HuggingFace tokenizer
        prompts: List of prompt strings to complete
        max_new_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature (0.7 recommended)
        do_sample: If True, use sampling; if False, use greedy
        verbose: If True, print progress

    Returns:
        Tuple of (average_bias, per_sample_scores, completions)
    """
    if verbose:
        print(f"Evaluating racial bias on {len(prompts)} prompts...")
        print(f"  Generation: max_new_tokens={max_new_tokens}, "
              f"temperature={temperature}, do_sample={do_sample}")

    model.eval()
    device = next(model.parameters()).device

    # Initialize racial bias classifier
    print("  Loading ModernBERT racial bias classifier...")
    from transformers import pipeline as hf_pipeline
    
    racial_classifier = hf_pipeline(
        "text-classification",
        model="cirimus/modernbert-large-bias-type-classifier",
        return_all_scores=True,
        device=-1  # CPU for classifier to save GPU memory
    )
    
    def score_racial_bias(text: str) -> float:
        """Extract racial bias score from ModernBERT classifier."""
        try:
            results = racial_classifier(text)
            # results[0] is list of dicts with 'label' and 'score'
            for item in results[0]:
                if 'racial' in item['label'].lower() or 'race' in item['label'].lower():
                    return item['score']
            return 0.0
        except Exception as e:
            print(f"⚠️ WARNING: Bias scoring failed: {e}")
            return 0.0

    bias_scores = []
    completions = []

    for prompt in tqdm(prompts, desc="Generating & scoring", disable=not verbose):
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        # Generate completion
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    top_p=0.9 if do_sample else None,  # Nucleus sampling
                )
            except Exception as e:
                if verbose:
                    print(f"⚠️  WARNING: Generation failed for prompt: {e}")
                bias_scores.append(0.0)
                completions.append("")
                continue

        # Decode completion (ONLY the generated part, not the prompt)
        prompt_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][prompt_length:]  # Skip prompt tokens
        completion_only = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Also keep full text for reference
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completions.append(full_text)

        # Score racial bias ONLY on the generated completion (not prompt)
        try:
            bias_score = score_racial_bias(completion_only)
            bias_scores.append(bias_score)
        except Exception as e:
            if verbose:
                print(f"⚠️  WARNING: Bias scoring failed: {e}")
            bias_scores.append(0.0)

    # Compute average
    avg_bias = np.mean(bias_scores) if bias_scores else 0.0

    if verbose:
        print(f"✓ Average racial bias: {avg_bias:.4f}")
        print(f"  Min: {min(bias_scores):.4f}, Max: {max(bias_scores):.4f}")
        print(f"  Std: {np.std(bias_scores):.4f}")

    return avg_bias, bias_scores, completions


def run_full_evaluation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    wikitext_dataset,
    toxic_prompts: List[str],
    output_file: str = None,
    verbose: bool = True
) -> Dict:
    """
    Run complete evaluation: perplexity + toxicity.

    Args:
        model: LLaMA model to evaluate
        tokenizer: HuggingFace tokenizer
        wikitext_dataset: WikiText2 test dataset for perplexity
        toxic_prompts: List of toxic prompts for toxicity evaluation
        output_file: Optional path to save results JSON
        verbose: If True, print progress

    Returns:
        Dictionary with evaluation results:
        {
            'perplexity': float,
            'toxicity_avg': float,
            'toxicity_std': float,
            'toxicity_scores': List[float],
            'completions': List[str]
        }
    """
    if verbose:
        print("="*60)
        print("Running full evaluation (perplexity + toxicity)")
        print("="*60)

    results = {}

    # 1. Evaluate perplexity
    if verbose:
        print("\n1. Evaluating perplexity on WikiText2...")
    ppl = evaluate_perplexity(
        model, tokenizer, wikitext_dataset, verbose=verbose
    )
    results['perplexity'] = float(ppl)  # Ensure Python float for JSON

    # 2. Evaluate toxicity
    if verbose:
        print("\n2. Evaluating toxicity...")
    avg_tox, tox_scores, completions = evaluate_toxicity(
        model, tokenizer, toxic_prompts, verbose=verbose
    )
    results['toxicity_avg'] = float(avg_tox)  # Ensure Python float
    results['toxicity_std'] = float(np.std(tox_scores))
    results['toxicity_scores'] = [float(s) for s in tox_scores]
    results['completions'] = completions

    # 3. Print summary
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Perplexity (WikiText2):  {ppl:.2f}")
        print(f"Toxicity (avg):          {avg_tox:.4f} ± {np.std(tox_scores):.4f}")
        print("="*60)

    # 4. Save results if requested
    if output_file:
        import json
        from pathlib import Path

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        if verbose:
            print(f"✓ Saved results to {output_file}")

    return results


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("\nAvailable functions:")
    print("  - evaluate_perplexity()")
    print("  - evaluate_toxicity()")
    print("  - run_full_evaluation()")
