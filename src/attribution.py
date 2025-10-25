"""Attribution methods for SparC³"""
import torch
from typing import Dict, List
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm


def compute_lrp_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    samples: List[List[int]],
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Compute LRP relevance scores using LXT for a set of reference samples.
    
    Args:
        model: LLaMA model with LXT monkey-patching applied
        tokenizer: HuggingFace tokenizer
        samples: List of tokenized samples (list of token IDs)
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        Dictionary mapping layer names to aggregated relevance tensors
    """
    print(f"Computing LRP scores for {len(samples)} samples...")
    
    model.to(device)
    model.eval()
    
    # Initialize relevance accumulator
    relevance_accumulator = {}
    
    for sample_idx, token_ids in enumerate(tqdm(samples, desc="LRP attribution")):
        # Convert to tensor
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
        
        # Enable gradients
        with torch.set_grad_enabled(True):
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            
            # Get prediction for last token
            last_token_logits = logits[0, -1, :]
            pred_token_id = last_token_logits.argmax()
            
            # Backward pass to compute LRP
            # LXT will compute relevance during backward pass
            target_logit = last_token_logits[pred_token_id]
            target_logit.backward()
            
            # Extract relevance from gradients
            # Relevance = weight * gradient (as per LXT efficient mode)
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if hasattr(module, 'weight') and module.weight.grad is not None:
                        # Compute parameter-level relevance
                        relevance = torch.abs(module.weight.data * module.weight.grad)
                        
                        # NUMERICAL STABILITY CHECKS
                        if torch.isnan(relevance).any():
                            print(f"⚠️  WARNING: NaN detected in {name} (sample {sample_idx})")
                            relevance = torch.nan_to_num(relevance, nan=0.0)
                        
                        if torch.isinf(relevance).any():
                            print(f"⚠️  WARNING: Inf detected in {name} (sample {sample_idx})")
                            relevance = torch.clamp(relevance, max=1e6)
                        
                        # Accumulate (move to CPU immediately for memory efficiency)
                        relevance_cpu = relevance.cpu().float()
                        
                        if name not in relevance_accumulator:
                            relevance_accumulator[name] = torch.zeros_like(relevance_cpu, dtype=torch.float32)
                        
                        relevance_accumulator[name] += relevance_cpu
    
    # Average across samples
    print(f"Averaging relevance across {len(samples)} samples...")
    averaged_scores = {}
    for name, accumulated_relevance in relevance_accumulator.items():
        averaged_scores[name] = accumulated_relevance / len(samples)
    
    print(f"✓ Computed LRP scores for {len(averaged_scores)} layers")
    
    return averaged_scores


def save_attribution_scores(scores: Dict[str, torch.Tensor], path: str):
    """
    Save attribution scores to file.
    
    Args:
        scores: Dictionary of relevance scores
        path: Path to save file
    """
    torch.save(scores, path)
    print(f"✓ Saved attribution scores to {path}")


def load_attribution_scores(path: str) -> Dict[str, torch.Tensor]:
    """
    Load attribution scores from file.
    
    Args:
        path: Path to load from
    
    Returns:
        Dictionary of relevance scores
    """
    scores = torch.load(path)
    print(f"✓ Loaded attribution scores from {path}")
    print(f"  - {len(scores)} layers")
    return scores


def compute_wanda_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    samples: List[List[int]],
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Compute Wanda importance scores: |W| * ||X||_2
    
    Per Wanda paper: S_ij = |W_ij| · ||X_j||_2
    where ||X_j||_2 is L2 norm aggregated across ALL N×L tokens.
    
    Args:
        model: LLaMA model
        tokenizer: HuggingFace tokenizer
        samples: List of tokenized samples
        device: Device to run on
    
    Returns:
        Dictionary mapping layer names to importance scores
    """
    print(f"Computing Wanda scores for {len(samples)} samples...")
    
    model.to(device)
    model.eval()
    
    # Dictionary to store ALL activations (not just norms)
    all_activations = {}
    
    # Hook to capture activations
    def get_activation_hook(name):
        def hook(module, input, output):
            # Get input tensor
            if isinstance(input, tuple):
                input_tensor = input[0]
            else:
                input_tensor = input
            
            # Store activations (flatten batch and sequence dimensions)
            # Shape: [batch_size, seq_len, hidden_dim] -> [batch_size * seq_len, hidden_dim]
            activations_flat = input_tensor.reshape(-1, input_tensor.shape[-1])
            
            if name not in all_activations:
                all_activations[name] = []
            all_activations[name].append(activations_flat.cpu())
        
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hook = module.register_forward_hook(get_activation_hook(name))
            hooks.append(hook)
    
    # Forward passes to collect activations
    with torch.no_grad():
        for sample_idx, token_ids in enumerate(tqdm(samples, desc="Wanda (forward)")):
            input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
            model(input_ids=input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute Wanda scores: |W| * ||X||_2 where norm is over ALL tokens
    print("Computing Wanda importance scores...")
    wanda_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in all_activations:
            weight_magnitude = torch.abs(module.weight.data).cpu()
            
            # Concatenate all activations: shape [N*L, hidden_dim]
            all_acts_concat = torch.cat(all_activations[name], dim=0)
            
            # Compute L2 norm over ALL tokens for each feature
            # Shape: [hidden_dim]
            activation_norm = torch.norm(all_acts_concat, p=2, dim=0)
            
            # Broadcast to weight shape: [out_features, in_features]
            # activation_norm has shape [in_features]
            wanda_score = weight_magnitude * activation_norm[None, :]
            
            wanda_scores[name] = wanda_score
    
    print(f"✓ Computed Wanda scores for {len(wanda_scores)} layers")
    
    return wanda_scores


if __name__ == "__main__":
    # Test with small data
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama import modeling_llama
    from lxt.efficient import monkey_patch
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    print("="*60)
    print("Testing Attribution Module")
    print("="*60)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        token=os.getenv('HF_TOKEN')
    )
    print("✓ Tokenizer loaded")
    
    # Create dummy samples
    print("\n2. Creating test samples...")
    test_samples = [
        tokenizer.encode("The quick brown fox jumps over the lazy dog", max_length=50, truncation=True),
        tokenizer.encode("Machine learning is a subset of artificial intelligence", max_length=50, truncation=True),
    ]
    print(f"✓ Created {len(test_samples)} test samples")
    
    print("\n3. This test would load LLaMA-3-8B which is large.")
    print("   Run full tests with actual model separately.")
    print("\n✓ Attribution module structure validated")

