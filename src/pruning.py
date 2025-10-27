"""Pruning utilities for SparC³"""
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def compute_differential_scores(
    general_scores: Dict[str, torch.Tensor],
    toxic_scores: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Compute differential attribution scores: R_diff = R_general - R_toxic

    Per paper Equation 7: Components with most negative R_diff are most
    toxic-specific (high relevance for toxic, low relevance for general).

    Args:
        general_scores: Attribution scores on general reference samples
        toxic_scores: Attribution scores on toxic reference samples

    Returns:
        Dictionary mapping layer names to differential scores.
        Most negative values indicate toxic-specific parameters.

    Raises:
        ValueError: If score dictionaries have no overlapping layers
    """
    if not general_scores or not toxic_scores:
        raise ValueError("Score dictionaries cannot be empty")

    diff_scores = {}

    for layer_name in general_scores:
        if layer_name in toxic_scores:
            # Compute differential: R_general - R_toxic
            general = general_scores[layer_name]
            toxic = toxic_scores[layer_name]

            # Validate shapes match
            if general.shape != toxic.shape:
                print(f"⚠️  WARNING: Shape mismatch for {layer_name}: "
                      f"{general.shape} vs {toxic.shape}. Skipping.")
                continue

            diff_scores[layer_name] = general - toxic

    if not diff_scores:
        raise ValueError(
            "No overlapping layers between general and toxic scores. "
            f"General has {len(general_scores)} layers, "
            f"toxic has {len(toxic_scores)} layers."
        )

    print(f"✓ Computed differential scores for {len(diff_scores)} layers")
    return diff_scores


def aggregate_to_neuron_level(
    weight_scores: torch.Tensor
) -> torch.Tensor:
    """
    Aggregate weight-level scores to neuron-level.

    For weight matrix W with shape [out_features, in_features],
    each row represents one output neuron's incoming weights.
    Neuron i's score = sum over all its incoming weights W[i, :].

    Args:
        weight_scores: Weight-level importance scores, shape [out_features, in_features]

    Returns:
        Neuron-level scores, shape [out_features]

    Raises:
        ValueError: If input tensor is not 2D
    """
    if weight_scores.dim() != 2:
        raise ValueError(
            f"Expected 2D weight matrix, got {weight_scores.dim()}D tensor "
            f"with shape {weight_scores.shape}"
        )

    # Sum over input dimension (dim=1) to get per-neuron score
    neuron_scores = weight_scores.sum(dim=1)

    return neuron_scores


def identify_neurons_to_prune(
    diff_scores: Dict[str, torch.Tensor],
    layer_pattern: str = 'up_proj',
    num_neurons: int = 100
) -> List[Tuple[str, int]]:
    """
    Identify neurons to prune based on differential scores.

    Selects neurons with most negative R_diff (most toxic-specific)
    globally across all matching layers.

    Args:
        diff_scores: Differential attribution scores (general - toxic)
        layer_pattern: Layer name pattern to filter (e.g., 'up_proj', 'mlp')
        num_neurons: Number of neurons to prune

    Returns:
        List of (layer_name, neuron_idx) tuples for neurons to prune,
        sorted by score (most toxic-specific first)

    Raises:
        ValueError: If no matching layers found or num_neurons invalid
    """
    if num_neurons <= 0:
        raise ValueError(f"num_neurons must be positive, got {num_neurons}")

    # Collect all neuron scores from target layers
    neuron_scores = []

    for layer_name, scores in diff_scores.items():
        if layer_pattern in layer_name:
            # Aggregate to neuron level
            try:
                neuron_level = aggregate_to_neuron_level(scores)
            except ValueError as e:
                print(f"⚠️  WARNING: Skipping {layer_name}: {e}")
                continue

            # Collect (score, layer_name, neuron_idx) for each neuron
            for neuron_idx, score in enumerate(neuron_level):
                neuron_scores.append((score.item(), layer_name, neuron_idx))

    if not neuron_scores:
        raise ValueError(
            f"No layers matching pattern '{layer_pattern}' found in diff_scores. "
            f"Available layers: {list(diff_scores.keys())[:5]}..."
        )

    # Sort ascending: most negative R_diff first (most toxic-specific)
    neuron_scores.sort(key=lambda x: x[0])

    # Return top num_neurons
    actual_num = min(num_neurons, len(neuron_scores))
    if actual_num < num_neurons:
        print(f"⚠️  WARNING: Only {actual_num} neurons available, "
              f"requested {num_neurons}")

    neurons_to_prune = [(name, idx) for score, name, idx in neuron_scores[:actual_num]]

    print(f"✓ Identified {len(neurons_to_prune)} neurons to prune from "
          f"{len(set(n[0] for n in neurons_to_prune))} layers")

    return neurons_to_prune


def prune_neurons(
    model,
    neurons_to_prune: List[Tuple[str, int]],
    verbose: bool = True
) -> int:
    """
    Prune neurons by zeroing their weights.

    For LLaMA MLP neurons identified by up_proj layer:
    - Zero row i in up_proj.weight[i, :]  (output neuron)
    - Zero row i in gate_proj.weight[i, :] (gating neuron)
    - Zero column i in down_proj.weight[:, i] (input to down projection)

    Args:
        model: LLaMA model (or any model with .named_modules())
        neurons_to_prune: List of (layer_name, neuron_idx) tuples
        verbose: If True, print progress

    Returns:
        Number of neurons successfully pruned

    Raises:
        ValueError: If neurons_to_prune is empty
    """
    if not neurons_to_prune:
        raise ValueError("neurons_to_prune cannot be empty")

    # Build a dict of layer_name -> module for fast lookup
    module_dict = {name: module for name, module in model.named_modules()}

    pruned_count = 0

    for layer_name, neuron_idx in neurons_to_prune:
        # Get the module
        if layer_name not in module_dict:
            if verbose:
                print(f"⚠️  WARNING: Layer {layer_name} not found in model")
            continue

        layer_module = module_dict[layer_name]

        # Check if it's a Linear layer
        if not isinstance(layer_module, torch.nn.Linear):
            if verbose:
                print(f"⚠️  WARNING: {layer_name} is not a Linear layer")
            continue

        # Zero the neuron in this layer
        if 'up_proj' in layer_name or 'gate_proj' in layer_name:
            # Zero output neuron (row in weight matrix)
            if neuron_idx >= layer_module.weight.size(0):
                if verbose:
                    print(f"⚠️  WARNING: Neuron index {neuron_idx} out of bounds "
                          f"for {layer_name} (size {layer_module.weight.size(0)})")
                continue

            layer_module.weight.data[neuron_idx, :] = 0

            # Zero bias if it exists
            if layer_module.bias is not None:
                layer_module.bias.data[neuron_idx] = 0

            # If this is up_proj, also zero gate_proj and down_proj
            if 'up_proj' in layer_name:
                # Find corresponding gate_proj and down_proj
                base_name = layer_name.replace('up_proj', '')
                gate_name = base_name + 'gate_proj'
                down_name = base_name + 'down_proj'

                # Zero gate_proj row
                if gate_name in module_dict:
                    gate_module = module_dict[gate_name]
                    if isinstance(gate_module, torch.nn.Linear):
                        if neuron_idx < gate_module.weight.size(0):
                            gate_module.weight.data[neuron_idx, :] = 0
                            if gate_module.bias is not None:
                                gate_module.bias.data[neuron_idx] = 0

                # Zero down_proj column
                if down_name in module_dict:
                    down_module = module_dict[down_name]
                    if isinstance(down_module, torch.nn.Linear):
                        if neuron_idx < down_module.weight.size(1):
                            down_module.weight.data[:, neuron_idx] = 0

            pruned_count += 1

        elif 'down_proj' in layer_name:
            # Zero input connection (column in weight matrix)
            if neuron_idx >= layer_module.weight.size(1):
                if verbose:
                    print(f"⚠️  WARNING: Neuron index {neuron_idx} out of bounds "
                          f"for {layer_name} column (size {layer_module.weight.size(1)})")
                continue

            layer_module.weight.data[:, neuron_idx] = 0
            # Note: No bias for input connections

            pruned_count += 1

    if verbose:
        print(f"✓ Successfully pruned {pruned_count}/{len(neurons_to_prune)} neurons")

    return pruned_count


def load_and_average_seeds(
    seed_paths: List[str],
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Load attribution scores from multiple seeds and average them.

    Per paper: "three sets of 128 samples, each from a different random
    seed to ensure robustness" - we average the attribution scores from
    all seeds before computing differential.

    Args:
        seed_paths: List of paths to attribution score files (.pt)
        device: Device to load tensors on

    Returns:
        Dictionary of averaged attribution scores

    Raises:
        ValueError: If no valid seed files found or layer mismatch
    """
    if not seed_paths:
        raise ValueError("seed_paths cannot be empty")

    print(f"Loading and averaging {len(seed_paths)} seed files...")

    # Load all seeds
    all_seeds = []
    for path in seed_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"⚠️  WARNING: File not found: {path}")
            continue

        try:
            scores = torch.load(path, map_location=device)
            all_seeds.append(scores)
            print(f"  ✓ Loaded {path_obj.name}: {len(scores)} layers")
        except Exception as e:
            print(f"⚠️  WARNING: Failed to load {path}: {e}")
            continue

    if not all_seeds:
        raise ValueError(f"No valid seed files loaded from {seed_paths}")

    # Validate all seeds have same layers
    layer_names = set(all_seeds[0].keys())
    for i, seed_scores in enumerate(all_seeds[1:], start=1):
        if set(seed_scores.keys()) != layer_names:
            print(f"⚠️  WARNING: Seed {i} has different layers than seed 0")

    # Compute average
    averaged_scores = {}

    for layer_name in layer_names:
        # Collect tensors from all seeds that have this layer
        layer_tensors = []
        for seed_scores in all_seeds:
            if layer_name in seed_scores:
                layer_tensors.append(seed_scores[layer_name])

        if not layer_tensors:
            continue

        # Stack and average
        stacked = torch.stack(layer_tensors, dim=0)  # [n_seeds, ...original shape...]
        averaged = stacked.mean(dim=0)  # Average over seeds
        averaged_scores[layer_name] = averaged

    print(f"✓ Averaged {len(all_seeds)} seeds → {len(averaged_scores)} layers")

    return averaged_scores


def save_neuron_indices(
    neurons_to_prune: List[Tuple[str, int]],
    save_path: str,
    metadata: Optional[Dict] = None
):
    """
    Save list of pruned neuron indices to file.

    Saves as JSON for easy inspection and reproducibility.

    Args:
        neurons_to_prune: List of (layer_name, neuron_idx) tuples
        save_path: Path to save JSON file
        metadata: Optional metadata dict (e.g., num_neurons, method, date)
    """
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'neurons': [
            {'layer': layer_name, 'index': idx}
            for layer_name, idx in neurons_to_prune
        ],
        'total': len(neurons_to_prune),
        'metadata': metadata or {}
    }

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved {len(neurons_to_prune)} neuron indices to {save_path}")


def load_neuron_indices(load_path: str) -> List[Tuple[str, int]]:
    """
    Load neuron indices from JSON file.

    Args:
        load_path: Path to JSON file

    Returns:
        List of (layer_name, neuron_idx) tuples
    """
    with open(load_path, 'r') as f:
        data = json.load(f)

    neurons = [
        (neuron['layer'], neuron['index'])
        for neuron in data['neurons']
    ]

    print(f"✓ Loaded {len(neurons)} neuron indices from {load_path}")

    return neurons


def save_pruned_model(
    model,
    save_path: str,
    tokenizer = None
):
    """
    Save pruned model weights to disk.

    Args:
        model: Pruned model
        save_path: Path to save model (directory or .pt file)
        tokenizer: Optional tokenizer to save alongside model
    """
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    if save_path_obj.suffix == '.pt':
        torch.save(model.state_dict(), save_path)
        print(f"✓ Saved pruned model state dict to {save_path}")
    else:
        # Save as HuggingFace format
        model.save_pretrained(save_path)
        print(f"✓ Saved pruned model to {save_path}")

        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)
            print(f"✓ Saved tokenizer to {save_path}")


if __name__ == "__main__":
    print("Pruning module loaded successfully!")
    print("\nAvailable functions:")
    print("  - compute_differential_scores()")
    print("  - aggregate_to_neuron_level()")
    print("  - identify_neurons_to_prune()")
    print("  - prune_neurons()")
    print("  - load_and_average_seeds()")
    print("  - save_neuron_indices() / load_neuron_indices()")
    print("  - save_pruned_model()")
