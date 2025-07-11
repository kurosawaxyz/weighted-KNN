import torch
from typing import List, Tuple

from Wknn.nn import Hyperparam

def compute_capacities_improved(increments: torch.Tensor, inclusion_matrix: torch.Tensor, 
                               subsets: List[Tuple]) -> torch.Tensor:
    """Improved capacity computation with strict monotonicity"""
    n_subsets = len(subsets)
    caps = torch.zeros(n_subsets)
    
    # Sort subsets by size for proper monotonic assignment
    sorted_indices = sorted(range(n_subsets), key=lambda i: len(subsets[i]))
    
    # Assign capacities in order of subset size
    for idx in sorted_indices:
        subset = subsets[idx]
        
        if len(subset) == 0:
            caps[idx] = 0.0  # Empty set has capacity 0
        else:
            # Find all proper subsets
            proper_subsets = []
            for j, other_subset in enumerate(subsets):
                if set(other_subset).issubset(set(subset)) and len(other_subset) < len(subset):
                    proper_subsets.append(j)
            
            # Capacity must be at least the max of all proper subsets
            if proper_subsets:
                min_cap = torch.max(caps[proper_subsets])
            else:
                min_cap = torch.tensor(0.0)
            
            # Add increment on top of minimum required capacity
            caps[idx] = min_cap + increments[idx]
    
    # Normalize so that the full set has capacity 1
    full_set_idx = n_subsets - 1  # Full set is always last
    if caps[full_set_idx] > 0:
        caps = caps / caps[full_set_idx]
    
    # Ensure empty set is exactly 0
    caps[0] = 0.0
    
    return caps

def capacity_regularization(capacities: torch.Tensor, subsets: List[Tuple], 
                          config: Hyperparam) -> torch.Tensor:
    """Regularization to ensure proper capacity learning"""
    total_loss = torch.tensor(0.0)
    
    # Monotonicity constraint: if A ⊆ B, then ν(A) ≤ ν(B)
    for i, A in enumerate(subsets):
        for j, B in enumerate(subsets):
            if set(A).issubset(set(B)) and len(A) < len(B):
                # Penalize violations of monotonicity
                violation = F.relu(capacities[i] - capacities[j])
                total_loss += violation
    
    # Encourage sparsity to prevent all capacities from being 1
    sparsity_loss = torch.mean(capacities ** 2)
    total_loss += 0.1 * sparsity_loss
    
    return total_loss