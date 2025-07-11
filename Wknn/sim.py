# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
from typing import List, Tuple

def choquet_similarity_batch(query: torch.Tensor, others: torch.Tensor, 
                           capacities: torch.Tensor, subsets: List[Tuple]) -> torch.Tensor:
    """Compute Choquet similarity - corrected to be actual similarity"""
    n_others = others.shape[0]
    similarities = torch.zeros(n_others)
    
    for i, other in enumerate(others):
        # Compute feature-wise similarities (1 - normalized distance)
        feature_sims = 1.0 - torch.abs(query - other)
        
        # Compute Choquet integral
        sim_score = 0.0
        for j, subset in enumerate(subsets):
            if len(subset) == 0:
                continue
            
            # Minimum similarity over the subset
            min_sim = torch.min(feature_sims[list(subset)])
            
            # Find capacity of subset without current element(s)
            # This is a simplified Choquet integral computation
            if j > 0:
                prev_capacity = capacities[j-1] if j > 0 else 0.0
                sim_score += (capacities[j] - prev_capacity) * min_sim
            else:
                sim_score += capacities[j] * min_sim
        
        similarities[i] = sim_score
    
    return similarities