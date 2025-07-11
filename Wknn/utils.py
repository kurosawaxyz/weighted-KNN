# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
import itertools
from typing import List, Tuple

def get_subsets(d: int) -> List[Tuple]:
    """Generate all subsets of features"""
    features = list(range(d))
    subsets = []
    for i in range(d + 1):
        subsets.extend(itertools.combinations(features, i))
    return subsets

def get_inclusion_matrix(subsets: List[Tuple]) -> torch.Tensor:
    """Compute inclusion matrix"""
    n = len(subsets)
    mat = torch.zeros((n, n), dtype=torch.float32)
    
    subset_sets = [set(s) for s in subsets]
    
    for i, A in enumerate(subset_sets):
        for j, B in enumerate(subset_sets):
            if B.issubset(A):
                mat[i, j] = 1.0
    return mat