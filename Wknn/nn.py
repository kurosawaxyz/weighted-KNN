# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from Wknn.capacity import compute_capacities_improved, capacity_regularization
from Wknn.utils import get_subsets, get_inclusion_matrix
from Wknn.sim import choquet_similarity_batch
from Wknn.hyperparam import Hyperparam

# === Feature Encoder ===
class FeatureEncoder(nn.Module):
    def __init__(self, d: int, hidden_dim: int = 32, dropout_rate: float = 0.1):
        super().__init__()
        self.d = d
        self.fc1 = nn.Linear(d, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim // 2)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        batch_size = X.shape[0]
        
        h = F.relu(self.fc1(X))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.layer_norm(h)
        h = self.fc3(h)
        
        if batch_size > 1:
            h = h.mean(dim=0)
        else:
            h = h.squeeze(0)
            
        return h
    
# === Capacity Generator ===
class CapacityGenerator(nn.Module):
    def __init__(self, latent_dim: int, num_subsets: int, config: Hyperparam):
        super().__init__()
        self.num_subsets = num_subsets
        self.config = config
        
        # Smaller network to prevent overfitting
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, num_subsets)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Initialize with small weights
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(latent))
        h = self.dropout(h)
        raw = self.fc2(h)
        
        # Use sigmoid to ensure bounded output
        increments = torch.sigmoid(raw) * 0.1  # Scale down to prevent saturation
        return increments
    

# === WKnn Class ===
class WKnn(nn.Module):
    """Complete Choquet XAI classifier"""
    def __init__(self, d: int, num_classes: int, config: Hyperparam):
        super().__init__()
        self.d = d
        self.num_classes = num_classes
        self.config = config
        self.subsets = get_subsets(d)
        self.inclusion_mat = get_inclusion_matrix(self.subsets)
        
        self.encoder = FeatureEncoder(d, config.hidden_dim, config.dropout_rate)
        self.capacity_generator = CapacityGenerator(
            config.latent_dim, len(self.subsets), config
        )
        
    def forward(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with leave-one-out cross-validation"""
        batch_size = X.shape[0]
        
        # Generate capacities from the data
        latent = self.encoder(X)
        increments = self.capacity_generator(latent)
        capacities = compute_capacities_improved(increments, self.inclusion_mat, self.subsets)
        
        # Leave-one-out predictions
        predictions = []
        
        for i in range(batch_size):
            query = X[i]
            
            # Get all other samples (leave-one-out)
            others_idx = [j for j in range(batch_size) if j != i]
            others = X[others_idx]
            others_labels = y[others_idx]
            
            # Compute Choquet similarities
            sims = choquet_similarity_batch(query, others, capacities, self.subsets)
            
            # Get k nearest neighbors
            if len(others_idx) > self.config.k_neighbors:
                _, top_k_idx = torch.topk(sims, self.config.k_neighbors)
                selected_labels = others_labels[top_k_idx]
                selected_sims = sims[top_k_idx]
            else:
                selected_labels = others_labels
                selected_sims = sims
            
            # Weighted voting with temperature
            weights = F.softmax(selected_sims / self.config.temperature, dim=0)
            
            # Compute class probabilities
            vote = torch.zeros(self.num_classes)
            for j, label in enumerate(selected_labels):
                vote[label] += weights[j]
            
            predictions.append(vote)
        
        prediction_scores = torch.stack(predictions)
        return prediction_scores, capacities