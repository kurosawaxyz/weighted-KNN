# -*- coding: utf-8 -*-
# @author: H. T. Duong V.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Hyperparam:
    """Configuration class for hyperparameters and settings"""
    def __init__(self, **kwargs):
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else: 
            self.hidden_dim = 64
            self.latent_dim = 32
            self.learning_rate = 1e-3
            self.epochs = 100
            self.batch_size = 32
            self.dropout_rate = 0.2
            self.weight_decay = 1e-4
            self.k_neighbors = 5
            self.temperature = 1.0
            self.capacity_reg = 0.1
            self.monotonicity_reg = 1.0

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