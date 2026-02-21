#!/usr/bin/env python3
"""
Task-Specific Prediction Heads for Multi-Task Learning
=======================================================
MLP-based prediction heads for each surfactant property.

Architecture options:
    1. Independent heads: Separate MLP for each task
    2. Shared-bottom heads: Shared layers + task-specific tops
    3. Attention-based heads: Cross-task attention

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


# =============================================================================
# Single Task Head
# =============================================================================

class TaskHead(nn.Module):
    """
    Single task prediction head (MLP).
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : list
        List of hidden layer dimensions.
    output_dim : int
        Output dimension (1 for regression).
    dropout : float
        Dropout probability.
    activation : str
        Activation function ('relu', 'gelu', 'silu').
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        output_dim: int = 1,
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Select activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        elif activation == 'silu':
            act_fn = nn.SiLU
        else:
            act_fn = nn.ReLU
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features [batch_size, input_dim].
            
        Returns
        -------
        torch.Tensor
            Predictions [batch_size, output_dim].
        """
        return self.network(x)


# =============================================================================
# Multi-Task Head Collection
# =============================================================================

class MultiTaskHeads(nn.Module):
    """
    Collection of task-specific prediction heads.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension from encoder.
    task_names : list
        List of task names.
    hidden_dims : list
        Hidden dimensions for each head.
    dropout : float
        Dropout probability.
    share_bottom : bool
        If True, share bottom layers across tasks.
    shared_layers : int
        Number of shared layers (if share_bottom=True).
    """
    
    def __init__(
        self,
        input_dim: int,
        task_names: List[str],
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2,
        share_bottom: bool = False,
        shared_layers: int = 1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.share_bottom = share_bottom
        
        if share_bottom and shared_layers > 0:
            # Shared bottom network
            shared_dim = hidden_dims[0] if hidden_dims else input_dim
            self.shared_network = nn.Sequential(
                nn.Linear(input_dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            head_input_dim = shared_dim
            head_hidden_dims = hidden_dims[1:] if len(hidden_dims) > 1 else [64]
        else:
            self.shared_network = None
            head_input_dim = input_dim
            head_hidden_dims = hidden_dims
        
        # Create individual task heads
        self.heads = nn.ModuleDict()
        for task_name in task_names:
            self.heads[task_name] = TaskHead(
                input_dim=head_input_dim,
                hidden_dims=head_hidden_dims,
                output_dim=1,
                dropout=dropout
            )
    
    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for all tasks.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features [batch_size, input_dim].
        return_dict : bool
            If True, return dict; else return stacked tensor.
            
        Returns
        -------
        dict or torch.Tensor
            Task predictions.
        """
        # Apply shared network if exists
        if self.shared_network is not None:
            x = self.shared_network(x)
        
        # Get predictions from each head
        predictions = {}
        for task_name in self.task_names:
            predictions[task_name] = self.heads[task_name](x)
        
        if return_dict:
            return predictions
        else:
            # Stack predictions: [batch_size, num_tasks]
            return torch.cat(
                [predictions[t] for t in self.task_names],
                dim=-1
            )
    
    def forward_task(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        """Forward pass for a single task."""
        if self.shared_network is not None:
            x = self.shared_network(x)
        return self.heads[task_name](x)


# =============================================================================
# Uncertainty-Weighted Multi-Task Head
# =============================================================================

class UncertaintyWeightedHeads(nn.Module):
    """
    Multi-task heads with learnable uncertainty weights.
    
    Based on:
        Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
        CVPR 2018
    
    Each task has a learnable log variance parameter that automatically
    balances the contribution of each task to the total loss.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    task_names : list
        List of task names.
    hidden_dims : list
        Hidden dimensions for each head.
    dropout : float
        Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int,
        task_names: List[str],
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.task_names = task_names
        self.num_tasks = len(task_names)
        
        # Create task heads
        self.heads = nn.ModuleDict()
        for task_name in task_names:
            self.heads[task_name] = TaskHead(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
                dropout=dropout
            )
        
        # Learnable log variance for each task (for uncertainty weighting)
        # Initialize to 0 (variance = 1)
        self.log_vars = nn.ParameterDict({
            task_name: nn.Parameter(torch.zeros(1))
            for task_name in task_names
        })
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features [batch_size, input_dim].
            
        Returns
        -------
        dict
            Task predictions.
        """
        predictions = {}
        for task_name in self.task_names:
            predictions[task_name] = self.heads[task_name](x)
        return predictions
    
    def get_task_weights(self) -> Dict[str, float]:
        """
        Get current task weights based on learned uncertainties.
        
        Returns
        -------
        dict
            Task weights (1 / exp(log_var)).
        """
        weights = {}
        for task_name in self.task_names:
            # Weight = 1 / (2 * variance) = 0.5 * exp(-log_var)
            weights[task_name] = (0.5 * torch.exp(-self.log_vars[task_name])).item()
        return weights
    
    def get_regularization_term(self) -> torch.Tensor:
        """
        Get regularization term for uncertainty weighting.
        
        The total loss is: sum_i [ (1/2σ²_i) * L_i + (1/2) * log(σ²_i) ]
        This returns the regularization term: sum_i (1/2) * log(σ²_i)
        """
        reg = 0.0
        for task_name in self.task_names:
            reg = reg + 0.5 * self.log_vars[task_name]
        return reg


# =============================================================================
# Cross-Stitch Network for Task Interaction
# =============================================================================

class CrossStitchUnit(nn.Module):
    """
    Cross-stitch unit for learning task relationships.
    
    Allows tasks to share information through learnable linear combinations.
    """
    
    def __init__(self, num_tasks: int):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # Initialize alpha matrix close to identity
        # This means tasks initially focus on their own features
        alpha = torch.eye(num_tasks) * 0.9 + torch.ones(num_tasks, num_tasks) * 0.1 / num_tasks
        self.alpha = nn.Parameter(alpha)
    
    def forward(self, task_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply cross-stitch operation.
        
        Parameters
        ----------
        task_features : list
            List of task feature tensors [batch_size, feature_dim].
            
        Returns
        -------
        list
            Combined task features.
        """
        # Stack features: [num_tasks, batch_size, feature_dim]
        stacked = torch.stack(task_features, dim=0)
        
        # Apply softmax to alpha to get valid mixing weights
        weights = F.softmax(self.alpha, dim=1)
        
        # Combine: output[i] = sum_j alpha[i,j] * input[j]
        combined = torch.einsum('ij,jbf->ibf', weights, stacked)
        
        # Split back to list
        return [combined[i] for i in range(self.num_tasks)]


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Task Heads")
    print("=" * 60)
    
    task_names = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']
    batch_size = 32
    input_dim = 256
    
    # Test single head
    print("\n--- Single Task Head ---")
    head = TaskHead(input_dim=input_dim, hidden_dims=[128, 64])
    x = torch.randn(batch_size, input_dim)
    out = head(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    
    # Test multi-task heads
    print("\n--- Multi-Task Heads ---")
    mtl_heads = MultiTaskHeads(
        input_dim=input_dim,
        task_names=task_names,
        hidden_dims=[128, 64]
    )
    predictions = mtl_heads(x)
    print(f"Predictions:")
    for task, pred in predictions.items():
        print(f"  {task}: {pred.shape}")
    
    # Test stacked output
    stacked = mtl_heads(x, return_dict=False)
    print(f"Stacked output: {stacked.shape}")
    
    # Test uncertainty-weighted heads
    print("\n--- Uncertainty-Weighted Heads ---")
    uw_heads = UncertaintyWeightedHeads(
        input_dim=input_dim,
        task_names=task_names
    )
    predictions = uw_heads(x)
    weights = uw_heads.get_task_weights()
    print(f"Task weights: {weights}")
    print(f"Regularization term: {uw_heads.get_regularization_term().item():.4f}")
    
    # Count parameters
    num_params = sum(p.numel() for p in mtl_heads.parameters())
    print(f"\nMulti-Task Heads parameters: {num_params:,}")
    
    print("\n✓ Task Heads test passed!")
