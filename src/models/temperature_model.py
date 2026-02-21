#!/usr/bin/env python3
"""
Temperature-Aware Multi-Task GNN for Surfactant Property Prediction
====================================================================
CORE NOVELTY: First model to predict all 6 surfactant properties 
as a function of temperature.

Key Features:
    - Temperature encoding (normalized + MLP projection)
    - Fusion with molecular graph embedding
    - Temperature-conditioned property prediction
    - Support for attention extraction

Scientific Justification:
    - Surfactant properties are highly temperature-dependent
    - CMC typically decreases with temperature for ionics
    - γCMC and other properties vary significantly with T
    - Industrial applications need predictions at various temperatures

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import Dict, Optional, List, Tuple

from .attentive_fp import AttentiveFPEncoder
from .task_heads import MultiTaskHeads, UncertaintyWeightedHeads


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_TASK_NAMES = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']


# =============================================================================
# Temperature Encoder
# =============================================================================

class TemperatureEncoder(nn.Module):
    """
    Encode temperature as a learnable embedding.
    
    Architecture:
        Temperature (scalar) → Normalize → MLP → Temperature Embedding
    
    Parameters
    ----------
    output_dim : int
        Output embedding dimension.
    hidden_dim : int
        Hidden layer dimension.
    temp_mean : float
        Mean temperature for normalization (default: 25°C).
    temp_std : float
        Standard deviation for normalization (default: 10°C).
    dropout : float
        Dropout probability.
    """
    
    def __init__(
        self,
        output_dim: int = 64,
        hidden_dim: int = 32,
        temp_mean: float = 25.0,
        temp_std: float = 10.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.temp_mean = temp_mean
        self.temp_std = temp_std
        
        # MLP to project normalized temperature
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        Encode temperature.
        
        Parameters
        ----------
        temperature : torch.Tensor
            Temperature values [batch_size] or [batch_size, 1].
            
        Returns
        -------
        torch.Tensor
            Temperature embeddings [batch_size, output_dim].
        """
        # Ensure correct shape
        if temperature.dim() == 1:
            temperature = temperature.unsqueeze(-1)
        
        # Normalize
        temp_normalized = (temperature - self.temp_mean) / self.temp_std
        
        # Encode
        return self.encoder(temp_normalized)


# =============================================================================
# Temperature-Aware Multi-Task GNN
# =============================================================================

class TemperatureAwareMTL(nn.Module):
    """
    Temperature-Aware Multi-Task GNN for Surfactant Property Prediction.
    
    NOVELTY: First model to predict all 6 surfactant properties as a 
    function of both molecular structure AND temperature.
    
    Architecture:
        Molecular Graph → AttentiveFP Encoder → Graph Embedding
        Temperature → Temperature Encoder → Temp Embedding
        [Graph Embedding ⊕ Temp Embedding ⊕ Global Features] → Task Heads → Predictions
    
    Parameters
    ----------
    atom_dim : int
        Input atom feature dimension.
    bond_dim : int
        Input bond feature dimension.
    global_dim : int
        Global molecular descriptor dimension.
    hidden_dim : int
        Hidden dimension for encoder.
    temp_embedding_dim : int
        Temperature embedding dimension.
    num_layers : int
        Number of GNN layers.
    num_timesteps : int
        Number of readout timesteps.
    dropout : float
        Dropout probability.
    task_names : list
        List of task names.
    head_hidden_dims : list
        Hidden dimensions for prediction heads.
    use_global_features : bool
        Whether to use global molecular descriptors.
    use_uncertainty_weighting : bool
        Whether to use uncertainty-based task weighting.
    use_temperature : bool
        Whether to use temperature (can disable for ablation).
    """
    
    def __init__(
        self,
        atom_dim: int = 34,
        bond_dim: int = 12,
        global_dim: int = 6,
        hidden_dim: int = 256,
        temp_embedding_dim: int = 64,
        num_layers: int = 3,
        num_timesteps: int = 2,
        dropout: float = 0.2,
        task_names: Optional[List[str]] = None,
        head_hidden_dims: List[int] = [128, 64],
        use_global_features: bool = True,
        use_uncertainty_weighting: bool = True,
        use_temperature: bool = True
    ):
        super().__init__()
        
        self.task_names = task_names or DEFAULT_TASK_NAMES
        self.num_tasks = len(self.task_names)
        self.use_global_features = use_global_features
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_temperature = use_temperature
        self.hidden_dim = hidden_dim
        self.temp_embedding_dim = temp_embedding_dim
        
        # Graph encoder (AttentiveFP)
        self.encoder = AttentiveFPEncoder(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )
        
        # Temperature encoder
        if use_temperature:
            self.temp_encoder = TemperatureEncoder(
                output_dim=temp_embedding_dim,
                hidden_dim=32,
                dropout=dropout
            )
        else:
            self.temp_encoder = None
            temp_embedding_dim = 0
        
        # Global feature projection
        if use_global_features:
            self.global_projection = nn.Sequential(
                nn.Linear(global_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, hidden_dim // 4)
            )
            global_proj_dim = hidden_dim // 4
        else:
            self.global_projection = None
            global_proj_dim = 0
        
        # Calculate total input dimension for heads
        head_input_dim = hidden_dim + temp_embedding_dim + global_proj_dim
        
        # Fusion layer (combine all features)
        self.fusion = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Multi-task prediction heads
        if use_uncertainty_weighting:
            self.task_heads = UncertaintyWeightedHeads(
                input_dim=hidden_dim,
                task_names=self.task_names,
                hidden_dims=head_hidden_dims,
                dropout=dropout
            )
        else:
            self.task_heads = MultiTaskHeads(
                input_dim=hidden_dim,
                task_names=self.task_names,
                hidden_dims=head_hidden_dims,
                dropout=dropout,
                share_bottom=False
            )
        
        # Store configuration
        self.config = {
            'atom_dim': atom_dim,
            'bond_dim': bond_dim,
            'global_dim': global_dim,
            'hidden_dim': hidden_dim,
            'temp_embedding_dim': temp_embedding_dim,
            'num_layers': num_layers,
            'num_timesteps': num_timesteps,
            'dropout': dropout,
            'task_names': self.task_names,
            'head_hidden_dims': head_hidden_dims,
            'use_global_features': use_global_features,
            'use_uncertainty_weighting': use_uncertainty_weighting,
            'use_temperature': use_temperature
        }
    
    def enable_attention_extraction(self, enable: bool = True):
        """Enable attention extraction for interpretability."""
        self.encoder.enable_attention_extraction(enable)
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get attention weights from encoder."""
        return self.encoder.get_all_attention_weights()
    
    def forward(
        self,
        batch: Batch,
        return_embedding: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batched graph data with:
            - x: Node features
            - edge_index: Edge connectivity
            - edge_attr: Edge features
            - batch: Batch assignment
            - temperature: Temperature values [batch_size]
            - global_features (optional): Global molecular descriptors
        return_embedding : bool
            If True, also return embeddings.
            
        Returns
        -------
        dict
            Dictionary of outputs including predictions.
        """
        # Encode molecular graph
        graph_embedding, node_embedding = self.encoder(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )
        
        # Build combined embedding
        embeddings_to_concat = [graph_embedding]
        
        # Add temperature embedding
        if self.use_temperature and hasattr(batch, 'temperature'):
            temp_embedding = self.temp_encoder(batch.temperature)
            embeddings_to_concat.append(temp_embedding)
        
        # Add global features
        if self.use_global_features and hasattr(batch, 'global_features'):
            global_feat = self.global_projection(batch.global_features)
            embeddings_to_concat.append(global_feat)
        
        # Concatenate all embeddings
        combined_embedding = torch.cat(embeddings_to_concat, dim=-1)
        
        # Fusion
        fused_embedding = self.fusion(combined_embedding)
        
        # Get predictions from task heads
        predictions = self.task_heads(fused_embedding)
        
        # Stack predictions into tensor
        pred_tensor = torch.cat(
            [predictions[task] for task in self.task_names],
            dim=-1
        )
        
        output = {
            'predictions': pred_tensor,
            'task_predictions': predictions
        }
        
        if return_embedding:
            output['graph_embedding'] = graph_embedding
            output['fused_embedding'] = fused_embedding
            output['node_embedding'] = node_embedding
            if self.use_temperature and hasattr(batch, 'temperature'):
                output['temp_embedding'] = temp_embedding
        
        return output
    
    def predict(self, batch: Batch) -> torch.Tensor:
        """Simple prediction interface."""
        self.eval()
        with torch.no_grad():
            output = self.forward(batch)
        return output['predictions']
    
    def get_task_weights(self) -> Optional[Dict[str, float]]:
        """Get task weights if using uncertainty weighting."""
        if self.use_uncertainty_weighting:
            return self.task_heads.get_task_weights()
        return None
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = sum(p.numel() for p in self.task_heads.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        
        temp_params = 0
        if self.temp_encoder is not None:
            temp_params = sum(p.numel() for p in self.temp_encoder.parameters())
        
        global_params = 0
        if self.global_projection is not None:
            global_params = sum(p.numel() for p in self.global_projection.parameters())
        
        total = encoder_params + head_params + fusion_params + temp_params + global_params
        
        return {
            'encoder': encoder_params,
            'temperature_encoder': temp_params,
            'global_projection': global_params,
            'fusion': fusion_params,
            'task_heads': head_params,
            'total': total
        }
    
    @classmethod
    def from_config(cls, config: dict) -> 'TemperatureAwareMTL':
        """Create model from configuration."""
        return cls(**config)
    
    def get_config(self) -> dict:
        """Get model configuration."""
        return self.config.copy()


# =============================================================================
# Model Factory
# =============================================================================

def create_temperature_model(
    use_temperature: bool = True,
    use_uncertainty: bool = True,
    **kwargs
) -> TemperatureAwareMTL:
    """
    Factory function for temperature-aware models.
    
    Parameters
    ----------
    use_temperature : bool
        Whether to include temperature encoding.
    use_uncertainty : bool
        Whether to use uncertainty weighting.
    **kwargs
        Additional model arguments.
    """
    return TemperatureAwareMTL(
        use_temperature=use_temperature,
        use_uncertainty_weighting=use_uncertainty,
        **kwargs
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    from torch_geometric.data import Data, Batch
    
    print("=" * 60)
    print("Testing Temperature-Aware MTL Model")
    print("=" * 60)
    
    # Create model
    model = TemperatureAwareMTL(
        atom_dim=34,
        bond_dim=12,
        global_dim=6,
        hidden_dim=256,
        temp_embedding_dim=64,
        num_layers=3,
        num_timesteps=2,
        dropout=0.2,
        use_global_features=True,
        use_uncertainty_weighting=True,
        use_temperature=True
    )
    
    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # Create dummy batch with temperature
    print("\nCreating dummy batch with temperature...")
    data_list = []
    temperatures = [20.0, 25.0, 30.0, 35.0]  # Different temperatures
    
    for i, temp in enumerate(temperatures):
        num_nodes = 20 + i * 5
        num_edges = num_nodes * 2
        
        data = Data(
            x=torch.randn(num_nodes, 34),
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            edge_attr=torch.randn(num_edges, 12),
            global_features=torch.randn(1, 6),
            temperature=torch.tensor([temp]),
            y=torch.randn(6),
            mask=torch.ones(6)
        )
        data_list.append(data)
    
    batch = Batch.from_data_list(data_list)
    
    # Need to reshape temperature for batch
    batch.temperature = torch.tensor(temperatures)
    
    print(f"Batch: {batch}")
    print(f"Temperatures: {batch.temperature}")
    
    # Forward pass
    print("\nForward pass...")
    model.eval()
    model.enable_attention_extraction(True)
    
    with torch.no_grad():
        output = model(batch, return_embedding=True)
    
    print(f"\nOutput:")
    print(f"  predictions shape: {output['predictions'].shape}")
    print(f"  graph_embedding shape: {output['graph_embedding'].shape}")
    print(f"  fused_embedding shape: {output['fused_embedding'].shape}")
    print(f"  temp_embedding shape: {output['temp_embedding'].shape}")
    
    # Get attention weights
    attention = model.get_attention_weights()
    print(f"\nAttention weights:")
    for name, weights in attention.items():
        print(f"  {name}: {weights.shape}")
    
    # Test without temperature (ablation)
    print("\n" + "-" * 60)
    print("Testing WITHOUT temperature (ablation baseline)")
    print("-" * 60)
    
    model_no_temp = TemperatureAwareMTL(
        use_temperature=False,
        use_uncertainty_weighting=True
    )
    
    print(f"Parameters without temp: {model_no_temp.count_parameters()['total']:,}")
    
    with torch.no_grad():
        output_no_temp = model_no_temp(batch)
    
    print(f"Predictions shape: {output_no_temp['predictions'].shape}")
    
    print("\n✓ Temperature-Aware MTL Model test passed!")
