"""
Multi-Task Learning Model
=========================
MTL architecture with shared encoder and task-specific heads.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .attentive_fp import AttentiveFP
from .task_heads import TaskHead


class MTLSurfactantModel(nn.Module):
    """
    Multi-task model for surfactant property prediction.

    Parameters
    ----------
    encoder_config : dict
        Configuration for AttentiveFP encoder.
    head_config : dict
        Configuration for task-specific heads.
    task_names : list
        Names of prediction tasks.
    use_temperature : bool
        Whether to use temperature as auxiliary input.
    """

    TASKS = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']

    def __init__(
            self,
            encoder_config: Dict,
            head_config: Dict,
            task_names: Optional[List[str]] = None,
            use_temperature: bool = True
    ):
        super().__init__()

        self.task_names = task_names or self.TASKS
        self.use_temperature = use_temperature

        # Shared encoder
        self.encoder = AttentiveFP(**encoder_config)

        # Temperature embedding (if used)
        encoder_output_dim = encoder_config.get('hidden_dim', 96)  # CHANGED from 256

        if use_temperature:
            self.temp_embedding = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 32)
            )
            head_input_dim = encoder_output_dim + 32
        else:
            head_input_dim = encoder_output_dim

        # Task-specific heads
        self.heads = nn.ModuleDict({
            task: TaskHead(
                input_dim=head_input_dim,
                **head_config
            )
            for task in self.task_names
        })

    def forward(self, data) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Batched graph data.

        Returns
        -------
        dict
            Dictionary of predictions for each task.
        """
        # Get molecular embedding from encoder
        mol_embed = self.encoder(data)

        # Add temperature if available
        if self.use_temperature and hasattr(data, 'temperature'):
            temp_embed = self.temp_embedding(data.temperature)
            combined = torch.cat([mol_embed, temp_embed], dim=-1)
        else:
            combined = mol_embed

        # Task-specific predictions
        predictions = {}
        for task in self.task_names:
            predictions[task] = self.heads[task](combined)

        return predictions

    def get_embeddings(self, data) -> torch.Tensor:
        """Get molecular embeddings without task predictions."""
        return self.encoder(data)