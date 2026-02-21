#!/usr/bin/env python3
"""
Loss Functions for Multi-Task Learning
======================================
Custom loss functions with mask support and uncertainty weighting.

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class MaskedMSELoss(nn.Module):
    """MSE loss with mask support for missing values."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        squared_error = (pred - target) ** 2
        masked_error = squared_error * mask

        if self.reduction == 'none':
            return masked_error
        elif self.reduction == 'sum':
            return masked_error.sum()
        else:  # mean
            num_valid = mask.sum()
            if num_valid > 0:
                return masked_error.sum() / num_valid
            else:
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)


class MaskedMAELoss(nn.Module):
    """MAE loss with mask support for missing values."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        abs_error = torch.abs(pred - target)
        masked_error = abs_error * mask

        if self.reduction == 'none':
            return masked_error
        elif self.reduction == 'sum':
            return masked_error.sum()
        else:
            num_valid = mask.sum()
            if num_valid > 0:
                return masked_error.sum() / num_valid
            else:
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)


class MaskedHuberLoss(nn.Module):
    """Huber loss with mask support."""

    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        abs_error = torch.abs(pred - target)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        huber = 0.5 * quadratic ** 2 + self.delta * linear
        masked_huber = huber * mask

        if self.reduction == 'none':
            return masked_huber
        elif self.reduction == 'sum':
            return masked_huber.sum()
        else:
            num_valid = mask.sum()
            if num_valid > 0:
                return masked_huber.sum() / num_valid
            else:
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with optional uncertainty weighting.

    Parameters
    ----------
    task_names : list
        List of task names.
    loss_type : str
        Base loss type: 'mse', 'mae', or 'huber'.
    use_uncertainty_weighting : bool
        If True, learn task weights automatically using
        homoscedastic uncertainty (Kendall et al., 2018).
    """

    def __init__(
            self,
            task_names: List[str],
            loss_type: str = 'mse',
            use_uncertainty_weighting: bool = False,
            task_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()

        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.loss_type = loss_type
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Base loss function
        if loss_type == 'mse':
            self.base_loss = MaskedMSELoss(reduction='none')
        elif loss_type == 'mae':
            self.base_loss = MaskedMAELoss(reduction='none')
        elif loss_type == 'huber':
            self.base_loss = MaskedHuberLoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Task weights (fixed or learnable)
        if use_uncertainty_weighting:
            # Learnable log variance for each task
            # Using nn.Parameter directly (will be moved with .to(device))
            self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))
        else:
            self.register_parameter('log_vars', None)
            # Fixed task weights
            if task_weights is None:
                self.task_weights = {name: 1.0 for name in task_names}
            else:
                self.task_weights = task_weights

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predictions [batch_size, num_tasks].
        target : torch.Tensor
            Targets [batch_size, num_tasks].
        mask : torch.Tensor
            Mask [batch_size, num_tasks].

        Returns
        -------
        dict
            Dictionary with 'total' loss and per-task losses.
        """
        device = pred.device
        dtype = pred.dtype

        losses = {}
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)

        for i, task_name in enumerate(self.task_names):
            # Extract task-specific data
            task_pred = pred[:, i]
            task_target = target[:, i]
            task_mask = mask[:, i]

            # Compute raw loss for this task
            task_loss_raw = self.base_loss(
                task_pred.unsqueeze(-1),
                task_target.unsqueeze(-1),
                task_mask.unsqueeze(-1)
            )

            # Average over valid samples
            num_valid = task_mask.sum()
            if num_valid > 0:
                task_loss = task_loss_raw.sum() / num_valid
            else:
                task_loss = torch.tensor(0.0, device=device, dtype=dtype)

            losses[task_name] = task_loss

            # Apply weighting
            if self.use_uncertainty_weighting and self.log_vars is not None:
                # Uncertainty weighting: precision = exp(-log_var)
                log_var = self.log_vars[i]  # Already on correct device
                precision = torch.exp(-log_var)

                # Loss = 0.5 * precision * L + 0.5 * log_var
                weighted_loss = 0.5 * precision * task_loss + 0.5 * log_var
                total_loss = total_loss + weighted_loss
            else:
                # Fixed weighting
                weight = self.task_weights.get(task_name, 1.0)
                total_loss = total_loss + weight * task_loss

        losses['total'] = total_loss

        return losses

    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights."""
        if self.use_uncertainty_weighting and self.log_vars is not None:
            weights = {}
            for i, task_name in enumerate(self.task_names):
                # Weight = 0.5 * exp(-log_var)
                weights[task_name] = (0.5 * torch.exp(-self.log_vars[i])).item()
            return weights
        else:
            return self.task_weights.copy()


# Convenience function
def create_loss_function(
        task_names: List[str],
        loss_type: str = 'mse',
        use_uncertainty: bool = False,
        task_weights: Optional[Dict[str, float]] = None
) -> nn.Module:
    """Factory function to create loss functions."""
    return MultiTaskLoss(
        task_names=task_names,
        loss_type=loss_type,
        use_uncertainty_weighting=use_uncertainty,
        task_weights=task_weights
    )


__all__ = [
    'MaskedMSELoss',
    'MaskedMAELoss',
    'MaskedHuberLoss',
    'MultiTaskLoss',
    'create_loss_function'
]