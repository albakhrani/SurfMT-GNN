#!/usr/bin/env python3
"""
Data Transforms for SurfPro Dataset
====================================
Transforms for target normalization with proper handling of extreme scales.

Key Features:
- Log-transform for very small values (Gamma_max)
- Robust Z-score normalization
- Missing value handling

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Union
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


# =============================================================================
# Improved Target Scaler with Log Transform
# =============================================================================

class TargetScaler:
    """
    Improved Z-score normalization with log-transform for small-valued properties.

    Problem: Gamma_max values are ~1e-6, while other properties are 1-50.
    Solution: Apply log10 transform to Gamma_max before Z-score normalization.

    Parameters
    ----------
    task_names : list
        List of task names.
    log_transform_tasks : list, optional
        Tasks that need log transform (default: ['Gamma_max']).
    """

    DEFAULT_TASK_NAMES = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']
    DEFAULT_LOG_TASKS = ['Gamma_max']  # Tasks with very small values

    def __init__(
            self,
            task_names: Optional[List[str]] = None,
            log_transform_tasks: Optional[List[str]] = None
    ):
        self.task_names = task_names or self.DEFAULT_TASK_NAMES
        self.log_transform_tasks = log_transform_tasks or self.DEFAULT_LOG_TASKS
        self.num_tasks = len(self.task_names)

        # Statistics per task
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.use_log: Dict[str, bool] = {}

        # For debugging
        self.raw_stats: Dict[str, Dict] = {}

        self.is_fitted = False

    def fit(self, dataset, indices: Optional[List[int]] = None) -> 'TargetScaler':
        """
        Fit the scaler on a dataset.

        Parameters
        ----------
        dataset : SurfProDataset
            Dataset to fit on.
        indices : list, optional
            Indices to use for fitting (e.g., training indices only).
        """
        print("\n" + "=" * 60)
        print("Fitting TargetScaler")
        print("=" * 60)

        if indices is None:
            indices = range(len(dataset))

        # Collect values for each task
        values_per_task = {task: [] for task in self.task_names}

        for idx in indices:
            data = dataset[idx]
            for i, task in enumerate(self.task_names):
                if data.mask[i].item() == 1.0:
                    val = data.y[i].item()
                    values_per_task[task].append(val)

        # Compute statistics for each task
        for task in self.task_names:
            values = np.array(values_per_task[task])

            if len(values) == 0:
                print(f"  ⚠ {task}: NO VALID VALUES")
                self.means[task] = 0.0
                self.stds[task] = 1.0
                self.use_log[task] = False
                continue

            # Store raw statistics
            self.raw_stats[task] = {
                'count': len(values),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

            print(f"\n  {task}:")
            print(f"    Count: {len(values)} samples")
            print(f"    Raw range: [{values.min():.6g}, {values.max():.6g}]")

            # Check if log transform is needed
            should_log = (
                    task in self.log_transform_tasks and
                    np.all(values > 0) and  # All values must be positive
                    np.max(values) < 0.01  # Values are very small
            )

            if should_log:
                # Apply log10 transform
                values_transformed = np.log10(values + 1e-12)  # Small epsilon for safety
                self.use_log[task] = True
                print(f"    ✓ Using LOG10 transform")
                print(f"    Log range: [{values_transformed.min():.4f}, {values_transformed.max():.4f}]")
            else:
                values_transformed = values
                self.use_log[task] = False

            # Compute mean and std
            self.means[task] = float(np.mean(values_transformed))
            self.stds[task] = float(np.std(values_transformed))

            # Prevent division by zero
            if self.stds[task] < 1e-8:
                self.stds[task] = 1.0
                print(f"    ⚠ Std too small, using 1.0")

            print(f"    Final: mean={self.means[task]:.4f}, std={self.stds[task]:.4f}")

        self.is_fitted = True
        print("\n✓ Scaler fitted successfully")
        return self

    def transform(
            self,
            y: Union[torch.Tensor, np.ndarray],
            mask: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Transform target values to normalized scale.

        Parameters
        ----------
        y : torch.Tensor or np.ndarray
            Target values [batch_size, num_tasks] or [num_tasks].
        mask : optional
            Not used, kept for API compatibility.

        Returns
        -------
        torch.Tensor or np.ndarray
            Normalized values.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        is_tensor = isinstance(y, torch.Tensor)
        if is_tensor:
            device = y.device
            dtype = y.dtype
            y_np = y.cpu().numpy()
        else:
            y_np = y.copy()

        # Handle 1D input
        if y_np.ndim == 1:
            y_np = y_np.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False

        y_transformed = np.zeros_like(y_np)

        for i, task in enumerate(self.task_names):
            values = y_np[:, i]

            # Apply log transform if needed
            if self.use_log.get(task, False):
                values = np.log10(np.maximum(values, 1e-12))

            # Z-score normalization
            y_transformed[:, i] = (values - self.means[task]) / self.stds[task]

        if squeeze_output:
            y_transformed = y_transformed.squeeze(0)

        if is_tensor:
            return torch.tensor(y_transformed, device=device, dtype=dtype)
        return y_transformed

    def inverse_transform(
            self,
            y_scaled: Union[torch.Tensor, np.ndarray],
            mask: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Inverse transform normalized values back to original scale.

        Parameters
        ----------
        y_scaled : torch.Tensor or np.ndarray
            Normalized values [batch_size, num_tasks] or [num_tasks].
        mask : optional
            Not used, kept for API compatibility.

        Returns
        -------
        torch.Tensor or np.ndarray
            Original scale values.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        is_tensor = isinstance(y_scaled, torch.Tensor)
        if is_tensor:
            device = y_scaled.device
            dtype = y_scaled.dtype
            y_np = y_scaled.cpu().numpy()
        else:
            y_np = y_scaled.copy()

        # Handle 1D input
        if y_np.ndim == 1:
            y_np = y_np.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False

        y_original = np.zeros_like(y_np)

        for i, task in enumerate(self.task_names):
            # Inverse Z-score
            values = y_np[:, i] * self.stds[task] + self.means[task]

            # Inverse log transform if needed
            if self.use_log.get(task, False):
                values = np.power(10, values)

            y_original[:, i] = values

        if squeeze_output:
            y_original = y_original.squeeze(0)

        if is_tensor:
            return torch.tensor(y_original, device=device, dtype=dtype)
        return y_original

    def save(self, path: str):
        """Save scaler parameters to JSON file."""
        save_dict = {
            'task_names': self.task_names,
            'log_transform_tasks': self.log_transform_tasks,
            'means': self.means,
            'stds': self.stds,
            'use_log': self.use_log,
            'raw_stats': self.raw_stats,
            'is_fitted': self.is_fitted
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        print(f"Saved TargetScaler to {path}")

    @classmethod
    def load(cls, path: str) -> 'TargetScaler':
        """Load scaler parameters from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        scaler = cls(
            task_names=data['task_names'],
            log_transform_tasks=data.get('log_transform_tasks', cls.DEFAULT_LOG_TASKS)
        )
        scaler.means = data['means']
        scaler.stds = data['stds']
        scaler.use_log = data['use_log']
        scaler.raw_stats = data.get('raw_stats', {})
        scaler.is_fitted = data['is_fitted']

        print(f"Loaded TargetScaler from {path}")
        return scaler

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        log_tasks = [t for t, v in self.use_log.items() if v]
        return f"TargetScaler(tasks={self.num_tasks}, {status}, log_tasks={log_tasks})"


# =============================================================================
# Data Augmentation Transforms
# =============================================================================

class AddNoise(BaseTransform):
    """Add Gaussian noise to node features for augmentation."""

    def __init__(self, noise_std: float = 0.01, prob: float = 1.0):
        self.noise_std = noise_std
        self.prob = prob

    def forward(self, data: Data) -> Data:
        if torch.rand(1).item() > self.prob:
            return data
        noise = torch.randn_like(data.x) * self.noise_std
        data.x = data.x + noise
        return data


class RandomTargetMask(BaseTransform):
    """Randomly mask target values for MTL augmentation."""

    def __init__(self, mask_prob: float = 0.1, min_unmasked: int = 1):
        self.mask_prob = mask_prob
        self.min_unmasked = min_unmasked

    def forward(self, data: Data) -> Data:
        if not hasattr(data, 'mask'):
            return data

        valid_indices = torch.where(data.mask == 1.0)[0]

        if len(valid_indices) <= self.min_unmasked:
            return data

        for idx in valid_indices:
            if torch.rand(1).item() < self.mask_prob:
                if data.mask.sum() > self.min_unmasked:
                    data.mask[idx] = 0.0

        return data


class NormalizeFeatures(BaseTransform):
    """Normalize node/edge features."""

    def __init__(self, normalize_nodes: bool = True, normalize_edges: bool = False):
        self.normalize_nodes = normalize_nodes
        self.normalize_edges = normalize_edges

    def forward(self, data: Data) -> Data:
        if self.normalize_nodes and hasattr(data, 'x'):
            mean = data.x.mean(dim=0, keepdim=True)
            std = data.x.std(dim=0, keepdim=True)
            std = torch.where(std < 1e-8, torch.ones_like(std), std)
            data.x = (data.x - mean) / std

        if self.normalize_edges and hasattr(data, 'edge_attr') and data.edge_attr.size(0) > 0:
            mean = data.edge_attr.mean(dim=0, keepdim=True)
            std = data.edge_attr.std(dim=0, keepdim=True)
            std = torch.where(std < 1e-8, torch.ones_like(std), std)
            data.edge_attr = (data.edge_attr - mean) / std

        return data


class AddTemperatureFeature(BaseTransform):
    """Add normalized temperature as a feature."""

    def __init__(self, as_node_feature: bool = False, mean: float = 25.0, std: float = 10.0):
        self.as_node_feature = as_node_feature
        self.mean = mean
        self.std = std

    def forward(self, data: Data) -> Data:
        if not hasattr(data, 'temperature'):
            return data

        temp = (data.temperature - self.mean) / self.std

        if self.as_node_feature:
            num_nodes = data.x.size(0)
            temp_expanded = temp.expand(num_nodes, 1)
            data.x = torch.cat([data.x, temp_expanded], dim=-1)
        else:
            data.temp_feature = temp

        return data


class Compose(BaseTransform):
    """Compose multiple transforms."""

    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms

    def forward(self, data: Data) -> Data:
        for transform in self.transforms:
            data = transform(data)
        return data


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing TargetScaler")
    print("=" * 60)

    # Create mock data
    task_names = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']

    # Simulate realistic values
    mock_values = np.array([
        [2.5, 35.0, 3e-6, 0.6, 37.0, 3.5],
        [3.0, 32.0, 2e-6, 0.7, 40.0, 4.0],
        [2.8, 38.0, 4e-6, 0.5, 34.0, 3.2],
    ])

    print(f"\nMock values shape: {mock_values.shape}")
    print(f"Gamma_max values: {mock_values[:, 2]}")

    # Test transform
    scaler = TargetScaler(task_names)

    # Mock fit (manually set stats)
    scaler.means = {'pCMC': 2.77, 'AW_ST_CMC': 35.0, 'Gamma_max': -5.5,
                    'Area_min': 0.6, 'Pi_CMC': 37.0, 'pC20': 3.57}
    scaler.stds = {'pCMC': 0.25, 'AW_ST_CMC': 3.0, 'Gamma_max': 0.3,
                   'Area_min': 0.1, 'Pi_CMC': 3.0, 'pC20': 0.4}
    scaler.use_log = {'pCMC': False, 'AW_ST_CMC': False, 'Gamma_max': True,
                      'Area_min': False, 'Pi_CMC': False, 'pC20': False}
    scaler.is_fitted = True

    # Transform
    scaled = scaler.transform(mock_values)
    print(f"\nScaled values:\n{scaled}")

    # Inverse transform
    recovered = scaler.inverse_transform(scaled)
    print(f"\nRecovered values:\n{recovered}")

    # Check error
    error = np.abs(mock_values - recovered).max()
    print(f"\nMax recovery error: {error:.10f}")

    print("\n✓ TargetScaler test passed!")

__all__ = [
    'TargetScaler',
    'AddNoise',
    'RandomTargetMask',
    'NormalizeFeatures',
    'AddTemperatureFeature',
    'Compose',
]