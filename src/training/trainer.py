#!/usr/bin/env python3
"""
Multi-Task GNN Trainer
======================
Training utilities for the SurfPro multi-task GNN model.

Handles:
- Training loop with validation
- Early stopping
- Learning rate scheduling
- Checkpointing
- Metrics computation
- Proper handling of PyTorch Geometric batched tensors

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration dataclass."""

    # Optimizer settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = 'adamw'

    # Scheduler settings
    scheduler: str = 'plateau'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # Training settings
    epochs: int = 300
    batch_size: int = 32
    gradient_clip: float = 1.0

    # Early stopping settings
    early_stopping: bool = True
    patience: int = 30
    min_delta: float = 1e-4

    # Loss settings
    loss_type: str = 'mse'
    use_uncertainty_weighting: bool = False

    # Logging settings
    log_interval: int = 10
    save_every: int = 10

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.

    Monitors a metric and stops training if no improvement is seen
    for a specified number of epochs (patience).
    """

    def __init__(
            self,
            patience: int = 30,
            min_delta: float = 1e-4,
            mode: str = 'min'
    ):
        """
        Initialize early stopping.

        Parameters
        ----------
        patience : int
            Number of epochs to wait for improvement.
        min_delta : float
            Minimum change to qualify as an improvement.
        mode : str
            'min' for loss (lower is better), 'max' for metrics (higher is better).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Parameters
        ----------
        score : float
            Current metric value to monitor.

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


# =============================================================================
# Metrics Computation
# =============================================================================

class MetricsComputer:
    """
    Compute regression metrics for multi-task learning.

    Computes R², RMSE, and MAE for each task and overall.
    Handles missing values using masks.
    """

    def __init__(self, task_names: List[str]):
        """
        Initialize metrics computer.

        Parameters
        ----------
        task_names : List[str]
            List of task names.
        """
        self.task_names = task_names
        self.num_tasks = len(task_names)

    def compute(
            self,
            predictions: np.ndarray,
            targets: np.ndarray,
            masks: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for all tasks.

        Parameters
        ----------
        predictions : np.ndarray
            Model predictions, shape [N, num_tasks].
        targets : np.ndarray
            Ground truth targets, shape [N, num_tasks].
        masks : np.ndarray
            Binary masks for valid targets, shape [N, num_tasks].

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary containing metrics for each task and overall.
        """
        metrics = {}

        all_valid_preds = []
        all_valid_targets = []

        for i, task_name in enumerate(self.task_names):
            # Get valid samples for this task
            mask = masks[:, i] > 0.5
            n_valid = mask.sum()

            if n_valid == 0:
                # No valid samples for this task
                metrics[task_name] = {
                    'r2': 0.0,
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'n_samples': 0
                }
                continue

            # Extract valid predictions and targets
            task_preds = predictions[mask, i]
            task_targets = targets[mask, i]

            # Compute metrics
            if n_valid > 1:
                r2 = r2_score(task_targets, task_preds)
            else:
                r2 = 0.0

            rmse = np.sqrt(mean_squared_error(task_targets, task_preds))
            mae = mean_absolute_error(task_targets, task_preds)

            metrics[task_name] = {
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'n_samples': int(n_valid)
            }

            # Collect for overall metrics
            all_valid_preds.extend(task_preds.tolist())
            all_valid_targets.extend(task_targets.tolist())

        # Compute overall metrics
        if len(all_valid_preds) > 1:
            all_valid_preds = np.array(all_valid_preds)
            all_valid_targets = np.array(all_valid_targets)

            metrics['overall'] = {
                'r2': float(r2_score(all_valid_targets, all_valid_preds)),
                'rmse': float(np.sqrt(mean_squared_error(all_valid_targets, all_valid_preds))),
                'mae': float(mean_absolute_error(all_valid_targets, all_valid_preds)),
                'n_samples': len(all_valid_preds)
            }
        else:
            metrics['overall'] = {
                'r2': 0.0,
                'rmse': float('inf'),
                'mae': float('inf'),
                'n_samples': len(all_valid_preds)
            }

        return metrics


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """
    Trainer for multi-task GNN model.

    Handles the complete training pipeline including:
    - Training loop with gradient clipping
    - Validation with metrics computation
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Training history logging

    IMPORTANT: This trainer properly handles PyTorch Geometric's batching,
    which concatenates y and mask tensors into 1D arrays.
    """

    def __init__(
            self,
            model: nn.Module,
            config: TrainingConfig,
            task_names: List[str],
            device: str = 'cuda',
            output_dir: str = './outputs',
            scaler: Optional[Any] = None
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        model : nn.Module
            The model to train (SurfProMTL).
        config : TrainingConfig
            Training configuration.
        task_names : List[str]
            Names of the prediction tasks.
        device : str
            Device to use for training ('cuda' or 'cpu').
        output_dir : str
            Directory to save checkpoints and logs.
        scaler : Optional[TargetScaler]
            Target scaler for inverse transforming predictions.
        """
        self.model = model
        self.config = config
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.scaler = scaler

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup components
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.metrics_computer = MetricsComputer(task_names)

        # Setup early stopping
        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.patience,
                min_delta=config.min_delta,
                mode='min'
            )
        else:
            self.early_stopping = None

        # Initialize training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.history = defaultdict(list)

    def _setup_criterion(self) -> nn.Module:
        """Setup loss function."""
        from src.training.losses import MultiTaskLoss

        return MultiTaskLoss(
            task_names=self.task_names,
            loss_fn=self.config.loss_type
            
        )

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on config."""
        optimizer_name = self.config.optimizer.lower()

        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _setup_scheduler(self) -> Optional[Any]:
        """Setup learning rate scheduler based on config."""
        scheduler_name = self.config.scheduler.lower()

        if scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr,
                
            )
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.scheduler_min_lr
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.5
            )
        elif scheduler_name == 'none' or scheduler_name is None:
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def _reshape_batch_targets(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape y and mask from PyTorch Geometric batch format.

        PyTorch Geometric concatenates node/graph-level attributes when
        creating batches. For graph-level targets (y) and masks, this means
        they get concatenated from [num_tasks] per graph to
        [batch_size * num_tasks] total.

        This method reshapes them back to [batch_size, num_tasks] for
        proper loss computation.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batched graph data from PyTorch Geometric DataLoader.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - y: Reshaped targets with shape [batch_size, num_tasks]
            - mask: Reshaped masks with shape [batch_size, num_tasks]
        """
        batch_size = batch.num_graphs

        # Reshape from [batch_size * num_tasks] to [batch_size, num_tasks]
        y = batch.y.view(batch_size, self.num_tasks)
        mask = batch.mask.view(batch_size, self.num_tasks)

        return y, mask

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.

        Returns
        -------
        Dict[str, float]
            Dictionary of average training losses.
        """
        self.model.train()

        total_losses = defaultdict(float)
        num_batches = 0

        for batch in train_loader:
            # Move batch to device
            batch = batch.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(batch)
            predictions = output['predictions']

            # Reshape targets from PyTorch Geometric batch format
            # This is the critical fix for the tensor shape mismatch
            y, mask = self._reshape_batch_targets(batch)

            # Compute loss
            losses = self.criterion(predictions, y, mask)

            # Backward pass
            losses['total'].backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            # Update weights
            self.optimizer.step()

            # Accumulate losses
            for key, value in losses.items():
                total_losses[key] += value.item()
            num_batches += 1

        # Compute average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}

        return avg_losses

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, Any]:
        """
        Validate on validation/test set.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'losses': Average losses
            - 'metrics': Per-task and overall metrics
            - 'predictions': All predictions
            - 'targets': All targets
            - 'masks': All masks
        """
        self.model.eval()

        total_losses = defaultdict(float)
        num_batches = 0

        all_predictions = []
        all_targets = []
        all_masks = []

        for batch in val_loader:
            # Move batch to device
            batch = batch.to(self.device)

            # Forward pass
            output = self.model(batch)
            predictions = output['predictions']

            # Reshape targets from PyTorch Geometric batch format
            y, mask = self._reshape_batch_targets(batch)

            # Compute loss
            losses = self.criterion(predictions, y, mask)

            # Accumulate losses
            for key, value in losses.items():
                total_losses[key] += value.item()
            num_batches += 1

            # Collect predictions and targets for metrics
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

        # Compute average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}

        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        # Inverse transform if scaler is provided (for metrics in original scale)
        if self.scaler is not None:
            all_predictions_original = self.scaler.inverse_transform(torch.from_numpy(all_predictions)).numpy()
            all_targets_original = self.scaler.inverse_transform(torch.from_numpy(all_targets)).numpy()
        else:
            all_predictions_original = all_predictions
            all_targets_original = all_targets

        # Compute metrics
        metrics = self.metrics_computer.compute(
            all_predictions_original,
            all_targets_original,
            all_masks
        )

        return {
            'losses': avg_losses,
            'metrics': metrics,
            'predictions': all_predictions_original,
            'targets': all_targets_original,
            'masks': all_masks
        }

    def train(
            self,
            train_loader,
            val_loader,
            verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        Dict[str, List[float]]
            Training history containing losses and metrics per epoch.
        """
        if verbose:
            print("\n" + "=" * 70)
            print("Starting Training")
            print("=" * 70)
            print(f"Epochs: {self.config.epochs}")
            print(f"Batch size: {self.config.batch_size}")
            print(f"Learning rate: {self.config.learning_rate}")
            print(f"Device: {self.device}")
            print(f"Output dir: {self.output_dir}")
            print("=" * 70)

        training_start_time = time.time()

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training phase
            train_losses = self.train_epoch(train_loader)

            # Validation phase
            val_results = self.validate(val_loader)
            val_losses = val_results['losses']
            val_metrics = val_results['metrics']

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()

            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['val_r2'].append(val_metrics['overall']['r2'])
            self.history['val_rmse'].append(val_metrics['overall']['rmse'])
            self.history['lr'].append(current_lr)

            # Record per-task metrics
            for task_name in self.task_names:
                self.history[f'val_r2_{task_name}'].append(val_metrics[task_name]['r2'])
                self.history[f'val_rmse_{task_name}'].append(val_metrics[task_name]['rmse'])

            # Check for best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pt')

            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(val_losses['total']):
                    if verbose:
                        print(f"\n*** Early stopping triggered at epoch {epoch + 1} ***")
                        print(f"*** No improvement for {self.config.patience} epochs ***")
                    break

            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start_time
                print(
                    f"Epoch {epoch + 1:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_losses['total']:.4f} | "
                    f"Val Loss: {val_losses['total']:.4f} | "
                    f"Val R²: {val_metrics['overall']['r2']:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                )

        # Training complete
        total_time = time.time() - training_start_time

        if verbose:
            print("\n" + "=" * 70)
            print("Training Complete!")
            print("=" * 70)
            print(f"Total time: {total_time / 60:.1f} minutes")
            print(f"Best epoch: {self.best_epoch + 1}")
            print(f"Best val loss: {self.best_val_loss:.4f}")
            print("=" * 70)

        # Save final checkpoint and history
        self.save_checkpoint('final_model.pt')
        self.save_history()

        return dict(self.history)

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.

        Parameters
        ----------
        filename : str
            Name of the checkpoint file.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config.to_dict(),
            'task_names': self.task_names,
            'num_tasks': self.num_tasks
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, path: Union[str, Path]):
        """
        Load model checkpoint.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the checkpoint file.

        Returns
        -------
        dict
            Loaded checkpoint dictionary.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)

        return checkpoint

    def save_history(self):
        """Save training history to JSON file."""
        history_path = self.output_dir / 'training_history.json'

        # Convert numpy types to Python types for JSON serialization
        history_dict = {}
        for key, values in self.history.items():
            history_dict[key] = [float(v) if isinstance(v, (np.floating, float)) else int(v)
                                 for v in values]

        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def get_best_metrics(self) -> Dict[str, float]:
        """Get metrics from the best epoch."""
        if self.best_epoch < len(self.history['val_r2']):
            return {
                'best_epoch': self.best_epoch + 1,
                'best_val_loss': self.best_val_loss,
                'best_val_r2': self.history['val_r2'][self.best_epoch],
                'best_val_rmse': self.history['val_rmse'][self.best_epoch]
            }
        return {}


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'TrainingConfig',
    'Trainer',
    'EarlyStopping',
    'MetricsComputer'
]
