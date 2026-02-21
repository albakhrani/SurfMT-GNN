#!/usr/bin/env python3
"""
Ensemble Trainer for Multi-Task GNN
====================================
Train multiple models and combine predictions for improved performance.

Key Features:
    - Train N models (one per CV fold)
    - Ensemble predictions by averaging
    - Uncertainty quantification via prediction variance
    - Calibration analysis

Based on literature showing ensembles improve GNN predictions:
    - Hödl et al. 2025: 10-model ensemble
    - Brozos et al. 2024: 40-model ensemble

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
from copy import deepcopy

from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =============================================================================
# Ensemble Configuration
# =============================================================================

@dataclass
class EnsembleConfig:
    """Configuration for ensemble training."""
    n_models: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 500
    batch_size: int = 32
    patience: int = 80
    min_delta: float = 1e-5
    gradient_clip: float = 1.0
    use_uncertainty_weighting: bool = True
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Single Model Trainer (Simplified)
# =============================================================================

class SingleModelTrainer:
    """Train a single model in the ensemble."""
    
    def __init__(
        self,
        model: nn.Module,
        task_names: List[str],
        device: str = 'cuda',
        lr: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.device = device
        
        # Import loss function
        from .losses import MultiTaskLoss
        
        self.criterion = MultiTaskLoss(
            task_names=task_names,
            loss_type='mse',
            use_uncertainty_weighting=True
        ).to(device)
        
        # Optimizer includes loss parameters
        params = list(model.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
        )
        
        self.best_val_loss = float('inf')
        self.best_state = None
        self.patience_counter = 0
    
    def _reshape_targets(self, batch):
        """Reshape targets from PyG batch format."""
        batch_size = batch.num_graphs
        y = batch.y.view(batch_size, self.num_tasks)
        mask = batch.mask.view(batch_size, self.num_tasks)
        return y, mask
    
    def train_epoch(self, train_loader: DataLoader, scaler) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            y, mask = self._reshape_targets(batch)
            
            # Scale targets
            y_scaled = scaler.transform(y.cpu().numpy())
            y_scaled = torch.tensor(y_scaled, dtype=y.dtype, device=y.device)
            
            self.optimizer.zero_grad()
            output = self.model(batch)
            predictions = output['predictions']
            
            losses = self.criterion(predictions, y_scaled, mask)
            loss = losses['total']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, scaler) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model."""
        self.model.eval()
        
        total_loss = 0.0
        n_batches = 0
        
        all_preds = []
        all_targets = []
        all_masks = []
        
        for batch in loader:
            batch = batch.to(self.device)
            y, mask = self._reshape_targets(batch)
            
            y_scaled = scaler.transform(y.cpu().numpy())
            y_scaled_t = torch.tensor(y_scaled, dtype=y.dtype, device=y.device)
            
            output = self.model(batch)
            predictions = output['predictions']
            
            losses = self.criterion(predictions, y_scaled_t, mask)
            total_loss += losses['total'].item()
            n_batches += 1
            
            # Inverse transform predictions
            preds_np = scaler.inverse_transform(predictions.cpu().numpy())
            
            all_preds.append(preds_np)
            all_targets.append(y.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
        
        avg_loss = total_loss / n_batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        
        return avg_loss, all_preds, all_targets, all_masks
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scaler,
        epochs: int = 500,
        patience: int = 80,
        verbose: bool = False
    ) -> Dict:
        """Full training loop."""
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, scaler)
            val_loss, _, _, _ = self.evaluate(val_loader, scaler)
            
            self.scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            # Early stopping
            if self.patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        return history


# =============================================================================
# Ensemble Trainer
# =============================================================================

class EnsembleTrainer:
    """
    Train and manage an ensemble of models.
    
    Parameters
    ----------
    model_class : type
        Model class to instantiate.
    model_kwargs : dict
        Arguments for model instantiation.
    config : EnsembleConfig
        Ensemble configuration.
    task_names : list
        List of task names.
    device : str
        Device to use.
    output_dir : str
        Output directory.
    """
    
    def __init__(
        self,
        model_class: type,
        model_kwargs: dict,
        config: EnsembleConfig,
        task_names: List[str],
        device: str = 'cuda',
        output_dir: str = 'experiments/ensemble'
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.config = config
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.device = device
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store trained models
        self.models: List[nn.Module] = []
        self.scalers = []
        self.histories = []
    
    def train_single_model(
        self,
        model_idx: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scaler,
        verbose: bool = True
    ) -> Tuple[nn.Module, Dict]:
        """Train a single model in the ensemble."""
        if verbose:
            print(f"\n--- Training Model {model_idx + 1}/{self.config.n_models} ---")
        
        # Create fresh model
        model = self.model_class(**self.model_kwargs)
        
        # Create trainer
        trainer = SingleModelTrainer(
            model=model,
            task_names=self.task_names,
            device=self.device,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            scaler=scaler,
            epochs=self.config.epochs,
            patience=self.config.patience,
            verbose=verbose
        )
        
        return trainer.model, history
    
    def train_ensemble(
        self,
        dataset,
        splits: List[Tuple[List[int], List[int]]],
        scaler_class,
        verbose: bool = True
    ) -> Dict:
        """
        Train full ensemble using CV splits.
        
        Parameters
        ----------
        dataset : SurfProDataset
            Full training dataset.
        splits : list
            List of (train_idx, val_idx) tuples.
        scaler_class : type
            Scaler class to use.
        verbose : bool
            Print progress.
            
        Returns
        -------
        dict
            Ensemble training results.
        """
        from .losses import MultiTaskLoss
        from ..data.dataset import create_dataloaders
        
        start_time = time.time()
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"Training Ensemble ({self.config.n_models} models)")
            print("=" * 70)
        
        self.models = []
        self.scalers = []
        self.histories = []
        
        n_models = min(self.config.n_models, len(splits))
        
        for i in range(n_models):
            train_idx, val_idx = splits[i]
            
            # Create dataloaders
            train_loader, val_loader = create_dataloaders(
                dataset, train_idx, val_idx,
                batch_size=self.config.batch_size,
                num_workers=0
            )
            
            # Fit scaler on this fold's training data
            scaler = scaler_class(task_names=self.task_names)
            scaler.fit(dataset, indices=train_idx)
            
            # Train model
            model, history = self.train_single_model(
                model_idx=i,
                train_loader=train_loader,
                val_loader=val_loader,
                scaler=scaler,
                verbose=verbose
            )
            
            self.models.append(model)
            self.scalers.append(scaler)
            self.histories.append(history)
            
            # Save individual model
            model_path = self.output_dir / f'model_{i}.pt'
            torch.save(model.state_dict(), model_path)
            
            scaler_path = self.output_dir / f'scaler_{i}.json'
            scaler.save(str(scaler_path))
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nEnsemble training complete in {total_time/60:.1f} minutes")
        
        # Save ensemble info
        info = {
            'n_models': len(self.models),
            'config': self.config.to_dict(),
            'training_time': total_time
        }
        
        with open(self.output_dir / 'ensemble_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        return info
    
    @torch.no_grad()
    def predict_ensemble(
        self,
        loader: DataLoader,
        return_individual: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Make ensemble predictions.
        
        Parameters
        ----------
        loader : DataLoader
            Data loader for prediction.
        return_individual : bool
            If True, return individual model predictions.
            
        Returns
        -------
        dict
            'mean': Ensemble mean predictions
            'std': Prediction standard deviation (uncertainty)
            'individual': Individual model predictions (if requested)
        """
        all_model_preds = []
        
        for model_idx, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            model.eval()
            
            preds_list = []
            
            for batch in loader:
                batch = batch.to(self.device)
                output = model(batch)
                predictions = output['predictions']
                
                # Inverse transform
                preds_np = scaler.inverse_transform(predictions.cpu().numpy())
                preds_list.append(preds_np)
            
            model_preds = np.concatenate(preds_list, axis=0)
            all_model_preds.append(model_preds)
        
        # Stack: [n_models, n_samples, n_tasks]
        all_preds = np.stack(all_model_preds, axis=0)
        
        # Compute ensemble statistics
        ensemble_mean = np.mean(all_preds, axis=0)  # [n_samples, n_tasks]
        ensemble_std = np.std(all_preds, axis=0)    # [n_samples, n_tasks]
        
        result = {
            'mean': ensemble_mean,
            'std': ensemble_std
        }
        
        if return_individual:
            result['individual'] = all_preds
        
        return result
    
    def evaluate_ensemble(
        self,
        loader: DataLoader,
        targets: np.ndarray,
        masks: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate ensemble performance.
        
        Returns
        -------
        dict
            Metrics for each task and overall.
        """
        predictions = self.predict_ensemble(loader)
        preds = predictions['mean']
        uncertainties = predictions['std']
        
        metrics = {}
        
        for i, task in enumerate(self.task_names):
            mask = masks[:, i] > 0.5
            n_valid = mask.sum()
            
            if n_valid < 2:
                metrics[task] = {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0, 'n': 0}
                continue
            
            y_true = targets[mask, i]
            y_pred = preds[mask, i]
            y_std = uncertainties[mask, i]
            
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            # Mean uncertainty
            mean_unc = np.mean(y_std)
            
            metrics[task] = {
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'mean_uncertainty': float(mean_unc),
                'n': int(n_valid)
            }
        
        # Overall
        valid_r2 = [m['r2'] for m in metrics.values() if m['n'] > 1]
        valid_rmse = [m['rmse'] for m in metrics.values() if m['n'] > 1]
        
        metrics['overall'] = {
            'r2': float(np.mean(valid_r2)) if valid_r2 else 0.0,
            'rmse': float(np.mean(valid_rmse)) if valid_rmse else 0.0,
            'n_tasks': len(valid_r2)
        }
        
        return metrics
    
    def save_ensemble(self, path: Optional[str] = None):
        """Save entire ensemble."""
        if path is None:
            path = self.output_dir
        else:
            path = Path(path)
        
        path.mkdir(parents=True, exist_ok=True)
        
        for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            torch.save(model.state_dict(), path / f'model_{i}.pt')
            scaler.save(str(path / f'scaler_{i}.json'))
        
        print(f"Saved ensemble to {path}")
    
    def load_ensemble(self, path: str, scaler_class):
        """Load ensemble from disk."""
        path = Path(path)
        
        # Find all model files
        model_files = sorted(path.glob('model_*.pt'))
        
        self.models = []
        self.scalers = []
        
        for model_file in model_files:
            # Load model
            model = self.model_class(**self.model_kwargs)
            model.load_state_dict(torch.load(model_file, map_location=self.device))
            model.to(self.device)
            self.models.append(model)
            
            # Load scaler
            idx = model_file.stem.split('_')[1]
            scaler_file = path / f'scaler_{idx}.json'
            scaler = scaler_class.load(str(scaler_file))
            self.scalers.append(scaler)
        
        print(f"Loaded {len(self.models)} models from {path}")


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("Ensemble Trainer module loaded successfully!")
    print("Use with: trainer = EnsembleTrainer(...)")
