#!/usr/bin/env python3
"""
Multi-Task GNN Training Script v2
=================================
Fixed version with:
- Proper target scaling (log-transform for Gamma_max)
- Correct metrics computation in original scale
- Better convergence settings

Usage:
    python scripts/training/train_mtl_v2.py --fold 0 --epochs 300
    python scripts/training/train_mtl_v2.py --fold -1 --epochs 300  # All folds

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import time

# Setup Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from torch_geometric.loader import DataLoader

from src.data import (
    SurfProDataset,
    load_cv_splits,
    get_cv_splits,
    create_dataloaders,
    create_test_dataloader,
    TARGET_COLUMNS
)
from src.data.transforms import TargetScaler
from src.models import SurfProMTL
from src.training.losses import MultiTaskLoss


# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        task_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute R², RMSE, MAE for each task."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    metrics = {}

    for i, task in enumerate(task_names):
        mask = masks[:, i] > 0.5
        n_valid = mask.sum()

        if n_valid < 2:
            metrics[task] = {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0, 'n': int(n_valid)}
            continue

        y_true = targets[mask, i]
        y_pred = predictions[mask, i]

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        metrics[task] = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'n': int(n_valid)
        }

    # Overall (average of valid tasks)
    valid_r2 = [m['r2'] for m in metrics.values() if m['n'] > 1]
    valid_rmse = [m['rmse'] for m in metrics.values() if m['n'] > 1]

    metrics['overall'] = {
        'r2': float(np.mean(valid_r2)) if valid_r2 else 0.0,
        'rmse': float(np.mean(valid_rmse)) if valid_rmse else 0.0,
        'n_tasks': len(valid_r2)
    }

    return metrics


# =============================================================================
# Training Loop
# =============================================================================

class MTLTrainer:
    """Trainer with proper target scaling."""

    def __init__(
            self,
            model: torch.nn.Module,
            scaler: TargetScaler,
            task_names: List[str],
            device: str = 'cuda',
            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            use_uncertainty: bool = True
    ):
        self.model = model.to(device)
        self.scaler = scaler
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.device = device

        # Loss function
        self.criterion = MultiTaskLoss(
            task_names=task_names,
            loss_type='mse',
            use_uncertainty_weighting=use_uncertainty
        )

        # Optimizer (include loss parameters if using uncertainty)
        params = list(model.parameters())
        if use_uncertainty:
            params += list(self.criterion.parameters())

        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
        )

        # History
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def _scale_targets(self, batch):
        """Scale targets in batch."""
        # Reshape from PyG concatenated format
        batch_size = batch.num_graphs
        y = batch.y.view(batch_size, self.num_tasks)
        mask = batch.mask.view(batch_size, self.num_tasks)

        # Scale targets
        y_scaled = self.scaler.transform(y.cpu().numpy())
        y_scaled = torch.tensor(y_scaled, dtype=y.dtype, device=y.device)

        return y_scaled, mask

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)

            # Get scaled targets
            y_scaled, mask = self._scale_targets(batch)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch)
            predictions = output['predictions']

            # Compute loss
            losses = self.criterion(predictions, y_scaled, mask)
            loss = losses['total']

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, compute_original_scale: bool = True) -> Dict:
        """Evaluate model."""
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        all_preds_scaled = []
        all_targets_original = []
        all_masks = []

        for batch in loader:
            batch = batch.to(self.device)

            # Get original and scaled targets
            batch_size = batch.num_graphs
            y_original = batch.y.view(batch_size, self.num_tasks)
            mask = batch.mask.view(batch_size, self.num_tasks)

            y_scaled = self.scaler.transform(y_original.cpu().numpy())
            y_scaled = torch.tensor(y_scaled, dtype=y_original.dtype, device=y_original.device)

            # Forward pass
            output = self.model(batch)
            predictions = output['predictions']

            # Loss (on scaled)
            losses = self.criterion(predictions, y_scaled, mask)
            total_loss += losses['total'].item()
            n_batches += 1

            # Collect for metrics
            all_preds_scaled.append(predictions.cpu().numpy())
            all_targets_original.append(y_original.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

        avg_loss = total_loss / n_batches

        # Concatenate
        all_preds_scaled = np.concatenate(all_preds_scaled, axis=0)
        all_targets_original = np.concatenate(all_targets_original, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        # Inverse transform predictions to original scale
        all_preds_original = self.scaler.inverse_transform(all_preds_scaled)

        # Compute metrics in original scale
        metrics = compute_metrics(
            all_preds_original,
            all_targets_original,
            all_masks,
            self.task_names
        )

        return {
            'loss': avg_loss,
            'metrics': metrics,
            'predictions': all_preds_original,
            'targets': all_targets_original,
            'masks': all_masks
        }

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 300,
            patience: int = 50,
            output_dir: Optional[Path] = None,
            verbose: bool = True
    ) -> Dict:
        """Full training loop."""

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        patience_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_results = self.evaluate(val_loader)
            val_loss = val_results['loss']
            val_metrics = val_results['metrics']

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # History
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_r2'].append(val_metrics['overall']['r2'])
            self.history['lr'].append(current_lr)

            # Best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0

                if output_dir:
                    self.save_checkpoint(output_dir / 'best_model.pt')
            else:
                patience_counter += 1

            # Logging
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch + 1:3d}/{epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"R²: {val_metrics['overall']['r2']:.4f} | "
                    f"LR: {current_lr:.2e}"
                )

            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"\n*** Early stopping at epoch {epoch + 1} ***")
                break

        total_time = time.time() - start_time

        if verbose:
            print(f"\nTraining complete in {total_time / 60:.1f} min")
            print(f"Best epoch: {self.best_epoch + 1}, Best val loss: {self.best_val_loss:.4f}")

        # Save final model
        if output_dir:
            self.save_checkpoint(output_dir / 'final_model.pt')
            self.save_history(output_dir / 'history.json')

        return dict(self.history)

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)

    def save_history(self, path: Path):
        """Save training history."""
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)


# =============================================================================
# Main Training Function
# =============================================================================

def train_fold(
        fold: int,
        train_dataset: SurfProDataset,
        test_dataset: SurfProDataset,
        splits: List,
        config: Dict,
        output_dir: Path
) -> Dict:
    """Train single fold."""

    print(f"\n{'=' * 70}")
    print(f"FOLD {fold + 1}/{len(splits)}")
    print('=' * 70)

    # Get indices
    train_idx, val_idx = splits[fold]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_dataset)}")

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, train_idx, val_idx,
        batch_size=config['batch_size'],
        num_workers=0
    )
    test_loader = create_test_dataloader(test_dataset, batch_size=config['batch_size'])

    # Fit scaler on training data
    print("\nFitting scaler on training data...")
    scaler = TargetScaler(task_names=TARGET_COLUMNS)
    scaler.fit(train_dataset, indices=train_idx)

    # Create model
    model = SurfProMTL(
        atom_dim=config['atom_dim'],
        bond_dim=config['bond_dim'],
        global_dim=config['global_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_timesteps=config['num_timesteps'],
        dropout=config['dropout'],
        task_names=TARGET_COLUMNS,
        head_hidden_dims=config['head_hidden_dims'],
        use_global_features=True,
        use_uncertainty_weighting=config['use_uncertainty']
    )

    print(f"\nModel parameters: {model.count_parameters()['total']:,}")

    # Create trainer
    fold_dir = output_dir / f'fold_{fold}'
    trainer = MTLTrainer(
        model=model,
        scaler=scaler,
        task_names=TARGET_COLUMNS,
        device=config['device'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        use_uncertainty=config['use_uncertainty']
    )

    # Train
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        patience=config['patience'],
        output_dir=fold_dir,
        verbose=True
    )

    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    trainer.load_checkpoint(fold_dir / 'best_model.pt')
    test_results = trainer.evaluate(test_loader)

    # Print test results
    print(f"\n{'=' * 50}")
    print(f"TEST RESULTS - Fold {fold + 1}")
    print('=' * 50)

    for task in TARGET_COLUMNS:
        m = test_results['metrics'][task]
        print(f"  {task:12s}: R²={m['r2']:7.4f}, RMSE={m['rmse']:10.4f}, n={m['n']}")

    overall = test_results['metrics']['overall']
    print(f"\n  {'Overall':12s}: R²={overall['r2']:7.4f}")

    # Save results
    scaler.save(str(fold_dir / 'scaler.json'))

    with open(fold_dir / 'test_results.json', 'w') as f:
        json.dump({
            'fold': fold,
            'best_epoch': trainer.best_epoch,
            'best_val_loss': trainer.best_val_loss,
            'test_loss': test_results['loss'],
            'test_metrics': test_results['metrics']
        }, f, indent=2)

    return {
        'fold': fold,
        'test_metrics': test_results['metrics'],
        'best_epoch': trainer.best_epoch
    }


def main():
    parser = argparse.ArgumentParser(description='MTL GNN Training v2')
    parser.add_argument('--fold', type=int, default=0, help='Fold to train (-1 for all)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_uncertainty', action='store_true')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Config
    config = {
        'atom_dim': 34,
        'bond_dim': 12,
        'global_dim': 6,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_timesteps': 2,
        'dropout': args.dropout,
        'head_hidden_dims': [128, 64],
        'lr': args.lr,
        'weight_decay': 1e-5,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'patience': args.patience,
        'use_uncertainty': not args.no_uncertainty,
        'device': args.device if torch.cuda.is_available() else 'cpu',
        'seed': args.seed
    }

    print("\n" + "=" * 70)
    print("Multi-Task GNN Training v2")
    print("=" * 70)
    print(f"Device: {config['device']}")
    print(f"Hidden dim: {config['hidden_dim']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Uncertainty weighting: {config['use_uncertainty']}")

    # Load data
    data_root = PROJECT_ROOT / 'data'
    train_dataset = SurfProDataset(root=str(data_root), split='train')
    test_dataset = SurfProDataset(root=str(data_root), split='test')

    print(f"\nDataset: Train={len(train_dataset)}, Test={len(test_dataset)}")

    # Load splits
    splits = load_cv_splits(str(data_root / 'splits' / 'cv_splits.json'))

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = PROJECT_ROOT / 'experiments' / f'mtl_v2_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Train
    folds_to_train = range(len(splits)) if args.fold == -1 else [args.fold]
    all_results = []

    for fold in folds_to_train:
        result = train_fold(fold, train_dataset, test_dataset, splits, config, output_dir)
        all_results.append(result)

    # Summary
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 70)

        summary = {}
        for task in TARGET_COLUMNS:
            r2_values = [r['test_metrics'][task]['r2'] for r in all_results]
            rmse_values = [r['test_metrics'][task]['rmse'] for r in all_results]

            summary[task] = {
                'r2_mean': float(np.mean(r2_values)),
                'r2_std': float(np.std(r2_values)),
                'rmse_mean': float(np.mean(rmse_values)),
                'rmse_std': float(np.std(rmse_values))
            }

            print(f"\n{task}:")
            print(f"  R²:   {summary[task]['r2_mean']:.4f} ± {summary[task]['r2_std']:.4f}")
            print(f"  RMSE: {summary[task]['rmse_mean']:.4f} ± {summary[task]['rmse_std']:.4f}")

        # Overall
        overall_r2 = [r['test_metrics']['overall']['r2'] for r in all_results]
        print(f"\nOverall R²: {np.mean(overall_r2):.4f} ± {np.std(overall_r2):.4f}")

        with open(output_dir / 'cv_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == '__main__':
    main()