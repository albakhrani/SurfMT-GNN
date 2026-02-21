#!/usr/bin/env python3
"""
Main Ensemble Training Script
==============================
Train temperature-aware multi-task GNN ensemble for surfactant property prediction.

This is the main entry point for training the model described in the paper:
"Temperature-Aware Multi-Task Graph Neural Networks for Interpretable 
Surfactant Property Prediction"

Usage:
    # Train full ensemble (10 models)
    python scripts/experiments/run_ensemble.py --epochs 500 --patience 80
    
    # Train single model for quick testing
    python scripts/experiments/run_ensemble.py --n_models 1 --epochs 100
    
    # Train without temperature (ablation baseline)
    python scripts/experiments/run_ensemble.py --no_temperature

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
from typing import Dict

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from src.data import (
    SurfProDataset,
    load_cv_splits,
    get_cv_splits,
    save_cv_splits,
    create_test_dataloader,
    TARGET_COLUMNS
)
from src.data.transforms import TargetScaler
from src.models.temperature_model import TemperatureAwareMTL
from src.training.ensemble_trainer import EnsembleTrainer, EnsembleConfig


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


def print_header():
    """Print training header."""
    print("\n" + "=" * 70)
    print("Temperature-Aware Multi-Task GNN for Surfactant Property Prediction")
    print("=" * 70)
    print("Paper: JCIM 2025")
    print("Author: Al-Futini Abdulhakim Nasser Ali")
    print("Supervisor: Prof. Huang Hexin")
    print("=" * 70)


def save_results(results: Dict, output_dir: Path):
    """Save results to JSON."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results = convert(results)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Temperature-Aware Multi-Task GNN Ensemble'
    )
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for data')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--temp_dim', type=int, default=64,
                        help='Temperature embedding dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--no_temperature', action='store_true',
                        help='Disable temperature modeling (ablation)')
    parser.add_argument('--no_uncertainty', action='store_true',
                        help='Disable uncertainty weighting')
    
    # Training arguments
    parser.add_argument('--n_models', type=int, default=10,
                        help='Number of models in ensemble')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum epochs per model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=80,
                        help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = args.experiment_name or 'ensemble'
        temp_suffix = '_no_temp' if args.no_temperature else '_with_temp'
        output_dir = PROJECT_ROOT / 'experiments' / f'{exp_name}{temp_suffix}_{timestamp}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\n" + "-" * 50)
    print("Loading Data")
    print("-" * 50)
    
    data_root = PROJECT_ROOT / args.data_root
    
    train_dataset = SurfProDataset(
        root=str(data_root),
        split='train',
        include_temperature=True
    )
    
    test_dataset = SurfProDataset(
        root=str(data_root),
        split='test',
        include_temperature=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Temperature statistics
    temp_stats = train_dataset.get_temperature_stats()
    print(f"Temperature range: {temp_stats['min']:.1f}°C - {temp_stats['max']:.1f}°C")
    
    # Load or create CV splits
    splits_path = data_root / 'splits' / 'cv_splits.json'
    
    if splits_path.exists():
        print(f"Loading CV splits from {splits_path}")
        splits = load_cv_splits(str(splits_path))
    else:
        print("Creating new CV splits...")
        splits = get_cv_splits(train_dataset, n_folds=10, seed=args.seed)
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        save_cv_splits(splits, str(splits_path))
    
    print(f"CV folds: {len(splits)}")
    
    # Model configuration
    print("\n" + "-" * 50)
    print("Model Configuration")
    print("-" * 50)
    
    model_kwargs = {
        'atom_dim': 34,
        'bond_dim': 12,
        'global_dim': 6,
        'hidden_dim': args.hidden_dim,
        'temp_embedding_dim': args.temp_dim,
        'num_layers': args.num_layers,
        'num_timesteps': 2,
        'dropout': args.dropout,
        'task_names': TARGET_COLUMNS,
        'head_hidden_dims': [128, 64],
        'use_global_features': True,
        'use_uncertainty_weighting': not args.no_uncertainty,
        'use_temperature': not args.no_temperature
    }
    
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Temperature embedding: {args.temp_dim}")
    print(f"GNN layers: {args.num_layers}")
    print(f"Use temperature: {not args.no_temperature}")
    print(f"Use uncertainty weighting: {not args.no_uncertainty}")
    
    # Count parameters
    temp_model = TemperatureAwareMTL(**model_kwargs)
    param_counts = temp_model.count_parameters()
    print(f"\nTotal parameters: {param_counts['total']:,}")
    del temp_model
    
    # Training configuration
    print("\n" + "-" * 50)
    print("Training Configuration")
    print("-" * 50)
    
    ensemble_config = EnsembleConfig(
        n_models=args.n_models,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        use_uncertainty_weighting=not args.no_uncertainty
    )
    
    print(f"Ensemble size: {args.n_models} models")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print(f"Batch size: {args.batch_size}")
    
    # Save configuration
    config = {
        'model': model_kwargs,
        'training': ensemble_config.to_dict(),
        'data': {
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'n_folds': len(splits)
        },
        'seed': args.seed,
        'device': device
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create ensemble trainer
    print("\n" + "-" * 50)
    print("Training Ensemble")
    print("-" * 50)
    
    trainer = EnsembleTrainer(
        model_class=TemperatureAwareMTL,
        model_kwargs=model_kwargs,
        config=ensemble_config,
        task_names=TARGET_COLUMNS,
        device=device,
        output_dir=str(output_dir)
    )
    
    # Train ensemble
    train_info = trainer.train_ensemble(
        dataset=train_dataset,
        splits=splits,
        scaler_class=TargetScaler,
        verbose=True
    )
    
    # Evaluate on test set
    print("\n" + "-" * 50)
    print("Evaluating Ensemble on Test Set")
    print("-" * 50)
    
    test_loader = create_test_dataloader(test_dataset, batch_size=args.batch_size)
    
    # Get test predictions
    predictions = trainer.predict_ensemble(test_loader, return_individual=True)
    
    # Get test targets and masks
    test_targets = []
    test_masks = []
    
    for data in test_dataset:
        test_targets.append(data.y.numpy())
        test_masks.append(data.mask.numpy())
    
    test_targets = np.stack(test_targets, axis=0)
    test_masks = np.stack(test_masks, axis=0)
    
    # Compute metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    results = {
        'ensemble_metrics': {},
        'individual_metrics': [],
        'uncertainties': {}
    }
    
    print("\n" + "=" * 60)
    print("ENSEMBLE TEST RESULTS")
    print("=" * 60)
    
    ensemble_preds = predictions['mean']
    ensemble_std = predictions['std']
    
    for i, task in enumerate(TARGET_COLUMNS):
        mask = test_masks[:, i] > 0.5
        n_valid = mask.sum()
        
        if n_valid < 2:
            continue
        
        y_true = test_targets[mask, i]
        y_pred = ensemble_preds[mask, i]
        y_std = ensemble_std[mask, i]
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        results['ensemble_metrics'][task] = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'mean_uncertainty': float(np.mean(y_std)),
            'n_samples': int(n_valid)
        }
        
        print(f"{task:12s}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, n={n_valid}")
    
    # Overall metrics
    valid_r2 = [m['r2'] for m in results['ensemble_metrics'].values() if m['n_samples'] > 1]
    valid_rmse = [m['rmse'] for m in results['ensemble_metrics'].values() if m['n_samples'] > 1]
    
    results['ensemble_metrics']['overall'] = {
        'r2_mean': float(np.mean(valid_r2)),
        'r2_std': float(np.std(valid_r2)),
        'rmse_mean': float(np.mean(valid_rmse)),
        'n_tasks': len(valid_r2)
    }
    
    print(f"\n{'Overall':12s}: R²={np.mean(valid_r2):.4f} ± {np.std(valid_r2):.4f}")
    
    # Save results
    save_results(results, output_dir)
    
    # Save predictions
    np.save(output_dir / 'test_predictions_mean.npy', ensemble_preds)
    np.save(output_dir / 'test_predictions_std.npy', ensemble_std)
    np.save(output_dir / 'test_targets.npy', test_targets)
    np.save(output_dir / 'test_masks.npy', test_masks)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"Models saved: {args.n_models}")
    print(f"Overall R²: {np.mean(valid_r2):.4f}")
    
    # Summary for paper
    print("\n" + "-" * 50)
    print("SUMMARY FOR PAPER")
    print("-" * 50)
    
    print(f"\nModel: Temperature-Aware MTL-GNN")
    print(f"Ensemble size: {args.n_models} models")
    print(f"Temperature modeling: {'Yes' if not args.no_temperature else 'No'}")
    print(f"\nTest Set Performance:")
    
    for task in TARGET_COLUMNS:
        if task in results['ensemble_metrics']:
            m = results['ensemble_metrics'][task]
            print(f"  {task}: R²={m['r2']:.3f}, RMSE={m['rmse']:.3f}")
    
    return results


if __name__ == '__main__':
    main()
