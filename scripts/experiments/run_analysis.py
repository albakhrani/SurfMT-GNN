#!/usr/bin/env python3
"""
Interpretability Analysis Script (FIXED)
=========================================
Run attention analysis, task correlation analysis, and uncertainty calibration.

Usage:
    python scripts/experiments/run_analysis.py --experiment_dir experiments/ensemble_with_temp_xxx

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from src.data import (
    SurfProDataset,
    create_test_dataloader,
    TARGET_COLUMNS
)
from src.data.transforms import TargetScaler
from src.models.temperature_model import TemperatureAwareMTL

# Import analysis modules directly
from src.analysis.attention_analysis import (
    AttentionAnalyzer,
    visualize_molecule_attention,
    aggregate_attention_by_atom_type,
    plot_atom_type_importance,
)
from src.analysis.task_correlation import (
    TaskCorrelationAnalyzer,
    plot_correlation_heatmap,
    plot_task_weights,
    plot_correlation_comparison,
    validate_physical_relationships,
)
from src.analysis.uncertainty import (
    calibration_analysis,
    plot_calibration_curve,
    plot_coverage_comparison,
)


# =============================================================================
# Utility Functions
# =============================================================================

def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    return obj


def move_batch_to_device(batch, device):
    """Ensure all batch attributes are on the correct device."""
    batch = batch.to(device)

    # Explicitly move all tensor attributes
    for key in batch.keys:
        attr = getattr(batch, key, None)
        if attr is not None and isinstance(attr, torch.Tensor):
            setattr(batch, key, attr.to(device))

    # Handle global_features specifically
    if hasattr(batch, 'global_features') and batch.global_features is not None:
        batch.global_features = batch.global_features.to(device)

    # Handle temperature specifically
    if hasattr(batch, 'temperature') and batch.temperature is not None:
        batch.temperature = batch.temperature.to(device)

    return batch


# =============================================================================
# Analysis Functions
# =============================================================================

def run_attention_analysis(
        model,
        dataset,
        output_dir: Path,
        n_samples: int = 50,
        device: str = 'cuda'
):
    """Run attention analysis on sample molecules."""
    print("\n" + "=" * 50)
    print("Attention Analysis")
    print("=" * 50)

    output_dir = output_dir / 'attention'
    output_dir.mkdir(exist_ok=True)

    # Create analyzer
    analyzer = AttentionAnalyzer(
        model=model,
        task_names=TARGET_COLUMNS,
        device=device
    )

    # Analyze samples
    all_results = []

    from torch_geometric.data import Batch

    for i in range(min(n_samples, len(dataset))):
        data = dataset[i]

        # Create batch and move ALL tensors to device
        batch = Batch.from_data_list([data])
        batch = move_batch_to_device(batch, device)

        # Fix temperature shape
        if hasattr(batch, 'temperature') and batch.temperature is not None:
            if batch.temperature.dim() == 0:
                batch.temperature = batch.temperature.unsqueeze(0)
            batch.temperature = batch.temperature.to(device)

        try:
            result = analyzer.analyze_molecule(
                smiles=data.smiles,
                data=data,
                batch=batch
            )
            all_results.append(result)

            # Visualize first few molecules
            if i < 5:
                fig = visualize_molecule_attention(
                    smiles=data.smiles,
                    atom_importance=result['node_importance'],
                    title=f"Molecule {i + 1}",
                    save_path=str(output_dir / f'molecule_{i + 1}.png')
                )
                if fig:
                    plt.close(fig)

        except Exception as e:
            if i < 3:  # Only print first few warnings
                print(f"  Warning: Could not analyze sample {i}: {e}")
            continue

    print(f"  Analyzed {len(all_results)} molecules")

    # Aggregate by atom type
    if all_results:
        aggregated = aggregate_attention_by_atom_type(all_results)

        print("\n  Top important atom types:")
        for symbol, stats in list(aggregated.items())[:5]:
            print(f"    {symbol}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")

        # Plot
        fig = plot_atom_type_importance(
            aggregated,
            title="Average Atom Importance by Type",
            save_path=str(output_dir / 'atom_type_importance.png')
        )
        plt.close(fig)

        # Save aggregated results
        aggregated_serializable = convert_to_serializable(aggregated)
        with open(output_dir / 'atom_type_importance.json', 'w') as f:
            json.dump(aggregated_serializable, f, indent=2)
    else:
        print("  No molecules analyzed - skipping attention plots")

    print(f"  Figures saved to {output_dir}")


def run_task_correlation_analysis(
        predictions: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        model,
        output_dir: Path
):
    """Run task correlation analysis."""
    print("\n" + "=" * 50)
    print("Task Correlation Analysis")
    print("=" * 50)

    output_dir = output_dir / 'task_correlation'
    output_dir.mkdir(exist_ok=True)

    # Create analyzer
    analyzer = TaskCorrelationAnalyzer(TARGET_COLUMNS)

    # Run full analysis
    results = analyzer.full_analysis(predictions, targets, masks, model)

    # Plot target correlations
    print("\n  Computing target correlations...")
    fig = plot_correlation_heatmap(
        results['target_correlations'],
        TARGET_COLUMNS,
        title="Target Property Correlations",
        save_path=str(output_dir / 'target_correlations.png')
    )
    plt.close(fig)

    # Plot prediction correlations
    fig = plot_correlation_heatmap(
        results['prediction_correlations'],
        TARGET_COLUMNS,
        title="Prediction Correlations",
        save_path=str(output_dir / 'prediction_correlations.png')
    )
    plt.close(fig)

    # Plot comparison
    fig = plot_correlation_comparison(
        results['target_correlations'],
        results['prediction_correlations'],
        TARGET_COLUMNS,
        save_path=str(output_dir / 'correlation_comparison.png')
    )
    plt.close(fig)

    # Task weights
    if results.get('task_weights'):
        print("\n  Learned task weights:")
        for task, weight in results['task_weights'].items():
            print(f"    {task}: {weight:.4f}")

        fig = plot_task_weights(
            results['task_weights'],
            title="Learned Task Weights (Uncertainty Weighting)",
            save_path=str(output_dir / 'task_weights.png')
        )
        plt.close(fig)

    # Validate physical relationships
    print("\n  Validating physical relationships...")
    validation = validate_physical_relationships(targets, predictions, masks, TARGET_COLUMNS)

    for rel, result in validation.items():
        status = "✓" if result.get('model_captures_relationship', False) else "✗"
        print(f"    {status} {rel}")

    # Save results - convert to serializable format
    save_results = convert_to_serializable(results)

    with open(output_dir / 'task_correlations.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    validation_serializable = convert_to_serializable(validation)
    with open(output_dir / 'physical_validation.json', 'w') as f:
        json.dump(validation_serializable, f, indent=2)

    print(f"\n  Figures saved to {output_dir}")


def run_uncertainty_analysis(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        output_dir: Path
):
    """Run uncertainty calibration analysis."""
    print("\n" + "=" * 50)
    print("Uncertainty Calibration Analysis")
    print("=" * 50)

    output_dir = output_dir / 'uncertainty'
    output_dir.mkdir(exist_ok=True)

    # Run calibration analysis
    results = calibration_analysis(
        predictions, uncertainties, targets, masks, TARGET_COLUMNS
    )

    print("\n  Calibration results:")
    for task in TARGET_COLUMNS:
        if task in results and isinstance(results[task], dict):
            r = results[task]
            print(f"    {task}:")
            print(f"      Coverage (1σ): {r.get('coverage_1std', 0):.1%} (expected: 68.3%)")
            print(f"      Unc-Err Corr:  {r.get('uncertainty_error_correlation', 0):.3f}")

    # Plot calibration curves
    for i, task in enumerate(TARGET_COLUMNS):
        if task in results and isinstance(results[task], dict):
            try:
                fig = plot_calibration_curve(
                    results,
                    task,
                    save_path=str(output_dir / f'calibration_{task}.png')
                )
                if fig:
                    plt.close(fig)
            except Exception as e:
                print(f"  Warning: Could not plot calibration for {task}: {e}")

    # Plot coverage comparison
    try:
        fig = plot_coverage_comparison(
            results,
            TARGET_COLUMNS,
            save_path=str(output_dir / 'coverage_comparison.png')
        )
        plt.close(fig)
    except Exception as e:
        print(f"  Warning: Could not plot coverage comparison: {e}")

    # Save results - CONVERT ALL TO NATIVE PYTHON TYPES
    save_results = convert_to_serializable(results)

    with open(output_dir / 'calibration_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n  Figures saved to {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run interpretability analysis')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to experiment directory')
    parser.add_argument('--n_attention_samples', type=int, default=50,
                        help='Number of samples for attention analysis')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--skip_attention', action='store_true',
                        help='Skip attention analysis (faster)')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Interpretability Analysis")
    print("=" * 70)

    # Setup paths
    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    output_dir = experiment_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load config
    config_path = experiment_dir / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loading experiment: {experiment_dir.name}")

    # Load data
    data_root = PROJECT_ROOT / 'data'
    test_dataset = SurfProDataset(
        root=str(data_root),
        split='test',
        include_temperature=True
    )
    print(f"Test samples: {len(test_dataset)}")

    # Load predictions
    predictions = np.load(experiment_dir / 'test_predictions_mean.npy')
    uncertainties = np.load(experiment_dir / 'test_predictions_std.npy')
    targets = np.load(experiment_dir / 'test_targets.npy')
    masks = np.load(experiment_dir / 'test_masks.npy')

    print(f"Predictions shape: {predictions.shape}")

    # Load first model for attention analysis
    model = None
    if not args.skip_attention:
        model = TemperatureAwareMTL(**config['model'])
        model_path = experiment_dir / 'model_0.pt'

        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: Model not found, skipping attention analysis")
            model = None

    # Run analyses
    if model is not None and not args.skip_attention:
        try:
            run_attention_analysis(
                model=model,
                dataset=test_dataset,
                output_dir=output_dir,
                n_samples=args.n_attention_samples,
                device=device
            )
        except Exception as e:
            print(f"Warning: Attention analysis failed: {e}")
            import traceback
            traceback.print_exc()

    run_task_correlation_analysis(
        predictions=predictions,
        targets=targets,
        masks=masks,
        model=model,
        output_dir=output_dir
    )

    run_uncertainty_analysis(
        predictions=predictions,
        uncertainties=uncertainties,
        targets=targets,
        masks=masks,
        output_dir=output_dir
    )

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")

    # Print summary
    print("\n" + "-" * 50)
    print("GENERATED FIGURES")
    print("-" * 50)

    for folder in ['attention', 'task_correlation', 'uncertainty']:
        folder_path = output_dir / folder
        if folder_path.exists():
            files = list(folder_path.glob('*.png'))
            print(f"\n{folder}/")
            for f in files:
                print(f"  - {f.name}")


if __name__ == '__main__':
    main()