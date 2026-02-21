#!/usr/bin/env python3
"""
Advanced Visualizations for JCIM Paper
======================================
Additional publication-quality figures including:
- Model architecture diagram
- Temperature effect analysis
- Surfactant class analysis
- Learning curve
- Summary table figure

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'blue': '#2E86AB',
    'magenta': '#A23B72',
    'orange': '#F18F01',
    'green': '#2ECC71',
    'red': '#C73E1D',
    'purple': '#9B59B6',
    'gray': '#6C757D',
    'light': '#E8E8E8',
}

TARGET_COLUMNS = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']

PROPERTY_NAMES = {
    'pCMC': 'pCMC',
    'AW_ST_CMC': 'γ$_{CMC}$',
    'Gamma_max': 'Γ$_{max}$',
    'Area_min': 'A$_{min}$',
    'Pi_CMC': 'π$_{CMC}$',
    'pC20': 'pC$_{20}$',
}


# =============================================================================
# Model Architecture Diagram
# =============================================================================

def create_architecture_diagram(save_path: str = None, figsize=(14, 8)):
    """
    Create a professional model architecture diagram.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Define boxes
    def draw_box(x, y, w, h, text, color, ax, fontsize=9):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.2",
                             facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', wrap=True)

    def draw_arrow(start, end, ax, style='->'):
        arrow = FancyArrowPatch(start, end, arrowstyle=style,
                                mutation_scale=15, lw=1.5, color='black')
        ax.add_patch(arrow)

    # Input layer
    draw_box(0.5, 5.5, 2.5, 1.5, 'Molecular\nGraph\n(SMILES)', '#E3F2FD', ax)
    draw_box(0.5, 3.5, 2.5, 1.5, 'Temperature\n(°C)', '#FFF3E0', ax)
    draw_box(0.5, 1.5, 2.5, 1.5, 'Global\nDescriptors', '#E8F5E9', ax)

    # Encoders
    draw_box(4, 5, 2.5, 2.5, 'AttentiveFP\nEncoder\n\n3 GNN Layers\n256 hidden dim', '#BBDEFB', ax)
    draw_box(4, 2.5, 2.5, 1.5, 'Temperature\nEncoder\nMLP', '#FFE0B2', ax)
    draw_box(4, 0.5, 2.5, 1.5, 'Feature\nProjection', '#C8E6C9', ax)

    # Fusion
    draw_box(7.5, 2.5, 2, 3.5, 'Feature\nFusion\n\nConcat +\nMLP', '#E1BEE7', ax)

    # Task heads
    draw_box(10.5, 6, 2.5, 1.2, 'pCMC Head', COLORS['blue'], ax)
    draw_box(10.5, 4.5, 2.5, 1.2, 'γCMC Head', COLORS['magenta'], ax)
    draw_box(10.5, 3, 2.5, 1.2, 'Γmax Head', COLORS['orange'], ax)
    draw_box(10.5, 1.5, 2.5, 1.2, 'Amin Head', COLORS['green'], ax)
    draw_box(10.5, 0, 2.5, 1.2, '...', COLORS['gray'], ax)

    # Arrows - input to encoders
    draw_arrow((3, 6.25), (4, 6.25), ax)
    draw_arrow((3, 4.25), (4, 3.25), ax)
    draw_arrow((3, 2.25), (4, 1.25), ax)

    # Arrows - encoders to fusion
    draw_arrow((6.5, 6.25), (7.5, 5), ax)
    draw_arrow((6.5, 3.25), (7.5, 4.25), ax)
    draw_arrow((6.5, 1.25), (7.5, 3.5), ax)

    # Arrows - fusion to heads
    draw_arrow((9.5, 5.5), (10.5, 6.6), ax)
    draw_arrow((9.5, 4.75), (10.5, 5.1), ax)
    draw_arrow((9.5, 4), (10.5, 3.6), ax)
    draw_arrow((9.5, 3.25), (10.5, 2.1), ax)

    # Title and labels
    ax.text(7, 7.5, 'Temperature-Aware Multi-Task GNN Architecture',
            fontsize=14, fontweight='bold', ha='center')

    # Legend
    legend_items = [
        (COLORS['blue'], 'Input Features'),
        ('#BBDEFB', 'Encoders'),
        ('#E1BEE7', 'Fusion Layer'),
        (COLORS['gray'], 'Task Heads'),
    ]

    for i, (color, label) in enumerate(legend_items):
        rect = mpatches.Rectangle((11.5, 7.3 - i * 0.4), 0.3, 0.25,
                                  facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(12, 7.42 - i * 0.4, label, fontsize=8, va='center')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Summary Results Table Figure
# =============================================================================

def create_results_table_figure(metrics: Dict, save_path: str = None, figsize=(10, 6)):
    """
    Create a visual summary table of results.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Table data
    columns = ['Property', 'R²', 'RMSE', 'MAE', 'n', 'Status']

    rows = []
    for task in TARGET_COLUMNS:
        m = metrics.get(task, {})
        r2 = m.get('r2', 0)
        rmse = m.get('rmse', 0)
        mae = m.get('mae', 0)
        n = m.get('n_samples', 0)

        status = '✓' if r2 >= 0.75 else '○' if r2 >= 0.6 else '✗'

        rows.append([
            PROPERTY_NAMES[task],
            f'{r2:.3f}',
            f'{rmse:.3f}',
            f'{mae:.3f}',
            str(n),
            status
        ])

    # Add overall
    overall = metrics.get('overall', {})
    rows.append([
        'Overall',
        f"{overall.get('r2_mean', 0):.3f} ± {overall.get('r2_std', 0):.3f}",
        f"{overall.get('rmse_mean', 0):.3f}",
        '-',
        '-',
        '✓' if overall.get('r2_mean', 0) >= 0.75 else '○'
    ])

    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.18, 0.12, 0.12, 0.08, 0.1]
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Header style
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Alternating row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
            if i == len(rows):  # Overall row
                table[(i, j)].set_facecolor('#E3F2FD')
                table[(i, j)].set_text_props(fontweight='bold')

    # Color status column
    status_colors = {'✓': '#2ECC71', '○': '#F39C12', '✗': '#E74C3C'}
    for i in range(1, len(rows) + 1):
        status = rows[i - 1][-1]
        table[(i, 5)].set_text_props(color=status_colors.get(status, 'black'), fontweight='bold')

    ax.set_title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)

    # Legend
    ax.text(0.5, -0.05, '✓ R² ≥ 0.75    ○ R² ≥ 0.60    ✗ R² < 0.60',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Surfactant Class Performance
# =============================================================================

def create_class_performance_figure(
        predictions: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        surf_types: List[str],
        save_path: str = None,
        figsize=(12, 6)
):
    """
    Create figure showing performance breakdown by surfactant class.
    """
    from scipy import stats
    from sklearn.metrics import r2_score

    # Get unique types
    unique_types = list(set(surf_types))
    unique_types = [t for t in unique_types if t and t != 'Unknown']

    if len(unique_types) == 0:
        print("No surfactant type information available")
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Calculate R² by class for pCMC (most data)
    class_r2 = {}
    class_counts = {}

    for stype in unique_types:
        type_mask = np.array([t == stype for t in surf_types])
        prop_mask = masks[:, 0] > 0.5  # pCMC
        combined_mask = type_mask & prop_mask

        if combined_mask.sum() >= 5:
            y_true = targets[combined_mask, 0]
            y_pred = predictions[combined_mask, 0]
            class_r2[stype] = r2_score(y_true, y_pred)
            class_counts[stype] = combined_mask.sum()

    if not class_r2:
        print("Not enough data for class analysis")
        return None

    # Sort by R²
    sorted_classes = sorted(class_r2.keys(), key=lambda x: class_r2[x], reverse=True)

    # Bar chart
    ax1 = axes[0]
    x = np.arange(len(sorted_classes))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_classes)))

    bars = ax1.bar(x, [class_r2[c] for c in sorted_classes], color=colors, edgecolor='black')

    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_classes, rotation=45, ha='right')
    ax1.set_ylabel('R² (pCMC)', fontsize=11)
    ax1.set_title('(a) Performance by Surfactant Class', fontsize=12, fontweight='bold')
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 1.05)

    # Add count labels
    for bar, cls in zip(bars, sorted_classes):
        count = class_counts[cls]
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'n={count}', ha='center', va='bottom', fontsize=8)

    # Pie chart of data distribution
    ax2 = axes[1]
    sizes = [class_counts[c] for c in sorted_classes]
    explode = [0.02] * len(sorted_classes)

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=sorted_classes, autopct='%1.1f%%',
        colors=colors, explode=explode, startangle=90,
        textprops={'fontsize': 9}
    )
    ax2.set_title('(b) Test Set Distribution', fontsize=12, fontweight='bold')

    plt.suptitle('Analysis by Surfactant Class', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Training History (if available)
# =============================================================================

def create_training_curve_figure(
        experiment_dir: Path,
        save_path: str = None,
        figsize=(12, 5)
):
    """
    Create training/validation loss curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Try to load history from multiple folds
    histories = []
    for i in range(10):
        history_path = experiment_dir / f'fold_{i}' / 'history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                histories.append(json.load(f))

    if not histories:
        # Try alternative path
        for i in range(10):
            history_path = experiment_dir / f'history_{i}.json'
            if history_path.exists():
                with open(history_path, 'r') as f:
                    histories.append(json.load(f))

    if not histories:
        print("No training history found")
        axes[0].text(0.5, 0.5, 'Training history\nnot available',
                     ha='center', va='center', transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, 'Training history\nnot available',
                     ha='center', va='center', transform=axes[1].transAxes)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        return fig

    # Plot training curves
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for i, (hist, color) in enumerate(zip(histories, colors)):
        if 'train_loss' in hist:
            epochs = range(1, len(hist['train_loss']) + 1)
            ax1.plot(epochs, hist['train_loss'], alpha=0.3, color=color)

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Loss', fontsize=11)
    ax1.set_title('(a) Training Loss', fontsize=12, fontweight='bold')

    # Plot validation curves
    ax2 = axes[1]

    for i, (hist, color) in enumerate(zip(histories, colors)):
        if 'val_loss' in hist:
            epochs = range(1, len(hist['val_loss']) + 1)
            ax2.plot(epochs, hist['val_loss'], alpha=0.3, color=color, label=f'Fold {i}')

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('(b) Validation Loss', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=7, ncol=2, loc='upper right')

    plt.suptitle('Training Curves (10-Fold Cross-Validation)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Generate advanced figures')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("\n" + "=" * 70)
    print("Advanced Figure Generation")
    print("=" * 70)

    experiment_dir = Path(args.experiment_dir)
    output_dir = experiment_dir / 'figures'
    output_dir.mkdir(exist_ok=True)

    # Load data
    predictions = np.load(experiment_dir / 'test_predictions_mean.npy')
    uncertainties = np.load(experiment_dir / 'test_predictions_std.npy')
    targets = np.load(experiment_dir / 'test_targets.npy')
    masks = np.load(experiment_dir / 'test_masks.npy')

    with open(experiment_dir / 'results.json', 'r') as f:
        results = json.load(f)
    metrics = results.get('ensemble_metrics', {})

    # Load dataset for surfactant types
    data_root = PROJECT_ROOT / args.data_dir
    from src.data import SurfProDataset
    test_dataset = SurfProDataset(root=str(data_root), split='test', include_temperature=True)
    surf_types = [getattr(data, 'surf_type', 'Unknown') for data in test_dataset]

    # Generate figures
    print("\nGenerating advanced figures...")

    print("1. Architecture Diagram...")
    create_architecture_diagram(save_path=str(output_dir / 'fig_architecture.png'))

    print("2. Results Table...")
    create_results_table_figure(metrics, save_path=str(output_dir / 'fig_results_table.png'))

    print("3. Class Performance...")
    create_class_performance_figure(
        predictions, targets, masks, surf_types,
        save_path=str(output_dir / 'fig_class_performance.png')
    )

    print("4. Training Curves...")
    create_training_curve_figure(
        experiment_dir,
        save_path=str(output_dir / 'fig_training_curves.png')
    )

    print("\n" + "=" * 70)
    print("Advanced Figure Generation Complete!")
    print("=" * 70)
    print(f"Figures saved to: {output_dir}")


if __name__ == '__main__':
    main()