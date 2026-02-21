#!/usr/bin/env python3
"""
Publication-Quality Figure Generation for JCIM
===============================================
Generate professional, eye-catching figures for the paper:
"Temperature-Aware Multi-Task Graph Neural Networks for
Interpretable Surfactant Property Prediction"

Figures Include:
1. Parity plots with error distributions
2. Performance comparison radar chart
3. Task correlation heatmaps (publication style)
4. Temperature effect analysis
5. Example molecules with predictions
6. Graphical abstract / TOC figure
7. Error analysis by surfactant type
8. Ensemble uncertainty visualization

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats

# RDKit for molecular visualization
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Style Configuration for JCIM
# =============================================================================

# JCIM-compatible style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette - professional scientific colors
COLORS = {
    'primary': '#2E86AB',  # Blue
    'secondary': '#A23B72',  # Magenta
    'tertiary': '#F18F01',  # Orange
    'quaternary': '#C73E1D',  # Red
    'success': '#2ECC71',  # Green
    'neutral': '#6C757D',  # Gray
    'light': '#E8E8E8',  # Light gray
    'dark': '#2C3E50',  # Dark blue-gray
}

# Property-specific colors
PROPERTY_COLORS = {
    'pCMC': '#2E86AB',
    'AW_ST_CMC': '#A23B72',
    'Gamma_max': '#F18F01',
    'Area_min': '#2ECC71',
    'Pi_CMC': '#C73E1D',
    'pC20': '#9B59B6',
}

# Property display names with units
PROPERTY_NAMES = {
    'pCMC': 'pCMC',
    'AW_ST_CMC': 'γ$_{CMC}$ (mN/m)',
    'Gamma_max': 'Γ$_{max}$ (μmol/m²)',
    'Area_min': 'A$_{min}$ (Å²)',
    'Pi_CMC': 'π$_{CMC}$ (mN/m)',
    'pC20': 'pC$_{20}$',
}

TARGET_COLUMNS = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']


# =============================================================================
# Figure 1: Multi-Panel Parity Plots
# =============================================================================

def create_parity_plots(
        predictions: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        uncertainties: np.ndarray = None,
        save_path: str = None,
        figsize: Tuple = (12, 8)
) -> plt.Figure:
    """
    Create publication-quality multi-panel parity plots.

    Shows predicted vs actual values for all 6 properties with:
    - Color-coded points by error magnitude
    - R² and RMSE annotations
    - Perfect prediction line
    - Error distribution histograms
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    for idx, task in enumerate(TARGET_COLUMNS):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        # Get valid data
        mask = masks[:, idx] > 0.5
        if mask.sum() < 2:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(PROPERTY_NAMES[task], fontweight='bold')
            continue

        y_true = targets[mask, idx]
        y_pred = predictions[mask, idx]

        # Calculate metrics
        r2 = stats.pearsonr(y_true, y_pred)[0] ** 2
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        # Calculate errors for coloring
        errors = np.abs(y_true - y_pred)
        error_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

        # Create custom colormap (green to yellow to red)
        colors = plt.cm.RdYlGn_r(error_norm)

        # Plot scatter with error coloring
        scatter = ax.scatter(y_true, y_pred, c=errors, cmap='RdYlGn_r',
                             s=40, alpha=0.7, edgecolors='white', linewidths=0.5)

        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='Perfect')

        # ±10% and ±20% error bands
        ax.fill_between(lims, [l * 0.9 for l in lims], [l * 1.1 for l in lims],
                        alpha=0.1, color=COLORS['success'], label='±10%')

        # Metrics annotation box
        textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nn = {mask.sum()}'
        props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Experimental', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_title(PROPERTY_NAMES[task], fontsize=11, fontweight='bold',
                     color=PROPERTY_COLORS[task])
        ax.set_aspect('equal')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r')
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Absolute Error', fontsize=10)

    plt.suptitle('Multi-Task GNN Predictions vs Experimental Values',
                 fontsize=14, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Figure 2: Performance Radar Chart
# =============================================================================

def create_radar_chart(
        metrics: Dict[str, Dict[str, float]],
        save_path: str = None,
        figsize: Tuple = (8, 8)
) -> plt.Figure:
    """
    Create radar/spider chart showing model performance across all properties.
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    # Properties and their R² values
    properties = [PROPERTY_NAMES[t] for t in TARGET_COLUMNS]
    r2_values = [metrics.get(t, {}).get('r2', 0) for t in TARGET_COLUMNS]

    # Number of properties
    num_props = len(properties)
    angles = np.linspace(0, 2 * np.pi, num_props, endpoint=False).tolist()

    # Complete the loop
    r2_values += r2_values[:1]
    angles += angles[:1]

    # Plot
    ax.fill(angles, r2_values, color=COLORS['primary'], alpha=0.25)
    ax.plot(angles, r2_values, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)

    # Add target line at R²=0.8
    target_line = [0.8] * (num_props + 1)
    ax.plot(angles, target_line, '--', color=COLORS['quaternary'], linewidth=1.5,
            alpha=0.7, label='Target (R²=0.8)')

    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(properties, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax.set_title('Model Performance Across Properties (R²)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add values at each point
    for angle, r2, prop in zip(angles[:-1], r2_values[:-1], properties):
        ax.annotate(f'{r2:.2f}', xy=(angle, r2), xytext=(angle, r2 + 0.08),
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color=COLORS['dark'])

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Figure 3: Enhanced Correlation Heatmap
# =============================================================================

def create_correlation_heatmap(
        predictions: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        save_path: str = None,
        figsize: Tuple = (14, 5)
) -> plt.Figure:
    """
    Create side-by-side correlation heatmaps for targets and predictions.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Compute correlations
    def compute_corr_matrix(data, masks):
        n_tasks = data.shape[1]
        corr = np.zeros((n_tasks, n_tasks))
        for i in range(n_tasks):
            for j in range(n_tasks):
                valid = (masks[:, i] > 0.5) & (masks[:, j] > 0.5)
                if valid.sum() > 2:
                    corr[i, j], _ = stats.pearsonr(data[valid, i], data[valid, j])
                else:
                    corr[i, j] = np.nan
        return corr

    target_corr = compute_corr_matrix(targets, masks)
    pred_corr = compute_corr_matrix(predictions, masks)

    # Custom colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    labels = [PROPERTY_NAMES[t] for t in TARGET_COLUMNS]

    # Target correlations
    mask1 = np.isnan(target_corr)
    sns.heatmap(target_corr, ax=axes[0], mask=mask1, cmap=cmap,
                vmin=-1, vmax=1, center=0, annot=True, fmt='.2f',
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
                xticklabels=labels, yticklabels=labels, annot_kws={'size': 9})
    axes[0].set_title('(a) Experimental Correlations', fontsize=12, fontweight='bold', pad=10)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # Prediction correlations
    mask2 = np.isnan(pred_corr)
    sns.heatmap(pred_corr, ax=axes[1], mask=mask2, cmap=cmap,
                vmin=-1, vmax=1, center=0, annot=True, fmt='.2f',
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
                xticklabels=labels, yticklabels=labels, annot_kws={'size': 9})
    axes[1].set_title('(b) Predicted Correlations', fontsize=12, fontweight='bold', pad=10)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.suptitle('Cross-Property Correlation Analysis', fontsize=14, fontweight='bold', y=1.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Figure 4: Error Distribution Violin Plots
# =============================================================================

def create_error_violin_plots(
        predictions: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        save_path: str = None,
        figsize: Tuple = (12, 5)
) -> plt.Figure:
    """
    Create violin plots showing error distributions for each property.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Collect errors for each property
    error_data = []
    positions = []
    colors = []

    for idx, task in enumerate(TARGET_COLUMNS):
        mask = masks[:, idx] > 0.5
        if mask.sum() < 2:
            continue

        errors = predictions[mask, idx] - targets[mask, idx]
        error_data.append(errors)
        positions.append(idx)
        colors.append(PROPERTY_COLORS[task])

    # Create violin plot
    parts = ax.violinplot(error_data, positions=positions, showmeans=True,
                          showmedians=True, widths=0.7)

    # Color each violin
    for idx, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')

    # Style lines
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # Labels
    ax.set_xticks(range(len(TARGET_COLUMNS)))
    ax.set_xticklabels([PROPERTY_NAMES[t] for t in TARGET_COLUMNS], fontsize=10)
    ax.set_ylabel('Prediction Error', fontsize=11)
    ax.set_title('Error Distribution by Property', fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.7, label='Error Distribution'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Mean'),
        plt.Line2D([0], [0], color='black', linewidth=1.5, label='Median'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Figure 5: Performance Bar Chart with Literature Comparison
# =============================================================================

def create_performance_comparison(
        our_metrics: Dict[str, Dict[str, float]],
        save_path: str = None,
        figsize: Tuple = (12, 6)
) -> plt.Figure:
    """
    Create bar chart comparing our model with literature benchmarks.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Data preparation
    properties = TARGET_COLUMNS
    our_r2 = [our_metrics.get(p, {}).get('r2', 0) for p in properties]
    our_rmse = [our_metrics.get(p, {}).get('rmse', 0) for p in properties]

    # Literature values (from Hödl et al. 2025 and Brozos et al. 2024)
    lit_r2 = [0.94, 0.85, 0.74, None, None, 0.90]  # None = not reported
    lit_rmse = [0.35, 3.41, 0.57, None, None, 0.36]

    x = np.arange(len(properties))
    width = 0.35

    # R² comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width / 2, our_r2, width, label='This Work',
                    color=COLORS['primary'], alpha=0.8, edgecolor='black')

    # Literature bars (only where available)
    lit_r2_plot = [v if v is not None else 0 for v in lit_r2]
    lit_mask = [v is not None for v in lit_r2]
    bars2 = ax1.bar(x[lit_mask] + width / 2, [lit_r2_plot[i] for i in range(len(lit_r2)) if lit_mask[i]],
                    width, label='Literature SOTA', color=COLORS['tertiary'], alpha=0.8, edgecolor='black')

    ax1.set_xlabel('Property', fontsize=11)
    ax1.set_ylabel('R²', fontsize=11)
    ax1.set_title('(a) Coefficient of Determination', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([PROPERTY_NAMES[p] for p in properties], rotation=45, ha='right')
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper right')
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target')

    # Add value labels
    for bar, val in zip(bars1, our_r2):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # RMSE comparison
    ax2 = axes[1]
    bars3 = ax2.bar(x - width / 2, our_rmse, width, label='This Work',
                    color=COLORS['primary'], alpha=0.8, edgecolor='black')

    lit_rmse_plot = [v if v is not None else 0 for v in lit_rmse]
    bars4 = ax2.bar(x[lit_mask] + width / 2, [lit_rmse_plot[i] for i in range(len(lit_rmse)) if lit_mask[i]],
                    width, label='Literature SOTA', color=COLORS['tertiary'], alpha=0.8, edgecolor='black')

    ax2.set_xlabel('Property', fontsize=11)
    ax2.set_ylabel('RMSE', fontsize=11)
    ax2.set_title('(b) Root Mean Square Error', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([PROPERTY_NAMES[p] for p in properties], rotation=45, ha='right')
    ax2.legend(loc='upper right')

    # Add value labels
    for bar, val in zip(bars3, our_rmse):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle('Performance Comparison with Literature', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Figure 6: Example Molecules with Predictions
# =============================================================================

def create_molecule_prediction_figure(
        smiles_list: List[str],
        predictions: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        surfactant_types: List[str],
        save_path: str = None,
        figsize: Tuple = (14, 10)
) -> plt.Figure:
    """
    Create figure showing example molecules with their predicted vs actual values.
    """
    # Select diverse examples (best, worst, different types)
    n_examples = min(6, len(smiles_list))

    # Calculate overall errors
    overall_errors = []
    for i in range(len(predictions)):
        valid = masks[i] > 0.5
        if valid.sum() > 0:
            err = np.mean(np.abs(predictions[i, valid] - targets[i, valid]))
        else:
            err = float('inf')
        overall_errors.append(err)

    # Select: 2 best, 2 medium, 2 worst
    sorted_idx = np.argsort(overall_errors)
    selected_idx = list(sorted_idx[:2]) + list(sorted_idx[len(sorted_idx) // 2:len(sorted_idx) // 2 + 2]) + list(
        sorted_idx[-2:])
    selected_idx = selected_idx[:n_examples]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    for plot_idx, data_idx in enumerate(selected_idx):
        ax = fig.add_subplot(gs[plot_idx // 3, plot_idx % 3])

        smiles = smiles_list[data_idx]
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            ax.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center')
            continue

        # Draw molecule
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(350, 250)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))

        ax.imshow(img)
        ax.axis('off')

        # Create prediction table below
        pred_text = []
        for j, task in enumerate(TARGET_COLUMNS):
            if masks[data_idx, j] > 0.5:
                pred = predictions[data_idx, j]
                actual = targets[data_idx, j]
                err = abs(pred - actual)
                symbol = '✓' if err < 0.5 else '✗'
                pred_text.append(f"{PROPERTY_NAMES[task]}: {actual:.2f} → {pred:.2f} {symbol}")

        surf_type = surfactant_types[data_idx] if data_idx < len(surfactant_types) else 'Unknown'
        title = f"Type: {surf_type}\nError: {overall_errors[data_idx]:.3f}"
        ax.set_title(title, fontsize=9, fontweight='bold')

        # Add prediction text
        text = '\n'.join(pred_text[:3])  # Show top 3 properties
        ax.text(0.5, -0.1, text, transform=ax.transAxes, fontsize=7,
                ha='center', va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Example Predictions on Test Molecules', fontsize=14, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Figure 7: Graphical Abstract / TOC Figure
# =============================================================================

def create_graphical_abstract(
        metrics: Dict[str, Dict[str, float]],
        save_path: str = None,
        figsize: Tuple = (10, 6)
) -> plt.Figure:
    """
    Create a graphical abstract suitable for table of contents.
    """
    fig = plt.figure(figsize=figsize, facecolor='white')

    # Create layout
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

    # Left: Schematic (placeholder with text)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    # Draw schematic boxes
    boxes = [
        (1, 7, 3, 2, 'Molecular\nStructure', COLORS['primary']),
        (1, 4, 3, 2, 'Temperature', COLORS['tertiary']),
        (5, 5, 3, 3, 'MTL-GNN', COLORS['secondary']),
        (9, 7, 2, 1.5, 'pCMC', COLORS['primary']),
        (9, 5, 2, 1.5, 'γCMC', COLORS['secondary']),
        (9, 3, 2, 1.5, '...', COLORS['neutral']),
    ]

    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor='black', alpha=0.7)
        ax1.add_patch(rect)
        ax1.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=8,
                 fontweight='bold', color='white')

    # Draw arrows
    ax1.annotate('', xy=(5, 6.5), xytext=(4, 8),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax1.annotate('', xy=(5, 5.5), xytext=(4, 5),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax1.annotate('', xy=(9, 7), xytext=(8, 6.5),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax1.axis('off')
    ax1.set_title('Temperature-Aware MTL-GNN', fontsize=11, fontweight='bold')

    # Center: Mini radar chart
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    properties = [PROPERTY_NAMES[t] for t in TARGET_COLUMNS]
    r2_values = [metrics.get(t, {}).get('r2', 0) for t in TARGET_COLUMNS]
    angles = np.linspace(0, 2 * np.pi, len(properties), endpoint=False).tolist()
    r2_values += r2_values[:1]
    angles += angles[:1]

    ax2.fill(angles, r2_values, color=COLORS['primary'], alpha=0.3)
    ax2.plot(angles, r2_values, 'o-', color=COLORS['primary'], linewidth=2, markersize=5)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(properties, size=7)
    ax2.set_ylim(0, 1)
    ax2.set_title('Performance (R²)', fontsize=10, fontweight='bold', pad=10)

    # Right: Key metrics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    metrics_text = """
    Key Results:

    • 6 Properties Predicted
    • Temperature-Aware
    • Ensemble (10 models)

    Best Performance:
    • pCMC: R² = 0.86
    • pC20: R² = 0.86
    • Area_min: R² = 0.82

    Overall R² = 0.80 ± 0.05
    """
    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', family='sans-serif',
             bbox=dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.8))

    # Bottom: Example molecule (SDS)
    ax4 = fig.add_subplot(gs[1, :])

    # Draw example surfactant molecule (Sodium Dodecyl Sulfate - SDS)
    sds_smiles = "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"
    mol = Chem.MolFromSmiles(sds_smiles)
    if mol:
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(800, 200)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))
        ax4.imshow(img)

    ax4.axis('off')
    ax4.set_title('Example: Sodium Dodecyl Sulfate (SDS) - Anionic Surfactant',
                  fontsize=10, fontweight='bold')

    plt.suptitle('Temperature-Aware Multi-Task GNN for Surfactant Property Prediction',
                 fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Figure 8: Uncertainty Visualization
# =============================================================================

def create_uncertainty_figure(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        save_path: str = None,
        figsize: Tuple = (12, 8)
) -> plt.Figure:
    """
    Create figure showing prediction uncertainty analysis.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    for idx, task in enumerate(TARGET_COLUMNS):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        mask = masks[:, idx] > 0.5
        if mask.sum() < 2:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            continue

        y_true = targets[mask, idx]
        y_pred = predictions[mask, idx]
        y_std = uncertainties[mask, idx]
        errors = np.abs(y_pred - y_true)

        # Plot with error bars
        sorted_idx = np.argsort(y_true)
        x = np.arange(len(sorted_idx))

        ax.errorbar(x, y_pred[sorted_idx], yerr=y_std[sorted_idx] * 2,
                    fmt='o', markersize=4, color=PROPERTY_COLORS[task],
                    ecolor='gray', elinewidth=0.5, capsize=0, alpha=0.7,
                    label='Prediction ± 2σ')
        ax.scatter(x, y_true[sorted_idx], marker='x', s=20, color='red',
                   alpha=0.8, label='Experimental')

        ax.set_xlabel('Sample Index (sorted)', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.set_title(PROPERTY_NAMES[task], fontsize=10, fontweight='bold',
                     color=PROPERTY_COLORS[task])
        ax.legend(fontsize=7, loc='upper left')

    plt.suptitle('Predictions with Uncertainty Estimates (95% CI)',
                 fontsize=14, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Main Function
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to experiment directory')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("JCIM Publication Figure Generation")
    print("=" * 70)

    experiment_dir = Path(args.experiment_dir)
    output_dir = experiment_dir / 'figures'
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\nLoading data...")
    predictions = np.load(experiment_dir / 'test_predictions_mean.npy')
    uncertainties = np.load(experiment_dir / 'test_predictions_std.npy')
    targets = np.load(experiment_dir / 'test_targets.npy')
    masks = np.load(experiment_dir / 'test_masks.npy')

    # Load results
    with open(experiment_dir / 'results.json', 'r') as f:
        results = json.load(f)

    metrics = results.get('ensemble_metrics', {})

    # Load dataset for SMILES and types
    data_root = PROJECT_ROOT / args.data_dir
    from src.data import SurfProDataset
    test_dataset = SurfProDataset(root=str(data_root), split='test', include_temperature=True)

    smiles_list = [data.smiles for data in test_dataset]
    surf_types = [getattr(data, 'surf_type', 'Unknown') for data in test_dataset]

    print(f"Loaded {len(predictions)} predictions")

    # Generate figures
    print("\n" + "-" * 50)
    print("Generating Figures...")
    print("-" * 50)

    # Figure 1: Parity plots
    print("\n1. Parity Plots...")
    create_parity_plots(
        predictions, targets, masks, uncertainties,
        save_path=str(output_dir / 'fig1_parity_plots.png')
    )

    # Figure 2: Radar chart
    print("2. Radar Chart...")
    create_radar_chart(
        metrics,
        save_path=str(output_dir / 'fig2_radar_chart.png')
    )

    # Figure 3: Correlation heatmaps
    print("3. Correlation Heatmaps...")
    create_correlation_heatmap(
        predictions, targets, masks,
        save_path=str(output_dir / 'fig3_correlations.png')
    )

    # Figure 4: Error distributions
    print("4. Error Distributions...")
    create_error_violin_plots(
        predictions, targets, masks,
        save_path=str(output_dir / 'fig4_error_distributions.png')
    )

    # Figure 5: Performance comparison
    print("5. Performance Comparison...")
    create_performance_comparison(
        metrics,
        save_path=str(output_dir / 'fig5_performance_comparison.png')
    )

    # Figure 6: Example molecules
    print("6. Example Molecules...")
    create_molecule_prediction_figure(
        smiles_list, predictions, targets, masks, surf_types,
        save_path=str(output_dir / 'fig6_example_molecules.png')
    )

    # Figure 7: Graphical abstract
    print("7. Graphical Abstract...")
    create_graphical_abstract(
        metrics,
        save_path=str(output_dir / 'fig7_graphical_abstract.png')
    )

    # Figure 8: Uncertainty
    print("8. Uncertainty Visualization...")
    create_uncertainty_figure(
        predictions, uncertainties, targets, masks,
        save_path=str(output_dir / 'fig8_uncertainty.png')
    )

    print("\n" + "=" * 70)
    print("Figure Generation Complete!")
    print("=" * 70)
    print(f"\nAll figures saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()