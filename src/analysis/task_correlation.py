#!/usr/bin/env python3
"""
Task Correlation Analysis for Multi-Task Learning
==================================================
Analyze relationships between tasks to understand knowledge sharing.

Key Analyses:
    - Task performance correlations
    - Learned task weight analysis
    - Embedding similarity across tasks
    - Cross-task prediction patterns

Scientific Value:
    - Validates MTL assumption (related tasks share structure)
    - Identifies which properties benefit from joint learning
    - Connects to physical relationships (e.g., CMC ↔ C20)

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import r2_score
import torch


# =============================================================================
# Task Correlation Analyzer
# =============================================================================

class TaskCorrelationAnalyzer:
    """
    Analyze correlations and relationships between tasks.
    
    Parameters
    ----------
    task_names : list
        List of task names.
    """
    
    def __init__(self, task_names: List[str]):
        self.task_names = task_names
        self.num_tasks = len(task_names)
        
        # Store results
        self.target_correlations = None
        self.prediction_correlations = None
        self.error_correlations = None
    
    def compute_target_correlations(
        self,
        targets: np.ndarray,
        masks: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise correlations between target properties.
        
        Parameters
        ----------
        targets : np.ndarray
            Target values [n_samples, n_tasks].
        masks : np.ndarray
            Valid masks [n_samples, n_tasks].
            
        Returns
        -------
        np.ndarray
            Correlation matrix [n_tasks, n_tasks].
        """
        corr_matrix = np.zeros((self.num_tasks, self.num_tasks))
        
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                # Find samples where both tasks have valid values
                valid = (masks[:, i] > 0.5) & (masks[:, j] > 0.5)
                n_valid = valid.sum()
                
                if n_valid > 2:
                    corr, _ = stats.pearsonr(targets[valid, i], targets[valid, j])
                    corr_matrix[i, j] = corr
                else:
                    corr_matrix[i, j] = np.nan
        
        self.target_correlations = corr_matrix
        return corr_matrix
    
    def compute_prediction_correlations(
        self,
        predictions: np.ndarray,
        masks: np.ndarray
    ) -> np.ndarray:
        """
        Compute correlations between model predictions.
        
        This shows whether the model has learned correlated representations.
        """
        corr_matrix = np.zeros((self.num_tasks, self.num_tasks))
        
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                valid = (masks[:, i] > 0.5) & (masks[:, j] > 0.5)
                n_valid = valid.sum()
                
                if n_valid > 2:
                    corr, _ = stats.pearsonr(predictions[valid, i], predictions[valid, j])
                    corr_matrix[i, j] = corr
                else:
                    corr_matrix[i, j] = np.nan
        
        self.prediction_correlations = corr_matrix
        return corr_matrix
    
    def compute_error_correlations(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray
    ) -> np.ndarray:
        """
        Compute correlations between prediction errors.
        
        Correlated errors suggest shared failure modes or related difficulties.
        """
        errors = predictions - targets
        corr_matrix = np.zeros((self.num_tasks, self.num_tasks))
        
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                valid = (masks[:, i] > 0.5) & (masks[:, j] > 0.5)
                n_valid = valid.sum()
                
                if n_valid > 2:
                    corr, _ = stats.pearsonr(errors[valid, i], errors[valid, j])
                    corr_matrix[i, j] = corr
                else:
                    corr_matrix[i, j] = np.nan
        
        self.error_correlations = corr_matrix
        return corr_matrix
    
    def analyze_task_weights(
        self,
        model
    ) -> Dict[str, float]:
        """
        Analyze learned task weights from uncertainty weighting.
        
        Parameters
        ----------
        model : nn.Module
            Trained model with task heads.
            
        Returns
        -------
        dict
            Task weights.
        """
        weights = model.get_task_weights()
        
        if weights is None:
            return {}
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        normalized = {k: v / total for k, v in weights.items()}
        
        return normalized
    
    def full_analysis(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        model=None
    ) -> Dict:
        """
        Run full correlation analysis.
        
        Returns
        -------
        dict
            All analysis results.
        """
        results = {
            'target_correlations': self.compute_target_correlations(targets, masks),
            'prediction_correlations': self.compute_prediction_correlations(predictions, masks),
            'error_correlations': self.compute_error_correlations(predictions, targets, masks),
        }
        
        if model is not None:
            results['task_weights'] = self.analyze_task_weights(model)
        
        return results


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    task_names: List[str],
    title: str = "Task Correlations",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'RdBu_r',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation matrix as heatmap.
    
    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix [n_tasks, n_tasks].
    task_names : list
        Task names for labels.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap.
    save_path : str, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for NaN values
    mask = np.isnan(corr_matrix)
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        ax=ax,
        mask=mask,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=0.5,
        xticklabels=task_names,
        yticklabels=task_names,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig


def plot_task_weights(
    weights: Dict[str, float],
    title: str = "Learned Task Weights",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot learned task weights.
    
    Parameters
    ----------
    weights : dict
        Task weights from model.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    tasks = list(weights.keys())
    values = list(weights.values())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(tasks, values, color='steelblue', alpha=0.8)
    
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=10
        )
    
    # Add reference line for uniform weights
    uniform = 1.0 / len(tasks)
    ax.axhline(y=uniform, color='r', linestyle='--', alpha=0.7, label=f'Uniform ({uniform:.3f})')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_comparison(
    target_corr: np.ndarray,
    pred_corr: np.ndarray,
    task_names: List[str],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare target and prediction correlations side by side.
    
    This shows how well the model captures inter-task relationships.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Target correlations
    mask1 = np.isnan(target_corr)
    sns.heatmap(
        target_corr, ax=axes[0], mask=mask1,
        cmap='RdBu_r', vmin=-1, vmax=1, center=0,
        annot=True, fmt='.2f', square=True,
        xticklabels=task_names, yticklabels=task_names
    )
    axes[0].set_title('Target Correlations', fontsize=12, fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # Prediction correlations
    mask2 = np.isnan(pred_corr)
    sns.heatmap(
        pred_corr, ax=axes[1], mask=mask2,
        cmap='RdBu_r', vmin=-1, vmax=1, center=0,
        annot=True, fmt='.2f', square=True,
        xticklabels=task_names, yticklabels=task_names
    )
    axes[1].set_title('Prediction Correlations', fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =============================================================================
# Embedding Analysis
# =============================================================================

def compute_task_embeddings(
    model,
    data_loader,
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings for task analysis.
    
    Parameters
    ----------
    model : nn.Module
        Trained model.
    data_loader : DataLoader
        Data loader.
    device : str
        Device.
        
    Returns
    -------
    dict
        Embeddings for different stages.
    """
    model.eval()
    
    graph_embeddings = []
    fused_embeddings = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            output = model(batch, return_embedding=True)
            
            graph_embeddings.append(output['graph_embedding'].cpu().numpy())
            fused_embeddings.append(output['fused_embedding'].cpu().numpy())
    
    return {
        'graph_embedding': np.concatenate(graph_embeddings, axis=0),
        'fused_embedding': np.concatenate(fused_embeddings, axis=0)
    }


def compute_embedding_similarity(
    embeddings: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    task_idx: int
) -> Dict[str, float]:
    """
    Compute how well embeddings cluster by task values.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Molecular embeddings [n_samples, embed_dim].
    targets : np.ndarray
        Target values [n_samples, n_tasks].
    masks : np.ndarray
        Valid masks [n_samples, n_tasks].
    task_idx : int
        Task index to analyze.
        
    Returns
    -------
    dict
        Clustering quality metrics.
    """
    valid = masks[:, task_idx] > 0.5
    
    if valid.sum() < 10:
        return {'silhouette': np.nan, 'correlation': np.nan}
    
    valid_embeddings = embeddings[valid]
    valid_targets = targets[valid, task_idx]
    
    # Compute correlation between embedding distances and target differences
    from scipy.spatial.distance import pdist, squareform
    
    # Subsample if too many samples (for speed)
    n_samples = min(500, len(valid_embeddings))
    idx = np.random.choice(len(valid_embeddings), n_samples, replace=False)
    
    embed_subset = valid_embeddings[idx]
    target_subset = valid_targets[idx]
    
    # Compute distance matrices
    embed_dists = pdist(embed_subset, metric='cosine')
    target_diffs = pdist(target_subset.reshape(-1, 1), metric='euclidean')
    
    # Correlation between distances
    corr, _ = stats.pearsonr(embed_dists, target_diffs)
    
    return {
        'embedding_target_correlation': float(corr),
        'n_samples': n_samples
    }


# =============================================================================
# Physical Relationship Validation
# =============================================================================

def validate_physical_relationships(
    targets: np.ndarray,
    predictions: np.ndarray,
    masks: np.ndarray,
    task_names: List[str]
) -> Dict[str, Dict]:
    """
    Validate that model captures known physical relationships.
    
    Known relationships:
        - pCMC and pC20 should be positively correlated
        - γCMC and πCMC should be negatively correlated (πCMC = γ0 - γCMC)
        - Γmax and Amin should be inversely related
    
    Parameters
    ----------
    targets : np.ndarray
        Target values.
    predictions : np.ndarray
        Model predictions.
    masks : np.ndarray
        Valid masks.
    task_names : list
        Task names.
        
    Returns
    -------
    dict
        Validation results for each relationship.
    """
    results = {}
    
    # Define known relationships
    relationships = [
        ('pCMC', 'pC20', 'positive', 'Both measure surfactant efficiency'),
        ('AW_ST_CMC', 'Pi_CMC', 'negative', 'πCMC = γ0 - γCMC'),
        ('Gamma_max', 'Area_min', 'negative', 'Amin = 1/(NA × Γmax)'),
    ]
    
    for task1, task2, expected, description in relationships:
        if task1 not in task_names or task2 not in task_names:
            continue
        
        idx1 = task_names.index(task1)
        idx2 = task_names.index(task2)
        
        valid = (masks[:, idx1] > 0.5) & (masks[:, idx2] > 0.5)
        n_valid = valid.sum()
        
        if n_valid < 5:
            results[f'{task1}-{task2}'] = {
                'expected': expected,
                'description': description,
                'valid': False,
                'reason': 'Insufficient data'
            }
            continue
        
        # Compute correlations
        target_corr, _ = stats.pearsonr(targets[valid, idx1], targets[valid, idx2])
        pred_corr, _ = stats.pearsonr(predictions[valid, idx1], predictions[valid, idx2])
        
        # Check if model captures the relationship
        target_sign = 'positive' if target_corr > 0 else 'negative'
        pred_sign = 'positive' if pred_corr > 0 else 'negative'
        
        results[f'{task1}-{task2}'] = {
            'expected': expected,
            'description': description,
            'target_correlation': float(target_corr),
            'prediction_correlation': float(pred_corr),
            'target_matches_expected': target_sign == expected,
            'model_matches_expected': pred_sign == expected,
            'model_captures_relationship': pred_sign == target_sign,
            'n_samples': int(n_valid)
        }
    
    return results


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Task Correlation Analysis")
    print("=" * 60)
    
    # Create dummy data
    task_names = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']
    n_samples = 100
    n_tasks = len(task_names)
    
    # Generate correlated targets
    np.random.seed(42)
    base = np.random.randn(n_samples)
    
    targets = np.zeros((n_samples, n_tasks))
    targets[:, 0] = base + np.random.randn(n_samples) * 0.3  # pCMC
    targets[:, 1] = -base + np.random.randn(n_samples) * 0.5  # AW_ST_CMC
    targets[:, 2] = base * 0.5 + np.random.randn(n_samples) * 0.4
    targets[:, 3] = -targets[:, 2] + np.random.randn(n_samples) * 0.3  # Inverse
    targets[:, 4] = -targets[:, 1] + np.random.randn(n_samples) * 0.4  # Related to AW_ST_CMC
    targets[:, 5] = targets[:, 0] + np.random.randn(n_samples) * 0.2  # Related to pCMC
    
    # Random mask (70% valid)
    masks = (np.random.rand(n_samples, n_tasks) > 0.3).astype(float)
    
    # Simulate predictions with noise
    predictions = targets + np.random.randn(n_samples, n_tasks) * 0.5
    
    # Run analysis
    analyzer = TaskCorrelationAnalyzer(task_names)
    
    # Compute correlations
    target_corr = analyzer.compute_target_correlations(targets, masks)
    pred_corr = analyzer.compute_prediction_correlations(predictions, masks)
    error_corr = analyzer.compute_error_correlations(predictions, targets, masks)
    
    print("\nTarget Correlations:")
    print(np.round(target_corr, 2))
    
    print("\nPrediction Correlations:")
    print(np.round(pred_corr, 2))
    
    # Validate physical relationships
    print("\nPhysical Relationship Validation:")
    validation = validate_physical_relationships(targets, predictions, masks, task_names)
    for rel, result in validation.items():
        status = "✓" if result.get('model_captures_relationship', False) else "✗"
        print(f"  {status} {rel}: {result['description']}")
    
    print("\n✓ Task correlation analysis test passed!")
