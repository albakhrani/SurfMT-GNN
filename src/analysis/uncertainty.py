#!/usr/bin/env python3
"""
Uncertainty Quantification and Calibration Analysis
====================================================
Analyze prediction uncertainty from ensemble models.

Key Analyses:
    - Ensemble uncertainty (prediction variance)
    - Calibration analysis (are uncertainties reliable?)
    - Confidence intervals
    - Error vs uncertainty relationship

Scientific Value:
    - Reliable uncertainty enables confident decision-making
    - Identifies when model predictions are trustworthy
    - Important for property imputation applications

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from scipy import stats


# =============================================================================
# Uncertainty Analyzer
# =============================================================================

class UncertaintyAnalyzer:
    """
    Analyze uncertainty quality from ensemble predictions.
    
    Parameters
    ----------
    task_names : list
        List of task names.
    """
    
    def __init__(self, task_names: List[str]):
        self.task_names = task_names
        self.num_tasks = len(task_names)
    
    def analyze_calibration(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Dict]:
        """
        Analyze calibration for each task.
        
        A well-calibrated model should have:
        - Errors proportional to predicted uncertainties
        - ~68% of samples within 1 std, ~95% within 2 std
        
        Parameters
        ----------
        predictions : np.ndarray
            Ensemble mean predictions [n_samples, n_tasks].
        uncertainties : np.ndarray
            Ensemble std [n_samples, n_tasks].
        targets : np.ndarray
            True targets [n_samples, n_tasks].
        masks : np.ndarray
            Valid masks [n_samples, n_tasks].
        n_bins : int
            Number of bins for calibration analysis.
            
        Returns
        -------
        dict
            Calibration metrics for each task.
        """
        results = {}
        
        for i, task in enumerate(self.task_names):
            valid = masks[:, i] > 0.5
            n_valid = valid.sum()
            
            if n_valid < 10:
                results[task] = {'valid': False, 'reason': 'Insufficient data'}
                continue
            
            pred = predictions[valid, i]
            unc = uncertainties[valid, i]
            true = targets[valid, i]
            
            # Compute z-scores
            errors = np.abs(pred - true)
            z_scores = errors / (unc + 1e-8)
            
            # Check coverage at different confidence levels
            coverage_1std = np.mean(z_scores <= 1.0)  # Should be ~68%
            coverage_2std = np.mean(z_scores <= 2.0)  # Should be ~95%
            coverage_3std = np.mean(z_scores <= 3.0)  # Should be ~99.7%
            
            # Compute correlation between uncertainty and error
            unc_error_corr, _ = stats.pearsonr(unc, errors)
            
            # Binned calibration
            bin_edges = np.percentile(unc, np.linspace(0, 100, n_bins + 1))
            bin_errors = []
            bin_uncertainties = []
            
            for j in range(n_bins):
                if j < n_bins - 1:
                    bin_mask = (unc >= bin_edges[j]) & (unc < bin_edges[j+1])
                else:
                    bin_mask = unc >= bin_edges[j]
                
                if bin_mask.sum() > 0:
                    bin_errors.append(np.mean(errors[bin_mask]))
                    bin_uncertainties.append(np.mean(unc[bin_mask]))
            
            results[task] = {
                'coverage_1std': float(coverage_1std),
                'coverage_2std': float(coverage_2std),
                'coverage_3std': float(coverage_3std),
                'expected_1std': 0.683,
                'expected_2std': 0.954,
                'uncertainty_error_correlation': float(unc_error_corr),
                'mean_uncertainty': float(np.mean(unc)),
                'mean_error': float(np.mean(errors)),
                'bin_errors': bin_errors,
                'bin_uncertainties': bin_uncertainties,
                'n_samples': int(n_valid)
            }
        
        return results
    
    def compute_calibration_error(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE) for each task.
        
        ECE measures the difference between predicted confidence
        and actual accuracy across confidence bins.
        """
        results = {}
        
        for i, task in enumerate(self.task_names):
            valid = masks[:, i] > 0.5
            
            if valid.sum() < 10:
                results[task] = np.nan
                continue
            
            pred = predictions[valid, i]
            unc = uncertainties[valid, i]
            true = targets[valid, i]
            
            errors = np.abs(pred - true)
            
            # Compute ECE
            n_bins = 10
            bin_edges = np.percentile(unc, np.linspace(0, 100, n_bins + 1))
            
            ece = 0.0
            total_samples = len(pred)
            
            for j in range(n_bins):
                if j < n_bins - 1:
                    bin_mask = (unc >= bin_edges[j]) & (unc < bin_edges[j+1])
                else:
                    bin_mask = unc >= bin_edges[j]
                
                n_bin = bin_mask.sum()
                if n_bin > 0:
                    avg_unc = np.mean(unc[bin_mask])
                    avg_err = np.mean(errors[bin_mask])
                    ece += (n_bin / total_samples) * np.abs(avg_err - avg_unc)
            
            results[task] = float(ece)
        
        return results


# =============================================================================
# Calibration Analysis Functions
# =============================================================================

def calibration_analysis(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    task_names: List[str]
) -> Dict:
    """
    Full calibration analysis.
    
    Parameters
    ----------
    predictions : np.ndarray
        Ensemble mean predictions.
    uncertainties : np.ndarray
        Ensemble std.
    targets : np.ndarray
        True targets.
    masks : np.ndarray
        Valid masks.
    task_names : list
        Task names.
        
    Returns
    -------
    dict
        Full calibration analysis results.
    """
    analyzer = UncertaintyAnalyzer(task_names)
    
    calibration = analyzer.analyze_calibration(
        predictions, uncertainties, targets, masks
    )
    
    ece = analyzer.compute_calibration_error(
        predictions, uncertainties, targets, masks
    )
    
    # Add ECE to results
    for task in task_names:
        if task in calibration and isinstance(calibration[task], dict):
            calibration[task]['ece'] = ece.get(task, np.nan)
    
    return calibration


def compute_confidence_intervals(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence intervals for predictions.
    
    Parameters
    ----------
    predictions : np.ndarray
        Ensemble mean predictions.
    uncertainties : np.ndarray
        Ensemble std.
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI).
        
    Returns
    -------
    tuple
        (lower_bound, upper_bound) arrays.
    """
    # Z-score for confidence level
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    lower = predictions - z * uncertainties
    upper = predictions + z * uncertainties
    
    return lower, upper


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_calibration_curve(
    calibration_results: Dict,
    task_name: str,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot calibration curve (uncertainty vs error).
    
    Parameters
    ----------
    calibration_results : dict
        Results from calibration_analysis().
    task_name : str
        Task to plot.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if task_name not in calibration_results:
        print(f"Task {task_name} not in results")
        return None
    
    result = calibration_results[task_name]
    
    if not isinstance(result, dict) or 'bin_uncertainties' not in result:
        print(f"Invalid results for {task_name}")
        return None
    
    bin_unc = result['bin_uncertainties']
    bin_err = result['bin_errors']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual calibration
    ax.scatter(bin_unc, bin_err, s=100, c='steelblue', alpha=0.8, label='Actual')
    
    # Plot perfect calibration line
    max_val = max(max(bin_unc), max(bin_err)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Perfect Calibration')
    
    # Fit line through points
    if len(bin_unc) > 2:
        slope, intercept, _, _, _ = stats.linregress(bin_unc, bin_err)
        x_fit = np.linspace(0, max_val, 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'g-', alpha=0.7, label=f'Fit (slope={slope:.2f})')
    
    ax.set_xlabel('Mean Predicted Uncertainty', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title(f'Calibration Curve: {task_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    # Add metrics text
    text = (f"Coverage (1σ): {result['coverage_1std']:.1%} (exp: 68.3%)\n"
            f"Coverage (2σ): {result['coverage_2std']:.1%} (exp: 95.4%)\n"
            f"Unc-Err Corr: {result['uncertainty_error_correlation']:.3f}")
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_coverage_comparison(
    calibration_results: Dict,
    task_names: List[str],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare coverage across tasks.
    
    Parameters
    ----------
    calibration_results : dict
        Results from calibration_analysis().
    task_names : list
        Tasks to include.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Get data
    tasks = []
    coverage_1std = []
    coverage_2std = []
    
    for task in task_names:
        if task in calibration_results and isinstance(calibration_results[task], dict):
            tasks.append(task)
            coverage_1std.append(calibration_results[task].get('coverage_1std', np.nan))
            coverage_2std.append(calibration_results[task].get('coverage_2std', np.nan))
    
    x = np.arange(len(tasks))
    width = 0.35
    
    # 1-sigma coverage
    axes[0].bar(x, coverage_1std, width, color='steelblue', alpha=0.8)
    axes[0].axhline(y=0.683, color='r', linestyle='--', label='Expected (68.3%)')
    axes[0].set_xlabel('Task')
    axes[0].set_ylabel('Coverage')
    axes[0].set_title('1σ Coverage', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    
    # 2-sigma coverage
    axes[1].bar(x, coverage_2std, width, color='steelblue', alpha=0.8)
    axes[1].axhline(y=0.954, color='r', linestyle='--', label='Expected (95.4%)')
    axes[1].set_xlabel('Task')
    axes[1].set_ylabel('Coverage')
    axes[1].set_title('2σ Coverage', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks, rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_uncertainty_vs_error(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    task_name: str,
    task_idx: int,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Scatter plot of uncertainty vs absolute error.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predictions.
    uncertainties : np.ndarray
        Uncertainties.
    targets : np.ndarray
        Targets.
    masks : np.ndarray
        Masks.
    task_name : str
        Task name.
    task_idx : int
        Task index.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save.
        
    Returns
    -------
    plt.Figure
        Figure.
    """
    valid = masks[:, task_idx] > 0.5
    
    pred = predictions[valid, task_idx]
    unc = uncertainties[valid, task_idx]
    true = targets[valid, task_idx]
    
    errors = np.abs(pred - true)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(unc, errors, alpha=0.5, s=20)
    
    # Add correlation
    corr, p_value = stats.pearsonr(unc, errors)
    
    # Fit line
    slope, intercept, _, _, _ = stats.linregress(unc, errors)
    x_fit = np.linspace(unc.min(), unc.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r-', alpha=0.7, label=f'Fit (r={corr:.3f})')
    
    # Perfect calibration
    ax.plot([0, unc.max()], [0, unc.max()], 'k--', alpha=0.5, label='Perfect')
    
    ax.set_xlabel('Predicted Uncertainty (σ)', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title(f'Uncertainty vs Error: {task_name}', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Uncertainty Analysis")
    print("=" * 60)
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 200
    task_names = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']
    n_tasks = len(task_names)
    
    # Generate data
    targets = np.random.randn(n_samples, n_tasks)
    
    # Well-calibrated uncertainties
    true_uncertainties = np.abs(np.random.randn(n_samples, n_tasks)) * 0.5 + 0.1
    
    # Predictions with noise proportional to uncertainty
    noise = np.random.randn(n_samples, n_tasks) * true_uncertainties
    predictions = targets + noise
    
    # Use true uncertainties as predicted uncertainties (perfect calibration)
    uncertainties = true_uncertainties
    
    # Random mask
    masks = (np.random.rand(n_samples, n_tasks) > 0.2).astype(float)
    
    # Run analysis
    print("\nRunning calibration analysis...")
    results = calibration_analysis(predictions, uncertainties, targets, masks, task_names)
    
    print("\nCalibration Results:")
    for task in task_names:
        if task in results and isinstance(results[task], dict):
            r = results[task]
            print(f"\n{task}:")
            print(f"  Coverage (1σ): {r['coverage_1std']:.1%} (expected: 68.3%)")
            print(f"  Coverage (2σ): {r['coverage_2std']:.1%} (expected: 95.4%)")
            print(f"  Unc-Err Corr: {r['uncertainty_error_correlation']:.3f}")
    
    # Compute confidence intervals
    print("\nComputing 95% confidence intervals...")
    lower, upper = compute_confidence_intervals(predictions, uncertainties, confidence=0.95)
    print(f"  Lower bound shape: {lower.shape}")
    print(f"  Upper bound shape: {upper.shape}")
    
    print("\n✓ Uncertainty analysis test passed!")
