"""
Evaluation Metrics
==================
Performance metrics for regression tasks.
"""

import numpy as np
import torch
from typing import Dict, Optional, Union
from sklearn.metrics import r2_score as sklearn_r2, mean_squared_error, mean_absolute_error


def r2_score(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Calculate R² score.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    mask : array-like, optional
        Binary mask for valid values.
        
    Returns
    -------
    float
        R² score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    if mask is not None:
        mask = mask.flatten().astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    # Remove NaN values
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    
    if len(y_true) < 2:
        return np.nan
    
    return sklearn_r2(y_true, y_pred)


def rmse(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    mask: Optional[np.ndarray] = None
) -> float:
    """Calculate Root Mean Squared Error."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    if mask is not None:
        mask = mask.flatten().astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    
    if len(y_true) == 0:
        return np.nan
    
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    mask: Optional[np.ndarray] = None
) -> float:
    """Calculate Mean Absolute Error."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    if mask is not None:
        mask = mask.flatten().astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    
    if len(y_true) == 0:
        return np.nan
    
    return mean_absolute_error(y_true, y_pred)


def calculate_metrics(
    y_true: Dict[str, np.ndarray],
    y_pred: Dict[str, np.ndarray],
    masks: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate all metrics for multiple tasks.
    
    Parameters
    ----------
    y_true : dict
        Ground truth values per task.
    y_pred : dict
        Predictions per task.
    masks : dict, optional
        Masks per task.
        
    Returns
    -------
    dict
        Nested dict with metrics per task.
    """
    results = {}
    
    for task in y_true.keys():
        if task not in y_pred:
            continue
            
        mask = masks.get(task) if masks else None
        
        results[task] = {
            'r2': r2_score(y_true[task], y_pred[task], mask),
            'rmse': rmse(y_true[task], y_pred[task], mask),
            'mae': mae(y_true[task], y_pred[task], mask)
        }
    
    return results
