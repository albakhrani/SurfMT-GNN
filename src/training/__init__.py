#!/usr/bin/env python3
"""
Training Module for SurfPro MTL-GNN
===================================
Training infrastructure for multi-task learning.

Components:
    - Loss functions with mask support
    - Trainer classes
    - Ensemble trainer
    - Utilities

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
"""

from .losses import (
    MaskedMSELoss,
    MaskedMAELoss,
    MaskedHuberLoss,
    MultiTaskLoss,
    create_loss_function,
)

from .trainer import (
    TrainingConfig,
    EarlyStopping,
    MetricsComputer,
    Trainer,
)

from .ensemble_trainer import (
    EnsembleConfig,
    EnsembleTrainer,
    SingleModelTrainer,
)

__all__ = [
    # Loss functions
    'MaskedMSELoss',
    'MaskedMAELoss',
    'MaskedHuberLoss',
    'MultiTaskLoss',
    'create_loss_function',
    
    # Training
    'TrainingConfig',
    'EarlyStopping',
    'MetricsComputer',
    'Trainer',
    
    # Ensemble
    'EnsembleConfig',
    'EnsembleTrainer',
    'SingleModelTrainer',
]
