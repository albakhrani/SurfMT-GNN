#!/usr/bin/env python3
"""
Data Module for SurfPro MTL-GNN
===============================
Data handling for surfactant property prediction.

Components:
    - SurfProDataset: PyTorch Geometric dataset with temperature
    - MolecularFeaturizer: Convert SMILES to graph features
    - TargetScaler: Normalize targets with log-transform
    - Data splitting utilities

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
"""

from .dataset import (
    SurfProDataset,
    TARGET_COLUMNS,
    get_cv_splits,
    save_cv_splits,
    load_cv_splits,
    create_dataloaders,
    create_test_dataloader,
    get_temperature_splits,
)

from .featurizer import (
    MoleculeFeaturizer,
)

from .transforms import (
    TargetScaler,
    AddNoise,
    RandomTargetMask,
    NormalizeFeatures,
    AddTemperatureFeature,
    Compose,
)

__all__ = [
    # Dataset
    'SurfProDataset',
    'TARGET_COLUMNS',
    
    # Splits
    'get_cv_splits',
    'save_cv_splits',
    'load_cv_splits',
    'get_temperature_splits',
    
    # DataLoaders
    'create_dataloaders',
    'create_test_dataloader',
    
    # Featurizer
    'MoleculeFeaturizer',
    
    # Transforms
    'TargetScaler',
    'AddNoise',
    'RandomTargetMask',
    'NormalizeFeatures',
    'AddTemperatureFeature',
    'Compose',
]
