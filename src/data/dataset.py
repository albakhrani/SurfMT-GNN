#!/usr/bin/env python3
"""
SurfPro Dataset with Temperature Support
=========================================
PyTorch Geometric dataset for surfactant property prediction.

Updated Features:
    - Temperature as input feature
    - Support for temperature-dependent predictions
    - Proper handling of temperature normalization

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Callable, Tuple

from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .featurizer import MoleculeFeaturizer

# =============================================================================
# Constants
# =============================================================================

TARGET_COLUMNS = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']
TEMPERATURE_COLUMN = 'temp'  # Temperature column in dataset
DEFAULT_TEMPERATURE = 25.0  # Default temperature if not specified


# =============================================================================
# SurfPro Dataset with Temperature
# =============================================================================

class SurfProDataset(InMemoryDataset):
    """
    SurfPro dataset for surfactant property prediction with temperature support.
    
    Parameters
    ----------
    root : str
        Root directory for dataset.
    split : str
        Dataset split: 'train' or 'test'.
    transform : callable, optional
        Transform applied to each sample.
    pre_transform : callable, optional
        Transform applied during preprocessing.
    include_temperature : bool
        Whether to include temperature as a feature.
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        include_temperature: bool = True
    ):
        self.split = split
        self.include_temperature = include_temperature
        self.featurizer = MoleculeFeaturizer()
        
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self) -> List[str]:
        return [f'surfpro_{self.split}.csv']
    
    @property
    def processed_file_names(self) -> List[str]:
        suffix = '_with_temp' if self.include_temperature else ''
        return [f'surfpro_{self.split}{suffix}.pt']
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')
    
    def download(self):
        """Check if raw files exist."""
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"Raw data file not found: {raw_path}\n"
                f"Please place surfpro_{self.split}.csv in {self.raw_dir}"
            )
    
    def process(self):
        """Process raw data into PyTorch Geometric format."""
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        df = pd.read_csv(raw_path)
        
        print(f"Processing {self.split} data: {len(df)} samples")
        print(f"Include temperature: {self.include_temperature}")
        
        data_list = []
        failed = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {self.split}"):
            try:
                smiles = row['SMILES']
                
                # Featurize molecule
                mol_data = self.featurizer(smiles)
                
                if mol_data is None:
                    failed += 1
                    continue
                
                # Featurizer already returns a Data object
                data = mol_data
                
                # Add temperature
                if self.include_temperature and TEMPERATURE_COLUMN in df.columns:
                    temp = row[TEMPERATURE_COLUMN]
                    if pd.isna(temp):
                        temp = DEFAULT_TEMPERATURE
                    data.temperature = torch.tensor([float(temp)], dtype=torch.float32)
                else:
                    data.temperature = torch.tensor([DEFAULT_TEMPERATURE], dtype=torch.float32)
                
                # Add surfactant type
                if 'type' in df.columns:
                    data.surf_type = row['type']
                
                # Add targets and mask
                targets = []
                mask = []
                
                for col in TARGET_COLUMNS:
                    if col in df.columns:
                        val = row[col]
                        if pd.isna(val):
                            targets.append(0.0)
                            mask.append(0.0)
                        else:
                            targets.append(float(val))
                            mask.append(1.0)
                    else:
                        targets.append(0.0)
                        mask.append(0.0)
                
                data.y = torch.tensor(targets, dtype=torch.float32)
                data.mask = torch.tensor(mask, dtype=torch.float32)
                
                # Add fold info if available
                if 'fold' in df.columns:
                    data.fold = int(row['fold'])
                
                data_list.append(data)
                
            except Exception as e:
                failed += 1
                continue
        
        print(f"Successfully processed: {len(data_list)} samples")
        print(f"Failed: {failed} samples")
        
        # Save processed data
        self.save(data_list, self.processed_paths[0])
    
    def get_temperature_stats(self) -> Dict[str, float]:
        """Get temperature statistics from dataset."""
        temps = []
        for data in self:
            if hasattr(data, 'temperature'):
                temps.append(data.temperature.item())
        
        temps = np.array(temps)
        return {
            'mean': float(np.mean(temps)),
            'std': float(np.std(temps)),
            'min': float(np.min(temps)),
            'max': float(np.max(temps))
        }
    
    def get_target_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each target property."""
        stats = {col: [] for col in TARGET_COLUMNS}
        
        for data in self:
            for i, col in enumerate(TARGET_COLUMNS):
                if data.mask[i].item() == 1.0:
                    stats[col].append(data.y[i].item())
        
        result = {}
        for col, values in stats.items():
            if len(values) > 0:
                values = np.array(values)
                result[col] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            else:
                result[col] = {'count': 0}
        
        return result


# =============================================================================
# Data Splitting Utilities
# =============================================================================

def get_cv_splits(
    dataset: SurfProDataset,
    n_folds: int = 10,
    seed: int = 42,
    stratify_by: str = 'type'
) -> List[Tuple[List[int], List[int]]]:
    """
    Create cross-validation splits stratified by surfactant type.
    
    Parameters
    ----------
    dataset : SurfProDataset
        The dataset to split.
    n_folds : int
        Number of CV folds.
    seed : int
        Random seed.
    stratify_by : str
        Property to stratify by ('type' or 'temperature').
        
    Returns
    -------
    list
        List of (train_indices, val_indices) tuples.
    """
    from sklearn.model_selection import StratifiedKFold
    
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    # Get stratification labels
    if stratify_by == 'type':
        labels = []
        for i, data in enumerate(dataset):
            if hasattr(data, 'surf_type'):
                labels.append(data.surf_type)
            else:
                labels.append('unknown')
        labels = np.array(labels)
    elif stratify_by == 'temperature':
        labels = []
        for data in dataset:
            temp = data.temperature.item()
            # Bin temperatures
            if temp < 20:
                labels.append('cold')
            elif temp < 30:
                labels.append('room')
            else:
                labels.append('warm')
        labels = np.array(labels)
    else:
        labels = np.zeros(n_samples)
    
    # Create stratified folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    splits = []
    for train_idx, val_idx in skf.split(indices, labels):
        splits.append((train_idx.tolist(), val_idx.tolist()))
    
    return splits


def save_cv_splits(splits: List[Tuple[List[int], List[int]]], path: str):
    """Save CV splits to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_data = [{'train': train, 'val': val} for train, val in splits]
    with open(path, 'w') as f:
        json.dump(save_data, f)
    print(f"Saved CV splits to {path}")


def load_cv_splits(path: str) -> List[Tuple[List[int], List[int]]]:
    """Load CV splits from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    # Handle both formats: list or dict with 'splits' key
    if isinstance(data, dict) and 'splits' in data:
        data = data['splits']
    return [(d['train'], d['val']) for d in data]


# =============================================================================
# DataLoader Utilities
# =============================================================================

def create_dataloaders(
    dataset: SurfProDataset,
    train_idx: List[int],
    val_idx: List[int],
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Parameters
    ----------
    dataset : SurfProDataset
        Full dataset.
    train_idx : list
        Training indices.
    val_idx : list
        Validation indices.
    batch_size : int
        Batch size.
    num_workers : int
        Number of data loading workers.
        
    Returns
    -------
    tuple
        (train_loader, val_loader)
    """
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


def create_test_dataloader(
    dataset: SurfProDataset,
    batch_size: int = 32,
    num_workers: int = 0
) -> DataLoader:
    """Create test dataloader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


# =============================================================================
# Temperature-Stratified Splits
# =============================================================================

def get_temperature_splits(
    dataset: SurfProDataset,
    test_temps: List[float] = None,
    tolerance: float = 2.0
) -> Tuple[List[int], List[int]]:
    """
    Split dataset by temperature for temperature extrapolation testing.
    
    Parameters
    ----------
    dataset : SurfProDataset
        Dataset to split.
    test_temps : list
        Temperatures to hold out for testing (e.g., [35.0, 40.0]).
    tolerance : float
        Temperature tolerance for matching.
        
    Returns
    -------
    tuple
        (train_indices, test_indices)
    """
    if test_temps is None:
        test_temps = [35.0, 40.0]  # Hold out higher temperatures
    
    train_idx = []
    test_idx = []
    
    for i, data in enumerate(dataset):
        temp = data.temperature.item()
        
        # Check if temperature is in test set
        is_test = any(abs(temp - t) <= tolerance for t in test_temps)
        
        if is_test:
            test_idx.append(i)
        else:
            train_idx.append(i)
    
    return train_idx, test_idx


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing SurfPro Dataset with Temperature")
    print("=" * 60)
    
    # Check if data exists
    data_root = Path('data')
    
    if (data_root / 'processed' / 'surfpro_train_with_temp.pt').exists():
        print("\nLoading existing processed data...")
        dataset = SurfProDataset(root=str(data_root), split='train', include_temperature=True)
        print(f"Dataset size: {len(dataset)}")
        
        # Check first sample
        sample = dataset[0]
        print(f"\nSample data:")
        print(f"  x shape: {sample.x.shape}")
        print(f"  edge_index shape: {sample.edge_index.shape}")
        print(f"  temperature: {sample.temperature.item():.1f}°C")
        print(f"  y (targets): {sample.y}")
        print(f"  mask: {sample.mask}")
        
        # Get temperature statistics
        temp_stats = dataset.get_temperature_stats()
        print(f"\nTemperature statistics:")
        for key, val in temp_stats.items():
            print(f"  {key}: {val:.2f}")
        
        # Get target statistics
        target_stats = dataset.get_target_stats()
        print(f"\nTarget statistics:")
        for col, stats in target_stats.items():
            if stats['count'] > 0:
                print(f"  {col}: n={stats['count']}, mean={stats['mean']:.4f}")
    else:
        print("\nProcessed data not found. Run data preparation first.")
    
    print("\n✓ Dataset test complete!")


