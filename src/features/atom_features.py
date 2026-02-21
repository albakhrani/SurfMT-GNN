"""
Atom Feature Extraction
=======================
RDKit-based atom featurization for GNN input.
"""

import torch
import numpy as np
from rdkit import Chem
from typing import List


# Atom feature dimensions
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # 118 elements
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}


def one_hot_encoding(value, allowable_set: List, include_unknown: bool = True):
    """One-hot encode a value."""
    if include_unknown:
        allowable_set = list(allowable_set) + ['unknown']
    
    if value not in allowable_set:
        value = 'unknown'
    
    return [int(value == v) for v in allowable_set]


def get_atom_features(atom: Chem.Atom) -> torch.Tensor:
    """
    Extract features for a single atom.
    
    Parameters
    ----------
    atom : rdkit.Chem.Atom
        RDKit atom object.
        
    Returns
    -------
    torch.Tensor
        Atom feature vector.
    """
    features = []
    
    # Atomic number (one-hot, 118 + unknown)
    features.extend(one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))
    
    # Degree (one-hot)
    features.extend(one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree']))
    
    # Formal charge (one-hot)
    features.extend(one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
    
    # Number of hydrogens (one-hot)
    features.extend(one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']))
    
    # Hybridization (one-hot)
    features.extend(one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization']))
    
    # Is aromatic
    features.append(int(atom.GetIsAromatic()))
    
    # Is in ring
    features.append(int(atom.IsInRing()))
    
    return torch.tensor(features, dtype=torch.float)


def get_atom_feature_dim() -> int:
    """Get total dimension of atom features."""
    dim = 0
    for key, values in ATOM_FEATURES.items():
        if key in ['is_aromatic', 'is_in_ring']:
            dim += 1
        else:
            dim += len(values) + 1  # +1 for unknown
    return dim
