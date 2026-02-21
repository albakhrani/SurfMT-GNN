"""
Bond Feature Extraction
=======================
RDKit-based bond featurization for GNN edge attributes.
"""

import torch
from rdkit import Chem
from typing import List


BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ],
    'is_conjugated': [False, True],
    'is_in_ring': [False, True],
}


def one_hot_encoding(value, allowable_set: List, include_unknown: bool = True):
    """One-hot encode a value."""
    if include_unknown:
        allowable_set = list(allowable_set) + ['unknown']
    
    if value not in allowable_set:
        value = 'unknown'
    
    return [int(value == v) for v in allowable_set]


def get_bond_features(bond: Chem.Bond) -> torch.Tensor:
    """
    Extract features for a single bond.
    
    Parameters
    ----------
    bond : rdkit.Chem.Bond
        RDKit bond object.
        
    Returns
    -------
    torch.Tensor
        Bond feature vector.
    """
    features = []
    
    # Bond type
    features.extend(one_hot_encoding(bond.GetBondType(), BOND_FEATURES['bond_type']))
    
    # Stereo configuration
    features.extend(one_hot_encoding(bond.GetStereo(), BOND_FEATURES['stereo']))
    
    # Is conjugated
    features.append(int(bond.GetIsConjugated()))
    
    # Is in ring
    features.append(int(bond.IsInRing()))
    
    return torch.tensor(features, dtype=torch.float)


def get_bond_feature_dim() -> int:
    """Get total dimension of bond features."""
    dim = 0
    for key, values in BOND_FEATURES.items():
        if key in ['is_conjugated', 'is_in_ring']:
            dim += 1
        else:
            dim += len(values) + 1  # +1 for unknown
    return dim
