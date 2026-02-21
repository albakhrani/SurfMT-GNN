#!/usr/bin/env python3
"""
Molecular Featurization for SurfPro Dataset
============================================
Convert SMILES strings to molecular graphs with rich atom/bond features.

Feature Dimensions:
- Atom features: 34 dimensions
- Bond features: 12 dimensions
- Global molecular descriptors: 6 dimensions

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch_geometric.data import Data
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Constants for Feature Encoding
# =============================================================================

ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Other']
DEGREES = [0, 1, 2, 3, 4, 5, 6]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
NUM_HS = [0, 1, 2, 3, 4]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
    Chem.rdchem.BondStereo.STEREOANY,
]


# =============================================================================
# Helper Functions
# =============================================================================

def one_hot_encode(value, allowable_set: List, include_unknown: bool = True) -> List[int]:
    """One-hot encode a value from an allowable set."""
    encoding = [0] * len(allowable_set)
    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1
    elif include_unknown and len(allowable_set) > 0:
        encoding[-1] = 1
    return encoding


# =============================================================================
# Atom Featurizer (34 dimensions)
# =============================================================================

class AtomFeaturizer:
    """
    Featurizer for atom-level features.
    
    Feature Dimensions (Total: 34):
    - Atom type (one-hot): 10 dim
    - Degree (one-hot): 7 dim
    - Formal charge (one-hot): 5 dim
    - Hybridization (one-hot): 5 dim
    - Aromaticity (binary): 1 dim
    - Ring membership (binary): 1 dim
    - Number of Hs (one-hot): 5 dim
    """
    
    def __init__(self):
        self.atom_types = ATOM_TYPES
        self.degrees = DEGREES
        self.formal_charges = FORMAL_CHARGES
        self.hybridizations = HYBRIDIZATIONS
        self.num_hs = NUM_HS
        self.feature_dim = 34  # 10 + 7 + 5 + 5 + 1 + 1 + 5
        
    def __call__(self, atom: Chem.Atom) -> List[float]:
        """Featurize a single atom."""
        features = []
        
        # 1. Atom type (10 dim)
        symbol = atom.GetSymbol()
        if symbol not in self.atom_types[:-1]:
            symbol = 'Other'
        features.extend(one_hot_encode(symbol, self.atom_types, include_unknown=False))
        
        # 2. Degree (7 dim)
        degree = min(atom.GetTotalDegree(), max(self.degrees))
        features.extend(one_hot_encode(degree, self.degrees, include_unknown=False))
        
        # 3. Formal charge (5 dim)
        charge = max(min(atom.GetFormalCharge(), 2), -2)
        features.extend(one_hot_encode(charge, self.formal_charges, include_unknown=False))
        
        # 4. Hybridization (5 dim)
        hybridization = atom.GetHybridization()
        hyb_encoding = [0] * len(self.hybridizations)
        if hybridization in self.hybridizations:
            hyb_encoding[self.hybridizations.index(hybridization)] = 1
        features.extend(hyb_encoding)
        
        # 5. Aromaticity (1 dim)
        features.append(int(atom.GetIsAromatic()))
        
        # 6. Ring membership (1 dim)
        features.append(int(atom.IsInRing()))
        
        # 7. Number of Hs (5 dim)
        n_hs = min(atom.GetTotalNumHs(), max(self.num_hs))
        features.extend(one_hot_encode(n_hs, self.num_hs, include_unknown=False))
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all atom features."""
        names = []
        names.extend([f'atom_type_{t}' for t in self.atom_types])
        names.extend([f'degree_{d}' for d in self.degrees])
        names.extend([f'formal_charge_{c}' for c in self.formal_charges])
        names.extend([f'hybridization_{i}' for i in range(len(self.hybridizations))])
        names.append('is_aromatic')
        names.append('is_in_ring')
        names.extend([f'num_hs_{h}' for h in self.num_hs])
        return names


# =============================================================================
# Bond Featurizer (12 dimensions)
# =============================================================================

class BondFeaturizer:
    """
    Featurizer for bond-level features.
    
    Feature Dimensions (Total: 12):
    - Bond type (one-hot): 4 dim
    - Conjugated (binary): 1 dim
    - In ring (binary): 1 dim
    - Stereo (one-hot): 6 dim
    """
    
    def __init__(self):
        self.bond_types = BOND_TYPES
        self.bond_stereos = BOND_STEREOS
        self.feature_dim = 12  # 4 + 1 + 1 + 6
        
    def __call__(self, bond: Chem.Bond) -> List[float]:
        """Featurize a single bond."""
        features = []
        
        # 1. Bond type (4 dim)
        bond_type = bond.GetBondType()
        bt_encoding = [0] * len(self.bond_types)
        if bond_type in self.bond_types:
            bt_encoding[self.bond_types.index(bond_type)] = 1
        features.extend(bt_encoding)
        
        # 2. Conjugated (1 dim)
        features.append(int(bond.GetIsConjugated()))
        
        # 3. In ring (1 dim)
        features.append(int(bond.IsInRing()))
        
        # 4. Stereo (6 dim)
        stereo = bond.GetStereo()
        stereo_encoding = [0] * len(self.bond_stereos)
        if stereo in self.bond_stereos:
            stereo_encoding[self.bond_stereos.index(stereo)] = 1
        features.extend(stereo_encoding)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all bond features."""
        names = ['bond_single', 'bond_double', 'bond_triple', 'bond_aromatic']
        names.append('is_conjugated')
        names.append('is_in_ring')
        names.extend([f'stereo_{i}' for i in range(len(self.bond_stereos))])
        return names


# =============================================================================
# Global Molecular Descriptors (6 dimensions)
# =============================================================================

class GlobalDescriptorCalculator:
    """
    Calculator for global molecular descriptors.
    
    Descriptors (Total: 6):
    - Molecular weight (normalized)
    - LogP
    - TPSA (normalized)
    - Number of rotatable bonds (normalized)
    - Number of H-bond donors (normalized)
    - Number of H-bond acceptors (normalized)
    """
    
    def __init__(self):
        self.feature_dim = 6
        self.descriptor_names = [
            'mol_weight', 'logp', 'tpsa',
            'num_rotatable_bonds', 'num_hbd', 'num_hba'
        ]
        
    def __call__(self, mol: Chem.Mol) -> List[float]:
        """Calculate global molecular descriptors."""
        try:
            features = [
                Descriptors.MolWt(mol) / 500.0,
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol) / 100.0,
                rdMolDescriptors.CalcNumRotatableBonds(mol) / 10.0,
                rdMolDescriptors.CalcNumHBD(mol) / 5.0,
                rdMolDescriptors.CalcNumHBA(mol) / 10.0,
            ]
        except Exception:
            features = [0.0] * self.feature_dim
        return features
    
    def get_feature_names(self) -> List[str]:
        return self.descriptor_names.copy()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FeaturizationConfig:
    """Configuration for molecule featurization."""
    add_self_loops: bool = False
    include_global_features: bool = True
    include_hydrogen: bool = False


# =============================================================================
# Molecule Featurizer (Main Class)
# =============================================================================

class MoleculeFeaturizer:
    """
    Complete molecule featurizer converting SMILES to PyG Data objects.
    
    Combines:
    - Atom features (34 dim)
    - Bond features (12 dim)
    - Global molecular descriptors (6 dim)
    """
    
    def __init__(self, config: Optional[FeaturizationConfig] = None):
        self.config = config or FeaturizationConfig()
        self.atom_featurizer = AtomFeaturizer()
        self.bond_featurizer = BondFeaturizer()
        self.global_calculator = GlobalDescriptorCalculator()
        
        self.atom_dim = self.atom_featurizer.feature_dim
        self.bond_dim = self.bond_featurizer.feature_dim
        self.global_dim = self.global_calculator.feature_dim
        
    def __call__(
        self,
        smiles: str,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Optional[Data]:
        """
        Convert SMILES to PyTorch Geometric Data object.
        
        Parameters
        ----------
        smiles : str
            SMILES string of the molecule.
        y : torch.Tensor, optional
            Target values tensor of shape [num_tasks].
        mask : torch.Tensor, optional
            Mask tensor indicating valid targets (1=valid, 0=missing).
        **kwargs : dict
            Additional attributes to add to Data object.
            
        Returns
        -------
        Data or None
            PyG Data object, or None if SMILES is invalid.
        """
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        if self.config.include_hydrogen:
            mol = Chem.AddHs(mol)
        
        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.atom_featurizer(atom))
        
        if len(atom_features) == 0:
            return None
            
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Get bond features and edge indices
        edge_index = []
        edge_attr = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feat = self.bond_featurizer(bond)
            
            # Add both directions (undirected graph)
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([bond_feat, bond_feat])
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.bond_dim), dtype=torch.float)
        
        # Add self-loops if configured
        if self.config.add_self_loops:
            num_nodes = x.size(0)
            self_loop_index = torch.stack([
                torch.arange(num_nodes),
                torch.arange(num_nodes)
            ], dim=0)
            self_loop_attr = torch.zeros((num_nodes, self.bond_dim), dtype=torch.float)
            edge_index = torch.cat([edge_index, self_loop_index], dim=1)
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Add global features
        if self.config.include_global_features:
            global_features = self.global_calculator(mol)
            data.global_features = torch.tensor([global_features], dtype=torch.float)
        
        # Add SMILES
        data.smiles = smiles
        
        # Add targets and mask
        if y is not None:
            data.y = y
        if mask is not None:
            data.mask = mask
        
        # Add additional attributes
        for key, value in kwargs.items():
            setattr(data, key, value)
        
        return data
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        return {
            'atom_dim': self.atom_dim,
            'bond_dim': self.bond_dim,
            'global_dim': self.global_dim
        }


# =============================================================================
# Batch Processing
# =============================================================================

def featurize_smiles_batch(
    smiles_list: List[str],
    featurizer: MoleculeFeaturizer,
    y_list: Optional[List[torch.Tensor]] = None,
    mask_list: Optional[List[torch.Tensor]] = None,
    show_progress: bool = True
) -> Tuple[List[Data], List[int]]:
    """Featurize a batch of SMILES strings."""
    data_list = []
    failed_indices = []
    
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Featurizing")
        except ImportError:
            iterator = enumerate(smiles_list)
    else:
        iterator = enumerate(smiles_list)
    
    for idx, smiles in iterator:
        y = y_list[idx] if y_list is not None else None
        mask = mask_list[idx] if mask_list is not None else None
        
        data = featurizer(smiles, y=y, mask=mask)
        
        if data is not None:
            data_list.append(data)
        else:
            failed_indices.append(idx)
    
    return data_list, failed_indices


def validate_featurization(smiles: str, verbose: bool = True) -> Dict:
    """Validate featurization for a single SMILES."""
    featurizer = MoleculeFeaturizer()
    data = featurizer(smiles)
    
    if data is None:
        return {'valid': False, 'error': 'Invalid SMILES'}
    
    results = {
        'valid': True,
        'smiles': smiles,
        'num_atoms': data.x.size(0),
        'num_bonds': data.edge_index.size(1) // 2,
        'atom_feat_dim': data.x.size(1),
        'bond_feat_dim': data.edge_attr.size(1) if data.edge_attr.size(0) > 0 else 0,
        'has_global_features': hasattr(data, 'global_features'),
        'global_feat_dim': data.global_features.size(1) if hasattr(data, 'global_features') else 0
    }
    
    if verbose:
        print(f"SMILES: {smiles}")
        print(f"  Atoms: {results['num_atoms']}, Bonds: {results['num_bonds']}")
        print(f"  Atom features: {results['atom_feat_dim']}, Bond features: {results['bond_feat_dim']}")
    
    return results


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("="*60)
    print("Testing Molecule Featurizer")
    print("="*60)
    
    test_smiles = [
        "CCCCCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]",  # SDS
        "CCCCCCCCCCCC[N+](C)(C)C.[Cl-]",  # DTAC
        "INVALID_SMILES",
    ]
    
    featurizer = MoleculeFeaturizer()
    print(f"\nFeature dimensions: atom={featurizer.atom_dim}, bond={featurizer.bond_dim}, global={featurizer.global_dim}")
    
    for smiles in test_smiles:
        print("-"*40)
        validate_featurization(smiles)
