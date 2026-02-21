"""
Molecular Graph Construction
============================
Convert SMILES to PyTorch Geometric graph objects.
"""

import torch
from torch_geometric.data import Data
from rdkit import Chem
from typing import Optional, Dict, Any
import numpy as np

from .atom_features import get_atom_features
from .bond_features import get_bond_features


def mol_to_graph(
    smiles: str,
    y: Optional[Dict[str, float]] = None,
    temperature: Optional[float] = None,
    surfactant_type: Optional[str] = None
) -> Optional[Data]:
    """
    Convert SMILES string to PyTorch Geometric Data object.
    
    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    y : dict, optional
        Dictionary of target properties.
    temperature : float, optional
        Temperature in Celsius.
    surfactant_type : str, optional
        Surfactant class type.
        
    Returns
    -------
    Data or None
        PyTorch Geometric Data object, or None if SMILES is invalid.
    """
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.stack(atom_features, dim=0)
    
    # Get bond features and edge indices
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bond_feat = get_bond_features(bond)
        
        # Add both directions (undirected graph)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bond_feat, bond_feat])
    
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr, dim=0)
    else:
        # Handle molecules with no bonds
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, get_bond_features(None).shape[0]), dtype=torch.float)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    
    # Add targets if provided
    if y is not None:
        for prop, value in y.items():
            if value is not None and not np.isnan(value):
                setattr(data, prop, torch.tensor([value], dtype=torch.float))
            else:
                setattr(data, f'{prop}_mask', torch.tensor([0], dtype=torch.float))
    
    # Add auxiliary features
    if temperature is not None:
        data.temperature = torch.tensor([temperature], dtype=torch.float)
    
    if surfactant_type is not None:
        data.surfactant_type = surfactant_type
    
    return data
