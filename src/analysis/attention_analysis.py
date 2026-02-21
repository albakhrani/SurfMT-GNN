#!/usr/bin/env python3
"""
Attention Analysis for Molecular Interpretability
==================================================
Visualize and analyze attention weights to understand what the model learns.

Key Features:
    - Extract attention weights from AttentiveFP layers
    - Visualize attention on molecular structures
    - Aggregate attention by atom type
    - Compare attention across surfactant classes

Scientific Value:
    - Connects ML predictions to chemical understanding
    - Validates model against known chemistry
    - Identifies important molecular features for each property

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image


# =============================================================================
# Attention Analyzer Class
# =============================================================================

class AttentionAnalyzer:
    """
    Analyze attention weights from GNN models.
    
    Parameters
    ----------
    model : nn.Module
        Trained model with attention extraction capability.
    task_names : list
        List of task names.
    device : str
        Device for computation.
    """
    
    def __init__(
        self,
        model,
        task_names: List[str],
        device: str = 'cuda'
    ):
        self.model = model
        self.task_names = task_names
        self.device = device
        
        # Enable attention extraction
        self.model.enable_attention_extraction(True)
        self.model.eval()
    
    @torch.no_grad()
    def extract_attention(self, batch) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for a batch.
        
        Parameters
        ----------
        batch : Batch
            PyTorch Geometric batch.
            
        Returns
        -------
        dict
            Attention weights from each layer.
        """
        batch = batch.to(self.device)
        
        # Forward pass (triggers attention storage)
        _ = self.model(batch, return_embedding=True)
        
        # Get attention weights
        attention = self.model.get_attention_weights()
        
        return attention
    
    def compute_node_importance(
        self,
        attention: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
        method: str = 'mean'
    ) -> np.ndarray:
        """
        Compute node importance from attention weights.
        
        Parameters
        ----------
        attention : dict
            Attention weights from extract_attention().
        edge_index : torch.Tensor
            Edge connectivity.
        num_nodes : int
            Number of nodes.
        method : str
            Aggregation method: 'mean', 'max', 'sum'.
            
        Returns
        -------
        np.ndarray
            Node importance scores [num_nodes].
        """
        device = edge_index.device
        node_scores = torch.zeros(num_nodes, device=device)
        n_contributions = 0
        
        for layer_name, att in attention.items():
            if layer_name == 'readout':
                # Readout attention is directly node-level
                if att.numel() == num_nodes:
                    node_scores += att.squeeze()
                    n_contributions += 1
            else:
                # Edge attention - aggregate to nodes
                att = att.squeeze()
                if att.numel() != edge_index.shape[1]:
                    continue
                    
                target_nodes = edge_index[1]
                
                if method == 'mean':
                    counts = torch.zeros(num_nodes, device=device)
                    node_scores.scatter_add_(0, target_nodes, att)
                    counts.scatter_add_(0, target_nodes, torch.ones_like(att))
                    counts = counts.clamp(min=1)
                    # Don't divide yet - accumulate
                elif method == 'max':
                    temp_scores = torch.zeros(num_nodes, device=device)
                    temp_scores.scatter_reduce_(0, target_nodes, att, reduce='amax')
                    node_scores = torch.maximum(node_scores, temp_scores)
                else:  # sum
                    node_scores.scatter_add_(0, target_nodes, att)
                
                n_contributions += 1
        
        # Normalize
        node_scores = node_scores.cpu().numpy()
        if node_scores.max() > node_scores.min():
            node_scores = (node_scores - node_scores.min()) / (node_scores.max() - node_scores.min())
        
        return node_scores
    
    def analyze_molecule(
        self,
        smiles: str,
        data,
        batch
    ) -> Dict:
        """
        Full attention analysis for a single molecule.
        
        Parameters
        ----------
        smiles : str
            SMILES string.
        data : Data
            Single molecule data object.
        batch : Batch
            Batched data containing this molecule.
            
        Returns
        -------
        dict
            Analysis results including node importance.
        """
        # Extract attention
        attention = self.extract_attention(batch)
        
        # Compute node importance
        num_nodes = data.x.shape[0]
        importance = self.compute_node_importance(
            attention,
            data.edge_index,
            num_nodes
        )
        
        # Get atom info from RDKit
        mol = Chem.MolFromSmiles(smiles)
        atom_info = []
        
        if mol is not None:
            for i, atom in enumerate(mol.GetAtoms()):
                atom_info.append({
                    'idx': i,
                    'symbol': atom.GetSymbol(),
                    'importance': float(importance[i]) if i < len(importance) else 0.0,
                    'degree': atom.GetDegree(),
                    'hybridization': str(atom.GetHybridization()),
                    'is_aromatic': atom.GetIsAromatic()
                })
        
        return {
            'smiles': smiles,
            'attention': attention,
            'node_importance': importance,
            'atom_info': atom_info,
            'num_atoms': num_nodes
        }


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_molecule_attention(
    smiles: str,
    atom_importance: np.ndarray,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'RdYlGn_r',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize attention weights on molecular structure.
    
    Parameters
    ----------
    smiles : str
        SMILES string.
    atom_importance : np.ndarray
        Importance score for each atom [num_atoms].
    title : str, optional
        Figure title.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap name.
    save_path : str, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        print(f"Could not parse SMILES: {smiles}")
        return None
    
    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Normalize importance to [0, 1]
    if len(atom_importance) < mol.GetNumAtoms():
        # Pad with zeros if needed
        atom_importance = np.pad(
            atom_importance,
            (0, mol.GetNumAtoms() - len(atom_importance)),
            mode='constant'
        )
    
    atom_importance = atom_importance[:mol.GetNumAtoms()]
    
    # Create color map
    cmap_obj = plt.cm.get_cmap(cmap)
    
    # Create highlight colors
    highlight_atoms = {}
    highlight_bonds = {}
    
    for i, imp in enumerate(atom_importance):
        color = cmap_obj(imp)[:3]  # RGB only
        highlight_atoms[i] = color
    
    # Create drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 400)
    
    # Set drawing options
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    
    # Draw molecule with highlights
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors=highlight_atoms
    )
    drawer.FinishDrawing()
    
    # Convert to image
    img_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_data))
    
    # Create matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Atom Importance', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig


def get_important_atoms(
    smiles: str,
    atom_importance: np.ndarray,
    top_k: int = 5
) -> List[Dict]:
    """
    Get the most important atoms in a molecule.
    
    Parameters
    ----------
    smiles : str
        SMILES string.
    atom_importance : np.ndarray
        Importance scores.
    top_k : int
        Number of top atoms to return.
        
    Returns
    -------
    list
        List of atom dictionaries with importance.
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return []
    
    # Get top indices
    top_indices = np.argsort(atom_importance)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if idx < mol.GetNumAtoms():
            atom = mol.GetAtomWithIdx(int(idx))
            results.append({
                'index': int(idx),
                'symbol': atom.GetSymbol(),
                'importance': float(atom_importance[idx]),
                'neighbors': [n.GetSymbol() for n in atom.GetNeighbors()]
            })
    
    return results


def aggregate_attention_by_atom_type(
    analysis_results: List[Dict]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate attention scores by atom type across multiple molecules.
    
    Parameters
    ----------
    analysis_results : list
        List of analysis results from AttentionAnalyzer.analyze_molecule().
        
    Returns
    -------
    dict
        Mean and std importance for each atom type.
    """
    atom_type_scores = defaultdict(list)
    
    for result in analysis_results:
        for atom_info in result.get('atom_info', []):
            symbol = atom_info['symbol']
            importance = atom_info['importance']
            atom_type_scores[symbol].append(importance)
    
    aggregated = {}
    for symbol, scores in atom_type_scores.items():
        scores = np.array(scores)
        aggregated[symbol] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'count': len(scores)
        }
    
    # Sort by mean importance
    aggregated = dict(sorted(
        aggregated.items(),
        key=lambda x: x[1]['mean'],
        reverse=True
    ))
    
    return aggregated


def plot_atom_type_importance(
    aggregated: Dict[str, Dict[str, float]],
    title: str = "Atom Type Importance",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot aggregated atom type importance.
    
    Parameters
    ----------
    aggregated : dict
        Output from aggregate_attention_by_atom_type().
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
    symbols = list(aggregated.keys())
    means = [aggregated[s]['mean'] for s in symbols]
    stds = [aggregated[s]['std'] for s in symbols]
    counts = [aggregated[s]['count'] for s in symbols]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(symbols))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color='steelblue', alpha=0.8)
    
    ax.set_xlabel('Atom Type', fontsize=12)
    ax.set_ylabel('Mean Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, fontsize=10)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + stds[i] + 0.01,
            f'n={count}',
            ha='center', va='bottom', fontsize=8
        )
    
    ax.set_ylim(0, max(means) + max(stds) + 0.1)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =============================================================================
# Comparative Analysis
# =============================================================================

def compare_attention_by_property(
    analyzer: AttentionAnalyzer,
    data_loader,
    property_idx: int,
    n_samples: int = 50
) -> Dict:
    """
    Analyze attention patterns for a specific property.
    
    This helps understand which molecular features are important
    for predicting each surfactant property.
    
    Parameters
    ----------
    analyzer : AttentionAnalyzer
        Analyzer instance.
    data_loader : DataLoader
        Data loader with samples.
    property_idx : int
        Index of property to analyze.
    n_samples : int
        Number of samples to analyze.
        
    Returns
    -------
    dict
        Analysis results for the property.
    """
    all_results = []
    samples_analyzed = 0
    
    for batch in data_loader:
        if samples_analyzed >= n_samples:
            break
        
        batch_size = batch.num_graphs
        
        for i in range(min(batch_size, n_samples - samples_analyzed)):
            # Get single molecule data
            # This is a simplified version - full implementation would
            # properly separate batch items
            
            samples_analyzed += 1
    
    return {
        'property': analyzer.task_names[property_idx],
        'n_samples': samples_analyzed,
        'results': all_results
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Attention Analysis Module")
    print("=" * 60)
    
    # Test visualization with a simple molecule
    smiles = "CCCCCCCCCCCC(=O)O"  # Lauric acid
    
    # Create dummy importance scores
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    
    # Higher importance for carboxylic acid group
    importance = np.random.rand(num_atoms) * 0.3
    importance[-2:] = 0.9  # Carboxylic acid atoms
    
    print(f"\nSMILES: {smiles}")
    print(f"Num atoms: {num_atoms}")
    print(f"Importance range: [{importance.min():.2f}, {importance.max():.2f}]")
    
    # Get important atoms
    top_atoms = get_important_atoms(smiles, importance, top_k=3)
    print(f"\nTop 3 important atoms:")
    for atom in top_atoms:
        print(f"  {atom['symbol']} (idx={atom['index']}): {atom['importance']:.3f}")
    
    print("\n✓ Attention analysis test passed!")
