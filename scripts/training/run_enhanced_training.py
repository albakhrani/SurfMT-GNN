#!/usr/bin/env python3
"""
ENHANCED Temperature-Aware Multi-Task GNN Training Script
==========================================================
Improvements over baseline:
1. Task-specific loss weighting (boost weak properties)
2. Global molecular descriptors (RDKit features)
3. Larger ensemble (60 models)
4. Learning rate warmup
5. Better regularization
6. Gradient accumulation for stability

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)

Target Performance:
- Overall R² > 0.87
- pCMC R² > 0.91
- Gamma_max R² > 0.82

Run:
    python run_enhanced_training.py --data_dir data/raw --device cuda

"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import math

warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

# PyTorch Geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP, global_mean_pool

# Scikit-learn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


# =============================================================================
# CONFIGURATION - ENHANCED
# =============================================================================

class Config:
    """Enhanced configuration for improved performance."""

    # Model Architecture - Slightly larger
    HIDDEN_DIM = 128  # Increased from 96
    OUTPUT_DIM = 256  # Increased from 192
    NUM_LAYERS = 3  # Increased from 2
    NUM_TIMESTEPS = 2
    DROPOUT = 0.15  # Slightly higher

    # Global descriptors
    NUM_GLOBAL_FEATURES = 12  # RDKit molecular descriptors

    # Temperature encoding
    TEMP_EMBED_DIM = 64

    # Training - Enhanced
    LEARNING_RATE = 5e-4  # Slightly lower for stability
    WEIGHT_DECAY = 1e-4  # Increased regularization
    BATCH_SIZE = 32
    MAX_EPOCHS = 400  # More epochs
    PATIENCE = 80  # More patience
    WARMUP_EPOCHS = 20  # Learning rate warmup

    # Ensemble - Larger
    N_ENSEMBLE = 60  # Increased from 40
    N_FOLDS = 10

    # Tasks
    TARGETS = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']
    NUM_TASKS = 6

    # Task-specific loss weights (boost weaker properties)
    # Based on: inverse of baseline R² performance
    TASK_WEIGHTS = {
        'pCMC': 1.0,  # R²=0.898, good
        'AW_ST_CMC': 1.3,  # R²=0.812, needs boost
        'Gamma_max': 1.5,  # R²=0.769, weakest - highest weight
        'Area_min': 1.1,  # R²=0.857, slight boost
        'Pi_CMC': 1.3,  # R²=0.806, needs boost
        'pC20': 1.0,  # R²=0.897, good
    }

    # Unit conversions
    UNIT_CONVERSIONS = {
        'pCMC': 1.0,
        'AW_ST_CMC': 1.0,
        'Gamma_max': 1e6,  # mol/m² → μmol/m²
        'Area_min': 1.0,
        'Pi_CMC': 1.0,
        'pC20': 1.0,
    }

    UNIT_NAMES = {
        'pCMC': '-log₁₀(M)',
        'AW_ST_CMC': 'mN/m',
        'Gamma_max': 'μmol/m²',
        'Area_min': 'nm²',
        'Pi_CMC': 'mN/m',
        'pC20': '-log₁₀(M)',
    }


# =============================================================================
# GLOBAL MOLECULAR DESCRIPTORS
# =============================================================================

def compute_global_descriptors(mol) -> List[float]:
    """
    Compute global molecular descriptors using RDKit.

    These descriptors capture molecular properties that are important
    for surfactant behavior but may not be learned from graph structure alone.

    Returns:
        List of 12 normalized descriptor values
    """
    try:
        descriptors = [
            # Size and shape
            Descriptors.MolWt(mol) / 500.0,  # Molecular weight (normalized)
            Descriptors.HeavyAtomCount(mol) / 50.0,  # Heavy atom count
            rdMolDescriptors.CalcNumRotatableBonds(mol) / 20.0,  # Rotatable bonds

            # Polarity and charge
            Descriptors.TPSA(mol) / 150.0,  # Topological polar surface area
            Descriptors.MolLogP(mol) / 10.0,  # LogP (hydrophobicity)

            # H-bonding
            Descriptors.NumHDonors(mol) / 10.0,  # H-bond donors
            Descriptors.NumHAcceptors(mol) / 10.0,  # H-bond acceptors

            # Ring information
            Descriptors.RingCount(mol) / 5.0,  # Number of rings
            Descriptors.NumAromaticRings(mol) / 3.0,  # Aromatic rings

            # Complexity
            Descriptors.FractionCSP3(mol),  # Fraction of sp3 carbons
            Descriptors.NumHeteroatoms(mol) / 15.0,  # Heteroatom count

            # Charge
            Descriptors.NumValenceElectrons(mol) / 200.0,  # Valence electrons
        ]

        # Replace NaN/Inf with 0
        descriptors = [0.0 if (np.isnan(d) or np.isinf(d)) else d for d in descriptors]

        return descriptors

    except Exception as e:
        return [0.0] * 12


# =============================================================================
# ATOM AND BOND FEATURES
# =============================================================================

def get_atom_features(atom) -> List[float]:
    """
    Compute atom features (39 total).
    """
    # Element type (10)
    elements = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si']
    element_features = [1.0 if atom.GetSymbol() == e else 0.0 for e in elements]

    # Degree (7)
    degree = [0.0] * 7
    degree[min(atom.GetTotalDegree(), 6)] = 1.0

    # Formal charge (5)
    charge = [0.0] * 5
    fc = atom.GetFormalCharge()
    charge_idx = min(max(fc + 2, 0), 4)
    charge[charge_idx] = 1.0

    # Hybridization (5)
    hyb_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    hybridization = [1.0 if atom.GetHybridization() == h else 0.0 for h in hyb_types]

    # Aromaticity (1)
    is_aromatic = [1.0 if atom.GetIsAromatic() else 0.0]

    # Number of H's (5)
    n_hs = [0.0] * 5
    n_hs[min(atom.GetTotalNumHs(), 4)] = 1.0

    # Chirality (4)
    chiral_types = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ]
    chirality = [1.0 if atom.GetChiralTag() == c else 0.0 for c in chiral_types]

    # Ring membership (1)
    is_in_ring = [1.0 if atom.IsInRing() else 0.0]

    # Normalized atomic mass (1)
    atomic_mass = [atom.GetMass() / 100.0]

    features = (element_features + degree + charge + hybridization +
                is_aromatic + n_hs + chirality + is_in_ring + atomic_mass)

    return features


def get_bond_features(bond) -> List[float]:
    """
    Compute bond features (10 total).
    """
    # Bond type (4)
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    bt = [1.0 if bond.GetBondType() == b else 0.0 for b in bond_types]

    # Conjugated and ring (2)
    is_conjugated = [1.0 if bond.GetIsConjugated() else 0.0]
    is_in_ring = [1.0 if bond.IsInRing() else 0.0]

    # Stereo (4)
    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS
    ]
    stereo = [1.0 if bond.GetStereo() == s else 0.0 for s in stereo_types]

    return bt + is_conjugated + is_in_ring + stereo


def smiles_to_graph(smiles: str, temperature: float = 25.0) -> Optional[Data]:
    """Convert SMILES to PyTorch Geometric Data object with global descriptors."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Node features
        atom_features = []
        for atom in mol.GetAtoms():
            feat = get_atom_features(atom)
            atom_features.append(feat)

        if len(atom_features) == 0:
            return None

        x = torch.tensor(atom_features, dtype=torch.float)

        # Edge features
        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feat = get_bond_features(bond)

            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(bond_feat)
            edge_attr.append(bond_feat)

        if len(edge_index) == 0:
            edge_index = [[0, 0]]
            edge_attr = [[0.0] * 10]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Global molecular descriptors (NEW!)
        global_desc = compute_global_descriptors(mol)

        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.temperature = torch.tensor([temperature], dtype=torch.float)
        data.global_features = torch.tensor(global_desc, dtype=torch.float)

        return data

    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None


# =============================================================================
# DATASET
# =============================================================================

class SurfactantDataset:
    """Dataset class for surfactant property prediction."""

    def __init__(self, df: pd.DataFrame, scalers: Dict, targets: List[str]):
        self.df = df
        self.scalers = scalers
        self.targets = targets
        self.num_tasks = len(targets)
        self.graphs = self._prepare_graphs()

    def _prepare_graphs(self) -> List[Data]:
        graphs = []

        for idx, row in self.df.iterrows():
            temp = row.get('temp', 25.0)
            if pd.isna(temp):
                temp = 25.0

            graph = smiles_to_graph(row['SMILES'], temp)
            if graph is None:
                continue

            targets_list = []
            masks_list = []

            for task in self.targets:
                if task in row and pd.notna(row[task]):
                    scaled_value = self.scalers[task].transform([[row[task]]])[0, 0]
                    targets_list.append(float(scaled_value))
                    masks_list.append(1.0)
                else:
                    targets_list.append(0.0)
                    masks_list.append(0.0)

            graph.targets = torch.tensor(targets_list, dtype=torch.float)
            graph.masks = torch.tensor(masks_list, dtype=torch.float)

            graphs.append(graph)

        return graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# =============================================================================
# ENHANCED MODEL
# =============================================================================

class EnhancedMTLGNN(nn.Module):
    """
    Enhanced Multi-Task GNN with:
    1. AttentiveFP encoder
    2. Global descriptor integration
    3. Temperature encoding
    4. Task-specific heads with residual connections
    """

    def __init__(
            self,
            in_channels: int = 39,
            hidden_channels: int = 128,
            out_channels: int = 256,
            edge_dim: int = 10,
            num_layers: int = 3,
            num_timesteps: int = 2,
            dropout: float = 0.15,
            num_tasks: int = 6,
            temp_embed_dim: int = 64,
            num_global_features: int = 12,
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.dropout = dropout

        # AttentiveFP encoder
        self.encoder = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

        # Temperature encoder
        self.temp_encoder = nn.Sequential(
            nn.Linear(1, temp_embed_dim),
            nn.GELU(),  # GELU often works better than ReLU
            nn.Dropout(dropout),
            nn.Linear(temp_embed_dim, temp_embed_dim),
        )

        # Global descriptor encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(num_global_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
        )

        # Fusion layer
        fusion_dim = out_channels + temp_embed_dim + 64  # graph + temp + global
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Shared representation layer
        self.shared = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )

        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, batch):
        # Graph encoding
        graph_embed = self.encoder(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

        batch_size = graph_embed.shape[0]

        # Temperature encoding
        temp = batch.temperature.view(batch_size, 1)
        temp_norm = (temp - 25.0) / 35.0
        temp_embed = self.temp_encoder(temp_norm)

        # Global descriptor encoding
        global_feat = batch.global_features.view(batch_size, -1)
        global_embed = self.global_encoder(global_feat)

        # Fusion
        combined = torch.cat([graph_embed, temp_embed, global_embed], dim=1)
        fused = self.fusion(combined)

        # Shared representation
        shared = self.shared(fused)

        # Task-specific predictions
        outputs = []
        for head in self.task_heads:
            outputs.append(head(shared))

        output = torch.cat(outputs, dim=1)

        return output


# =============================================================================
# WEIGHTED LOSS FUNCTION
# =============================================================================

class WeightedMaskedMSELoss(nn.Module):
    """MSE loss with task-specific weighting and masking."""

    def __init__(self, task_weights: Dict[str, float], targets: List[str]):
        super().__init__()
        # Convert dict to tensor
        weights = [task_weights.get(t, 1.0) for t in targets]
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape or pred.shape != mask.shape:
            raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}, mask={mask.shape}")

        # Compute squared error
        squared_error = (pred - target) ** 2

        # Apply task weights
        weighted_error = squared_error * self.weights.unsqueeze(0)

        # Apply mask
        masked_error = weighted_error * mask

        # Compute mean
        n_valid = mask.sum()
        if n_valid > 0:
            loss = masked_error.sum() / n_valid
        else:
            loss = torch.tensor(0.0, device=pred.device)

        return loss


# =============================================================================
# LEARNING RATE SCHEDULER WITH WARMUP
# =============================================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create a scheduler with linear warmup and cosine decay."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, loader, optimizer, scheduler, criterion, device, num_tasks, accumulation_steps=2):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)

        pred = model(batch)
        batch_size = pred.shape[0]

        targets = batch.targets.view(batch_size, num_tasks)
        masks = batch.masks.view(batch_size, num_tasks)

        loss = criterion(pred, targets, masks)
        loss = loss / accumulation_steps  # Normalize for accumulation

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

    # Handle remaining gradients
    if num_batches % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    if num_batches == 0:
        return float('nan')

    return total_loss / num_batches


def evaluate(model, loader, device, scalers, targets, num_tasks, unit_conversions):
    """Evaluate model with proper unit conversions."""
    model.eval()

    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            batch_size = pred.shape[0]

            batch_targets = batch.targets.view(batch_size, num_tasks)
            batch_masks = batch.masks.view(batch_size, num_tasks)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())
            all_masks.append(batch_masks.cpu().numpy())

    preds = np.vstack(all_preds)
    targets_arr = np.vstack(all_targets)
    masks = np.vstack(all_masks)

    metrics = {}
    for i, task in enumerate(targets):
        mask = masks[:, i] > 0.5
        n_samples = mask.sum()

        if n_samples < 2:
            continue

        y_true_original = scalers[task].inverse_transform(
            targets_arr[mask, i].reshape(-1, 1)
        ).flatten()
        y_pred_original = scalers[task].inverse_transform(
            preds[mask, i].reshape(-1, 1)
        ).flatten()

        conversion = unit_conversions.get(task, 1.0)
        y_true = y_true_original * conversion
        y_pred = y_pred_original * conversion

        metrics[task] = {
            'r2': float(r2_score(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'n': int(n_samples),
        }

    return metrics, preds


def train_single_model(model, train_loader, val_loader, device, config, scalers, criterion):
    """Train a single model with enhanced settings."""

    # Calculate total steps for scheduler
    total_steps = len(train_loader) * config.MAX_EPOCHS
    warmup_steps = len(train_loader) * config.WARMUP_EPOCHS

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(config.MAX_EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, config.NUM_TASKS, accumulation_steps=2
        )

        if np.isnan(train_loss):
            break

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                batch_size = pred.shape[0]

                targets = batch.targets.view(batch_size, config.NUM_TASKS)
                masks = batch.masks.view(batch_size, config.NUM_TASKS)

                loss = criterion(pred, targets, masks)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_batches += 1

        if val_batches == 0:
            break

        val_loss /= val_batches

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, best_val_loss


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(data_dir: str, output_dir: str = None, device: str = 'cuda'):
    """Main training function with enhancements."""

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"experiments/enhanced_mtl_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Print header
    print("=" * 70)
    print("ENHANCED TEMPERATURE-AWARE MULTI-TASK GNN TRAINING")
    print("=" * 70)
    print("\n🚀 IMPROVEMENTS:")
    print("  1. Task-specific loss weighting")
    print("  2. Global molecular descriptors (RDKit)")
    print("  3. Larger ensemble (60 models)")
    print("  4. Learning rate warmup with cosine decay")
    print("  5. Gradient accumulation")
    print("  6. Larger model architecture")

    config = Config()

    print(f"\n📋 Configuration:")
    print(f"  Hidden dim: {config.HIDDEN_DIM} (was 96)")
    print(f"  Output dim: {config.OUTPUT_DIM} (was 192)")
    print(f"  Num layers: {config.NUM_LAYERS} (was 2)")
    print(f"  N ensemble: {config.N_ENSEMBLE} (was 40)")
    print(f"  Warmup epochs: {config.WARMUP_EPOCHS}")
    print(f"  Task weights: {config.TASK_WEIGHTS}")

    # Load data
    print(f"\n📂 Loading data from {data_dir}...")
    train_df = pd.read_csv(f"{data_dir}/surfpro_train.csv")
    test_df = pd.read_csv(f"{data_dir}/surfpro_test.csv")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")

    # Setup scalers
    print("\n📊 Setting up RobustScaler...")
    scalers = {}
    for task in config.TARGETS:
        if task in train_df.columns:
            scalers[task] = RobustScaler()
            valid_mask = train_df[task].notna()
            if valid_mask.sum() > 0:
                scalers[task].fit(train_df.loc[valid_mask, task].values.reshape(-1, 1))
                print(f"  {task}: {valid_mask.sum()} samples (weight: {config.TASK_WEIGHTS[task]})")

    # Prepare datasets
    print("\n📦 Preparing datasets with global descriptors...")
    train_dataset = SurfactantDataset(train_df, scalers, config.TARGETS)
    test_dataset = SurfactantDataset(test_df, scalers, config.TARGETS)
    print(f"  Train graphs: {len(train_dataset)}")
    print(f"  Test graphs: {len(test_dataset)}")

    # Validate data
    print("\n🔍 Validating data...")
    sample_graph = train_dataset[0]
    print(f"  Sample graph:")
    print(f"    Node features shape: {sample_graph.x.shape}")
    print(f"    Edge index shape: {sample_graph.edge_index.shape}")
    print(f"    Global features shape: {sample_graph.global_features.shape}")
    print(f"    Temperature: {sample_graph.temperature}")

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Device: {device}")

    # Create weighted loss function
    criterion = WeightedMaskedMSELoss(config.TASK_WEIGHTS, config.TARGETS).to(device)

    # Training
    print(f"\n🚀 Training {config.N_ENSEMBLE}-model ensemble...")

    all_test_preds = []
    all_val_metrics = []

    train_graphs = train_dataset.graphs
    test_graphs = test_dataset.graphs

    kfold = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    models_per_fold = config.N_ENSEMBLE // config.N_FOLDS

    model_count = 0

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_graphs)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{config.N_FOLDS}")
        print(f"{'=' * 50}")

        train_subset = [train_graphs[i] for i in train_idx]
        val_subset = [train_graphs[i] for i in val_idx]

        print(f"  Train: {len(train_subset)}, Val: {len(val_subset)}")

        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=config.BATCH_SIZE, shuffle=False)

        for model_idx in range(models_per_fold):
            model_count += 1
            seed = fold * models_per_fold + model_idx

            torch.manual_seed(seed)
            np.random.seed(seed)

            model = EnhancedMTLGNN(
                in_channels=39,
                hidden_channels=config.HIDDEN_DIM,
                out_channels=config.OUTPUT_DIM,
                edge_dim=10,
                num_layers=config.NUM_LAYERS,
                num_timesteps=config.NUM_TIMESTEPS,
                dropout=config.DROPOUT,
                num_tasks=config.NUM_TASKS,
                temp_embed_dim=config.TEMP_EMBED_DIM,
                num_global_features=config.NUM_GLOBAL_FEATURES,
            ).to(device)

            model, best_loss = train_single_model(
                model, train_loader, val_loader,
                device, config, scalers, criterion
            )

            val_metrics, _ = evaluate(
                model, val_loader, device, scalers,
                config.TARGETS, config.NUM_TASKS, config.UNIT_CONVERSIONS
            )
            all_val_metrics.append(val_metrics)

            # Get test predictions
            model.eval()
            test_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    test_preds.append(pred.cpu().numpy())

            test_preds = np.vstack(test_preds)
            all_test_preds.append(test_preds)

            pcmc_r2 = val_metrics.get('pCMC', {}).get('r2', 0)
            gamma_r2 = val_metrics.get('Gamma_max', {}).get('r2', 0)
            print(f"  Model {model_count:2d}/{config.N_ENSEMBLE}: pCMC R²={pcmc_r2:.4f}, Γmax R²={gamma_r2:.4f}")

    # Ensemble predictions
    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS")
    print("=" * 70)

    ensemble_preds = np.mean(all_test_preds, axis=0)
    ensemble_std = np.std(all_test_preds, axis=0)

    # Get test targets
    test_targets = np.vstack([g.targets.numpy() for g in test_graphs])
    test_masks = np.vstack([g.masks.numpy() for g in test_graphs])

    # Compute final metrics
    print("\n📊 Test Set Performance:")
    print("-" * 70)
    print(f"{'Property':<12} {'Unit':<12} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'n':<6}")
    print("-" * 70)

    final_metrics = {}

    for i, task in enumerate(config.TARGETS):
        mask = test_masks[:, i] > 0.5
        n_samples = int(mask.sum())

        if n_samples < 2:
            print(f"{task:<12} Insufficient samples ({n_samples})")
            continue

        y_true_original = scalers[task].inverse_transform(
            test_targets[mask, i].reshape(-1, 1)
        ).flatten()
        y_pred_original = scalers[task].inverse_transform(
            ensemble_preds[mask, i].reshape(-1, 1)
        ).flatten()

        conversion = config.UNIT_CONVERSIONS.get(task, 1.0)
        y_true = y_true_original * conversion
        y_pred = y_pred_original * conversion

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        unit = config.UNIT_NAMES.get(task, '')

        final_metrics[task] = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'n': n_samples,
            'unit': unit
        }

        print(f"{task:<12} {unit:<12} {r2:.4f}   {rmse:<10.4f} {mae:<10.4f} {n_samples:<6}")

    print("-" * 70)

    # Overall metrics
    all_r2 = [m['r2'] for m in final_metrics.values()]
    overall_r2 = np.mean(all_r2)
    overall_std_r2 = np.std(all_r2)

    print(f"{'Overall':<12} {'':<12} {overall_r2:.4f} ± {overall_std_r2:.4f}")
    print("-" * 70)

    # Comparison with baseline and literature
    print("\n📊 Comparison:")
    print(f"  {'Metric':<20} {'Baseline':<12} {'Enhanced':<12} {'Improvement':<12}")
    print(f"  {'-' * 56}")

    baseline_r2 = {
        'pCMC': 0.898, 'AW_ST_CMC': 0.812, 'Gamma_max': 0.769,
        'Area_min': 0.857, 'Pi_CMC': 0.806, 'pC20': 0.897
    }

    for task in config.TARGETS:
        if task in final_metrics:
            base = baseline_r2.get(task, 0)
            curr = final_metrics[task]['r2']
            diff = curr - base
            sign = '+' if diff >= 0 else ''
            print(f"  {task:<20} {base:.4f}       {curr:.4f}       {sign}{diff:.4f}")

    print(f"  {'-' * 56}")
    print(
        f"  {'Overall':<20} 0.8399       {overall_r2:.4f}       {'+' if overall_r2 > 0.8399 else ''}{overall_r2 - 0.8399:.4f}")

    # Target assessment
    print("\n🎯 Target Assessment:")
    pcmc_r2 = final_metrics.get('pCMC', {}).get('r2', 0)
    gamma_r2 = final_metrics.get('Gamma_max', {}).get('r2', 0)

    print(f"  Overall R² > 0.87:  {'✅ YES' if overall_r2 > 0.87 else '❌ NO'} ({overall_r2:.4f})")
    print(f"  pCMC R² > 0.91:     {'✅ YES' if pcmc_r2 > 0.91 else '❌ NO'} ({pcmc_r2:.4f})")
    print(f"  Gamma_max R² > 0.82: {'✅ YES' if gamma_r2 > 0.82 else '❌ NO'} ({gamma_r2:.4f})")

    # Save results
    results = {
        'config': {
            'hidden_dim': config.HIDDEN_DIM,
            'output_dim': config.OUTPUT_DIM,
            'num_layers': config.NUM_LAYERS,
            'dropout': config.DROPOUT,
            'n_ensemble': config.N_ENSEMBLE,
            'n_folds': config.N_FOLDS,
            'task_weights': config.TASK_WEIGHTS,
            'warmup_epochs': config.WARMUP_EPOCHS,
        },
        'final_metrics': final_metrics,
        'overall': {
            'r2_mean': float(overall_r2),
            'r2_std': float(overall_std_r2)
        },
        'baseline_comparison': {
            task: {
                'baseline': baseline_r2.get(task, 0),
                'enhanced': final_metrics.get(task, {}).get('r2', 0),
                'improvement': final_metrics.get(task, {}).get('r2', 0) - baseline_r2.get(task, 0)
            }
            for task in config.TARGETS
        }
    }

    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    np.save(output_path / 'ensemble_predictions.npy', ensemble_preds)
    np.save(output_path / 'ensemble_std.npy', ensemble_std)
    np.save(output_path / 'test_targets.npy', test_targets)
    np.save(output_path / 'test_masks.npy', test_masks)
    np.save(output_path / 'all_test_preds.npy', np.array(all_test_preds))

    print(f"\n✅ Results saved to: {output_path}")

    # Print LaTeX table
    print("\n" + "=" * 70)
    print("LATEX TABLE FOR JCIM PAPER")
    print("=" * 70)
    print(r"""
\begin{table}[htbp]
\centering
\caption{Test set performance of the enhanced temperature-aware MTL-GNN ensemble model.}
\label{tab:results}
\begin{tabular}{lccccc}
\toprule
Property & Unit & R$^2$ & RMSE & MAE & n \\
\midrule""")

    for task in config.TARGETS:
        if task in final_metrics:
            m = final_metrics[task]
            if task == 'pCMC':
                task_latex = 'pCMC'
                unit_latex = r'$-\log_{10}$(M)'
            elif task == 'AW_ST_CMC':
                task_latex = r'$\gamma_{\text{CMC}}$'
                unit_latex = 'mN/m'
            elif task == 'Gamma_max':
                task_latex = r'$\Gamma_{\text{max}}$'
                unit_latex = r'$\mu$mol/m$^2$'
            elif task == 'Area_min':
                task_latex = r'$A_{\text{min}}$'
                unit_latex = r'nm$^2$'
            elif task == 'Pi_CMC':
                task_latex = r'$\pi_{\text{CMC}}$'
                unit_latex = 'mN/m'
            elif task == 'pC20':
                task_latex = r'pC$_{20}$'
                unit_latex = r'$-\log_{10}$(M)'
            else:
                task_latex = task
                unit_latex = m['unit']

            print(f"{task_latex} & {unit_latex} & {m['r2']:.3f} & {m['rmse']:.3f} & {m['mae']:.3f} & {m['n']} \\\\")

    print(r"""\midrule
Overall & -- & """ + f"{overall_r2:.3f} $\\pm$ {overall_std_r2:.3f}" + r""" & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
""")

    return final_metrics


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Enhanced Temperature-Aware Multi-Task GNN Training'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing surfpro_train.csv and surfpro_test.csv'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device: cuda or cpu'
    )

    args = parser.parse_args()

    main(args.data_dir, args.output_dir, args.device)