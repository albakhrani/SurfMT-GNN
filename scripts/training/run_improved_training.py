#!/usr/bin/env python3
"""
Complete Training Script for Temperature-Aware Multi-Task GNN
==============================================================
Based on Hödl et al. 2025 (SurfPro) optimal hyperparameters.

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)

Run:
    python run_improved_training.py --data_dir data/raw --device cuda

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

warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyTorch Geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP

# Scikit-learn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration based on Hödl et al. 2025 (SurfPro paper)."""

    # Model Architecture
    HIDDEN_DIM = 96
    OUTPUT_DIM = 192
    NUM_LAYERS = 2
    NUM_TIMESTEPS = 2
    DROPOUT = 0.1

    # Temperature encoding
    TEMP_EMBED_DIM = 64

    # Training
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 32
    MAX_EPOCHS = 300
    PATIENCE = 60

    # Ensemble
    N_ENSEMBLE = 40
    N_FOLDS = 10

    # Tasks
    TARGETS = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']
    NUM_TASKS = 6

    # Unit conversions for proper reporting
    # Gamma_max is stored in mol/m² but should be reported in μmol/m²
    UNIT_CONVERSIONS = {
        'pCMC': 1.0,  # Already in -log10(M)
        'AW_ST_CMC': 1.0,  # mN/m
        'Gamma_max': 1e6,  # mol/m² → μmol/m² (multiply by 1e6)
        'Area_min': 1.0,  # nm²
        'Pi_CMC': 1.0,  # mN/m
        'pC20': 1.0,  # -log10(M)
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
# ATOM AND BOND FEATURES
# =============================================================================

def get_atom_features(atom) -> List[float]:
    """
    Compute atom features (39 total).

    Features:
        - Element type (10): C, N, O, S, F, Cl, Br, I, P, Si
        - Degree (7): 0-6
        - Formal charge (5): -2 to +2
        - Hybridization (5): SP, SP2, SP3, SP3D, SP3D2
        - Aromaticity (1): is aromatic
        - Num H's (5): 0-4
        - Chirality (4): unspecified, CW, CCW, other
        - Ring membership (1): is in ring
        - Atomic mass (1): normalized mass
    """
    # Element type (10)
    elements = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si']
    element_features = [1.0 if atom.GetSymbol() == e else 0.0 for e in elements]

    # Degree (7)
    degree = [0.0] * 7
    degree[min(atom.GetTotalDegree(), 6)] = 1.0

    # Formal charge (5): -2, -1, 0, +1, +2
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

    Features:
        - Bond type (4): single, double, triple, aromatic
        - Is conjugated (1)
        - Is in ring (1)
        - Stereo (4): none, Z, E, CIS
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
    """Convert SMILES to PyTorch Geometric Data object."""
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

            # Add both directions
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(bond_feat)
            edge_attr.append(bond_feat)

        # Handle molecules with no bonds (single atoms)
        if len(edge_index) == 0:
            edge_index = [[0, 0]]
            edge_attr = [[0.0] * 10]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.temperature = torch.tensor([temperature], dtype=torch.float)

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
        """Convert DataFrame to list of graph Data objects."""
        graphs = []

        for idx, row in self.df.iterrows():
            # Get temperature (default 25°C)
            temp = row.get('temp', 25.0)
            if pd.isna(temp):
                temp = 25.0

            # Convert SMILES to graph
            graph = smiles_to_graph(row['SMILES'], temp)
            if graph is None:
                continue

            # Prepare targets and masks
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
# MODEL
# =============================================================================

class TemperatureAwareMTLGNN(nn.Module):
    """
    Temperature-Aware Multi-Task Graph Neural Network.

    Architecture:
        1. AttentiveFP encoder for molecular graphs
        2. MLP encoder for temperature
        3. Feature fusion layer
        4. Multi-task prediction heads
    """

    def __init__(
            self,
            in_channels: int = 39,
            hidden_channels: int = 96,
            out_channels: int = 192,
            edge_dim: int = 10,
            num_layers: int = 2,
            num_timesteps: int = 2,
            dropout: float = 0.1,
            num_tasks: int = 6,
            temp_embed_dim: int = 64,
    ):
        super().__init__()

        self.num_tasks = num_tasks

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
            nn.ReLU(),
            nn.Linear(temp_embed_dim, temp_embed_dim),
        )

        # Fusion layer
        fusion_dim = out_channels + temp_embed_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_tasks),
        )

    def forward(self, batch):
        # Graph encoding
        graph_embed = self.encoder(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

        # Get batch size
        batch_size = graph_embed.shape[0]

        # Temperature encoding
        temp = batch.temperature.view(batch_size, 1)
        temp_norm = (temp - 25.0) / 35.0
        temp_embed = self.temp_encoder(temp_norm)

        # Fusion
        combined = torch.cat([graph_embed, temp_embed], dim=1)
        fused = self.fusion(combined)

        # Prediction
        output = self.head(fused)

        return output


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class MaskedMSELoss(nn.Module):
    """MSE loss with masking for missing values."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape or pred.shape != mask.shape:
            raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}, mask={mask.shape}")

        squared_error = (pred - target) ** 2
        masked_error = squared_error * mask

        n_valid = mask.sum()
        if n_valid > 0:
            loss = masked_error.sum() / n_valid
        else:
            loss = torch.tensor(0.0, device=pred.device)

        return loss


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, num_tasks):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        batch_size = pred.shape[0]

        targets = batch.targets.view(batch_size, num_tasks)
        masks = batch.masks.view(batch_size, num_tasks)

        loss = criterion(pred, targets, masks)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return float('nan')

    return total_loss / num_batches


def evaluate(model, loader, device, scalers, targets, num_tasks, unit_conversions):
    """Evaluate model and return metrics with proper unit conversions."""
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

        # Inverse transform to original scale
        y_true_original = scalers[task].inverse_transform(
            targets_arr[mask, i].reshape(-1, 1)
        ).flatten()
        y_pred_original = scalers[task].inverse_transform(
            preds[mask, i].reshape(-1, 1)
        ).flatten()

        # Apply unit conversion (important for Gamma_max)
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


def train_single_model(model, train_loader, val_loader, device, config, scalers):
    """Train a single model."""

    optimizer = Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20
    )
    criterion = MaskedMSELoss()

    best_val_loss = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(config.MAX_EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, config.NUM_TASKS
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
        scheduler.step(val_loss)

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
    """Main training function."""

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"experiments/improved_mtl_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Print header
    print("=" * 70)
    print("TEMPERATURE-AWARE MULTI-TASK GNN TRAINING")
    print("Based on Hödl et al. 2025 optimal settings")
    print("=" * 70)

    config = Config()

    print(f"\n📋 Configuration:")
    print(f"  Hidden dim: {config.HIDDEN_DIM}")
    print(f"  Num layers: {config.NUM_LAYERS}")
    print(f"  Dropout: {config.DROPOUT}")
    print(f"  N ensemble: {config.N_ENSEMBLE}")
    print(f"  N folds: {config.N_FOLDS}")
    print(f"  Targets: {config.TARGETS}")

    # Load data
    print(f"\n📂 Loading data from {data_dir}...")
    train_df = pd.read_csv(f"{data_dir}/surfpro_train.csv")
    test_df = pd.read_csv(f"{data_dir}/surfpro_test.csv")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")

    # Check columns
    print(f"\n📋 Available columns: {train_df.columns.tolist()}")

    # Setup scalers
    print("\n📊 Setting up RobustScaler...")
    scalers = {}
    for task in config.TARGETS:
        if task in train_df.columns:
            scalers[task] = RobustScaler()
            valid_mask = train_df[task].notna()
            if valid_mask.sum() > 0:
                scalers[task].fit(train_df.loc[valid_mask, task].values.reshape(-1, 1))
                print(f"  {task}: {valid_mask.sum()} samples")
        else:
            print(f"  WARNING: {task} not found in columns!")

    # Prepare datasets
    print("\n📦 Preparing datasets...")
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
    print(f"    Edge attr shape: {sample_graph.edge_attr.shape}")
    print(f"    Temperature: {sample_graph.temperature}")
    print(f"    Targets: {sample_graph.targets}")
    print(f"    Masks: {sample_graph.masks}")

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Device: {device}")

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

            model = TemperatureAwareMTLGNN(
                in_channels=39,
                hidden_channels=config.HIDDEN_DIM,
                out_channels=config.OUTPUT_DIM,
                edge_dim=10,
                num_layers=config.NUM_LAYERS,
                num_timesteps=config.NUM_TIMESTEPS,
                dropout=config.DROPOUT,
                num_tasks=config.NUM_TASKS,
                temp_embed_dim=config.TEMP_EMBED_DIM,
            ).to(device)

            model, best_loss = train_single_model(
                model, train_loader, val_loader,
                device, config, scalers
            )

            # Evaluate with proper unit conversions
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
            print(f"  Model {model_count:2d}/{config.N_ENSEMBLE}: Val pCMC R² = {pcmc_r2:.4f}")

    # Ensemble predictions
    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS")
    print("=" * 70)

    ensemble_preds = np.mean(all_test_preds, axis=0)
    ensemble_std = np.std(all_test_preds, axis=0)

    # Get test targets
    test_targets = np.vstack([g.targets.numpy() for g in test_graphs])
    test_masks = np.vstack([g.masks.numpy() for g in test_graphs])

    # Compute final metrics with proper units
    print("\n📊 Test Set Performance:")
    print("-" * 60)
    print(f"{'Property':<12} {'Unit':<12} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'n':<6}")
    print("-" * 60)

    final_metrics = {}

    for i, task in enumerate(config.TARGETS):
        mask = test_masks[:, i] > 0.5
        n_samples = int(mask.sum())

        if n_samples < 2:
            print(f"{task:<12} Insufficient samples ({n_samples})")
            continue

        # Inverse transform
        y_true_original = scalers[task].inverse_transform(
            test_targets[mask, i].reshape(-1, 1)
        ).flatten()
        y_pred_original = scalers[task].inverse_transform(
            ensemble_preds[mask, i].reshape(-1, 1)
        ).flatten()

        # Apply unit conversion (important for Gamma_max: mol/m² → μmol/m²)
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

    print("-" * 60)

    # Overall metrics
    all_r2 = [m['r2'] for m in final_metrics.values()]
    overall_r2 = np.mean(all_r2)
    overall_std = np.std(all_r2)

    print(f"{'Overall':<12} {'':<12} {overall_r2:.4f} ± {overall_std:.4f}")
    print("-" * 60)

    # Comparison with literature
    pcmc_rmse = final_metrics.get('pCMC', {}).get('rmse', float('inf'))
    pcmc_r2 = final_metrics.get('pCMC', {}).get('r2', 0)

    print("\n📊 Comparison with Literature:")
    print(f"  Your pCMC RMSE: {pcmc_rmse:.3f} (Hödl et al.: 0.24)")
    print(f"  Your pCMC R²:   {pcmc_r2:.3f} (Hödl et al.: 0.94)")

    if pcmc_rmse < 0.30:
        print("\n✅ EXCELLENT! Target achieved.")
    elif pcmc_rmse < 0.40:
        print("\n🟡 GOOD. Competitive performance.")
    else:
        print("\n🔴 Needs improvement.")

    # Save results
    results = {
        'config': {
            'hidden_dim': config.HIDDEN_DIM,
            'output_dim': config.OUTPUT_DIM,
            'num_layers': config.NUM_LAYERS,
            'dropout': config.DROPOUT,
            'n_ensemble': config.N_ENSEMBLE,
            'n_folds': config.N_FOLDS,
        },
        'final_metrics': final_metrics,
        'overall': {
            'r2_mean': float(overall_r2),
            'r2_std': float(overall_std)
        },
        'unit_conversions': config.UNIT_CONVERSIONS,
        'unit_names': config.UNIT_NAMES,
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
    print("LATEX TABLE (Copy for JCIM Paper)")
    print("=" * 70)
    print(r"""
\begin{table}[htbp]
\centering
\caption{Test set performance of the temperature-aware MTL-GNN ensemble model.}
\label{tab:results}
\begin{tabular}{lccccc}
\toprule
Property & Unit & R$^2$ & RMSE & MAE & n \\
\midrule""")

    for task in config.TARGETS:
        if task in final_metrics:
            m = final_metrics[task]
            # Format task name for LaTeX
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
Overall & -- & """ + f"{overall_r2:.3f} $\\pm$ {overall_std:.3f}" + r""" & -- & -- & -- \\
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
        description='Train Temperature-Aware Multi-Task GNN for Surfactant Properties'
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
        help='Output directory (default: experiments/improved_mtl_TIMESTAMP)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device: cuda or cpu'
    )

    args = parser.parse_args()

    main(args.data_dir, args.output_dir, args.device)