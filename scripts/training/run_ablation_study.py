#!/usr/bin/env python3
"""
=============================================================================
ABLATION STUDY FOR SurfMT-GNN
=============================================================================
This script runs all ablation experiments required for JCIM paper:

1. Full Model (baseline) - Already trained
2. No Temperature Encoding - Remove temperature branch
3. No Global Descriptors - Remove descriptor branch
4. Equal Task Weights - All weights = 1.0
5. Single Model - No ensemble, just 1 model

Output:
- Ablation results table (plain text + LaTeX)
- Comparison figures
- JSON results for reproducibility

Author: Al-Futini Abdulhakim Nasser Ali
For: JCIM Paper - SurfMT-GNN

Run:
    python run_ablation_study.py --data_dir data/raw --device cuda
=============================================================================
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
from torch.optim.lr_scheduler import LambdaLR

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
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for ablation study."""

    # Model Architecture (same as paper)
    HIDDEN_DIM = 128
    OUTPUT_DIM = 256
    NUM_LAYERS = 3
    NUM_TIMESTEPS = 2
    DROPOUT = 0.15

    # Global descriptors
    NUM_GLOBAL_FEATURES = 12

    # Temperature encoding
    TEMP_EMBED_DIM = 64

    # Training
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 32
    MAX_EPOCHS = 400
    PATIENCE = 80
    WARMUP_EPOCHS = 20

    # Ablation settings - smaller ensemble for faster ablation
    N_ENSEMBLE_ABLATION = 20  # 10 folds × 2 models (faster)
    N_FOLDS = 10

    # Full ensemble for final comparison
    N_ENSEMBLE_FULL = 60  # 10 folds × 6 models

    # Tasks
    TARGETS = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']
    NUM_TASKS = 6

    # Task weights (from paper)
    TASK_WEIGHTS = {
        'pCMC': 1.0,
        'AW_ST_CMC': 1.3,
        'Gamma_max': 1.5,
        'Area_min': 1.1,
        'Pi_CMC': 1.3,
        'pC20': 1.0,
    }

    # Equal weights for ablation
    EQUAL_WEIGHTS = {
        'pCMC': 1.0,
        'AW_ST_CMC': 1.0,
        'Gamma_max': 1.0,
        'Area_min': 1.0,
        'Pi_CMC': 1.0,
        'pC20': 1.0,
    }

    # Unit conversions
    UNIT_CONVERSIONS = {
        'pCMC': 1.0,
        'AW_ST_CMC': 1.0,
        'Gamma_max': 1e6,
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
# FEATURE FUNCTIONS
# =============================================================================

def compute_global_descriptors(mol) -> List[float]:
    """Compute 12 RDKit descriptors."""
    try:
        descriptors = [
            Descriptors.MolWt(mol) / 500.0,
            Descriptors.HeavyAtomCount(mol) / 50.0,
            rdMolDescriptors.CalcNumRotatableBonds(mol) / 20.0,
            Descriptors.TPSA(mol) / 150.0,
            Descriptors.MolLogP(mol) / 10.0,
            Descriptors.NumHDonors(mol) / 10.0,
            Descriptors.NumHAcceptors(mol) / 10.0,
            Descriptors.RingCount(mol) / 5.0,
            Descriptors.NumAromaticRings(mol) / 3.0,
            Descriptors.FractionCSP3(mol),
            Descriptors.NumHeteroatoms(mol) / 15.0,
            Descriptors.NumValenceElectrons(mol) / 200.0,
        ]
        descriptors = [0.0 if (np.isnan(d) or np.isinf(d)) else d for d in descriptors]
        return descriptors
    except:
        return [0.0] * 12


def get_atom_features(atom) -> List[float]:
    """Compute atom features (39 total)."""
    elements = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si']
    element_features = [1.0 if atom.GetSymbol() == e else 0.0 for e in elements]

    degree = [0.0] * 7
    degree[min(atom.GetTotalDegree(), 6)] = 1.0

    charge = [0.0] * 5
    fc = atom.GetFormalCharge()
    charge_idx = min(max(fc + 2, 0), 4)
    charge[charge_idx] = 1.0

    hyb_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    hybridization = [1.0 if atom.GetHybridization() == h else 0.0 for h in hyb_types]

    is_aromatic = [1.0 if atom.GetIsAromatic() else 0.0]

    n_hs = [0.0] * 5
    n_hs[min(atom.GetTotalNumHs(), 4)] = 1.0

    chiral_types = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ]
    chirality = [1.0 if atom.GetChiralTag() == c else 0.0 for c in chiral_types]

    is_in_ring = [1.0 if atom.IsInRing() else 0.0]
    atomic_mass = [atom.GetMass() / 100.0]

    features = (element_features + degree + charge + hybridization +
                is_aromatic + n_hs + chirality + is_in_ring + atomic_mass)
    return features


def get_bond_features(bond) -> List[float]:
    """Compute bond features (10 total)."""
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    bt = [1.0 if bond.GetBondType() == b else 0.0 for b in bond_types]

    is_conjugated = [1.0 if bond.GetIsConjugated() else 0.0]
    is_in_ring = [1.0 if bond.IsInRing() else 0.0]

    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS
    ]
    stereo = [1.0 if bond.GetStereo() == s else 0.0 for s in stereo_types]

    return bt + is_conjugated + is_in_ring + stereo


def smiles_to_graph(smiles: str, temperature: float = 25.0, include_global: bool = True) -> Optional[Data]:
    """Convert SMILES to graph."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        atom_features = []
        for atom in mol.GetAtoms():
            feat = get_atom_features(atom)
            atom_features.append(feat)

        if len(atom_features) == 0:
            return None

        x = torch.tensor(atom_features, dtype=torch.float)

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

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.temperature = torch.tensor([temperature], dtype=torch.float)

        if include_global:
            global_desc = compute_global_descriptors(mol)
            data.global_features = torch.tensor(global_desc, dtype=torch.float)
        else:
            data.global_features = torch.zeros(12, dtype=torch.float)

        return data
    except Exception as e:
        return None


# =============================================================================
# DATASET
# =============================================================================

class SurfactantDataset:
    """Dataset class."""

    def __init__(self, df: pd.DataFrame, scalers: Dict, targets: List[str], include_global: bool = True):
        self.df = df
        self.scalers = scalers
        self.targets = targets
        self.include_global = include_global
        self.graphs = self._prepare_graphs()

    def _prepare_graphs(self) -> List[Data]:
        graphs = []
        for idx, row in self.df.iterrows():
            temp = row.get('temp', 25.0)
            if pd.isna(temp):
                temp = 25.0

            graph = smiles_to_graph(row['SMILES'], temp, self.include_global)
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
# MODEL VARIANTS FOR ABLATION
# =============================================================================

class FullModel(nn.Module):
    """Full SurfMT-GNN model (baseline)."""

    def __init__(self, config, use_temperature=True, use_descriptors=True):
        super().__init__()

        self.use_temperature = use_temperature
        self.use_descriptors = use_descriptors
        self.num_tasks = config.NUM_TASKS

        # Graph encoder
        self.encoder = AttentiveFP(
            in_channels=39,
            hidden_channels=config.HIDDEN_DIM,
            out_channels=config.OUTPUT_DIM,
            edge_dim=10,
            num_layers=config.NUM_LAYERS,
            num_timesteps=config.NUM_TIMESTEPS,
            dropout=config.DROPOUT,
        )

        # Temperature encoder
        if use_temperature:
            self.temp_encoder = nn.Sequential(
                nn.Linear(1, config.TEMP_EMBED_DIM),
                nn.GELU(),
                nn.Dropout(config.DROPOUT),
                nn.Linear(config.TEMP_EMBED_DIM, config.TEMP_EMBED_DIM),
            )
            temp_dim = config.TEMP_EMBED_DIM
        else:
            self.temp_encoder = None
            temp_dim = 0

        # Descriptor encoder
        if use_descriptors:
            self.global_encoder = nn.Sequential(
                nn.Linear(config.NUM_GLOBAL_FEATURES, 64),
                nn.GELU(),
                nn.Dropout(config.DROPOUT),
                nn.Linear(64, 64),
            )
            desc_dim = 64
        else:
            self.global_encoder = None
            desc_dim = 0

        # Fusion
        fusion_dim = config.OUTPUT_DIM + temp_dim + desc_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, config.OUTPUT_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
        )

        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(config.OUTPUT_DIM, 128),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 128),
        )

        # Task heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(config.DROPOUT),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            for _ in range(config.NUM_TASKS)
        ])

    def forward(self, batch):
        # Graph encoding
        graph_embed = self.encoder(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        batch_size = graph_embed.shape[0]

        embeddings = [graph_embed]

        # Temperature encoding
        if self.use_temperature and self.temp_encoder is not None:
            temp = batch.temperature.view(batch_size, 1)
            temp_norm = (temp - 25.0) / 35.0
            temp_embed = self.temp_encoder(temp_norm)
            embeddings.append(temp_embed)

        # Descriptor encoding
        if self.use_descriptors and self.global_encoder is not None:
            global_feat = batch.global_features.view(batch_size, -1)
            global_embed = self.global_encoder(global_feat)
            embeddings.append(global_embed)

        # Fusion
        combined = torch.cat(embeddings, dim=1)
        fused = self.fusion(combined)

        # Shared
        shared = self.shared(fused)

        # Task predictions
        outputs = [head(shared) for head in self.task_heads]
        output = torch.cat(outputs, dim=1)

        return output


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class WeightedMaskedMSELoss(nn.Module):
    """Weighted masked MSE loss."""

    def __init__(self, task_weights: Dict[str, float], targets: List[str]):
        super().__init__()
        weights = [task_weights.get(t, 1.0) for t in targets]
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float))

    def forward(self, pred, target, mask):
        squared_error = (pred - target) ** 2
        weighted_error = squared_error * self.weights.unsqueeze(0)
        masked_error = weighted_error * mask
        n_valid = mask.sum()
        if n_valid > 0:
            return masked_error.sum() / n_valid
        return torch.tensor(0.0, device=pred.device)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_single_model(model, train_loader, val_loader, device, config, criterion):
    """Train a single model."""
    total_steps = len(train_loader) * config.MAX_EPOCHS
    warmup_steps = len(train_loader) * config.WARMUP_EPOCHS

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(config.MAX_EPOCHS):
        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch)
            batch_size = pred.shape[0]
            targets = batch.targets.view(batch_size, config.NUM_TASKS)
            masks = batch.masks.view(batch_size, config.NUM_TASKS)

            loss = criterion(pred, targets, masks)
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

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
                if not torch.isnan(loss):
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
    return model


def evaluate_ensemble(models, test_loader, device, scalers, config):
    """Evaluate ensemble and return metrics."""
    all_preds = []

    for model in models:
        model.eval()
        model_preds = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch)
                model_preds.append(pred.cpu().numpy())

        all_preds.append(np.vstack(model_preds))

    # Ensemble prediction
    ensemble_preds = np.mean(all_preds, axis=0)

    # Get targets
    test_targets = []
    test_masks = []
    for batch in test_loader:
        batch_size = batch.targets.shape[0] // config.NUM_TASKS
        test_targets.append(batch.targets.view(batch_size, config.NUM_TASKS).numpy())
        test_masks.append(batch.masks.view(batch_size, config.NUM_TASKS).numpy())

    test_targets = np.vstack(test_targets)
    test_masks = np.vstack(test_masks)

    # Calculate metrics
    metrics = {}
    for i, task in enumerate(config.TARGETS):
        mask = test_masks[:, i] > 0.5
        if mask.sum() < 2:
            continue

        y_true = scalers[task].inverse_transform(test_targets[mask, i].reshape(-1, 1)).flatten()
        y_pred = scalers[task].inverse_transform(ensemble_preds[mask, i].reshape(-1, 1)).flatten()

        # Apply unit conversion
        conversion = config.UNIT_CONVERSIONS.get(task, 1.0)
        y_true *= conversion
        y_pred *= conversion

        metrics[task] = {
            'r2': float(r2_score(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'n': int(mask.sum()),
        }

    return metrics


def run_ablation_variant(name, train_df, test_df, scalers, config, device,
                         use_temperature=True, use_descriptors=True,
                         task_weights=None, n_ensemble=20):
    """Run a single ablation variant."""

    print(f"\n{'=' * 60}")
    print(f"ABLATION: {name}")
    print(f"{'=' * 60}")
    print(f"  Temperature: {use_temperature}")
    print(f"  Descriptors: {use_descriptors}")
    print(f"  Task weights: {'Custom' if task_weights != config.EQUAL_WEIGHTS else 'Equal'}")
    print(f"  Ensemble size: {n_ensemble}")

    # Prepare datasets
    train_dataset = SurfactantDataset(train_df, scalers, config.TARGETS, include_global=use_descriptors)
    test_dataset = SurfactantDataset(test_df, scalers, config.TARGETS, include_global=use_descriptors)

    train_graphs = train_dataset.graphs
    test_graphs = test_dataset.graphs
    test_loader = DataLoader(test_graphs, batch_size=config.BATCH_SIZE, shuffle=False)

    # Loss function
    weights = task_weights if task_weights else config.TASK_WEIGHTS
    criterion = WeightedMaskedMSELoss(weights, config.TARGETS).to(device)

    # Train ensemble
    models = []
    kfold = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    models_per_fold = max(1, n_ensemble // config.N_FOLDS)

    model_count = 0
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_graphs)):
        if model_count >= n_ensemble:
            break

        train_subset = [train_graphs[i] for i in train_idx]
        val_subset = [train_graphs[i] for i in val_idx]

        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)

        for model_idx in range(models_per_fold):
            if model_count >= n_ensemble:
                break

            model_count += 1
            seed = fold * models_per_fold + model_idx
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = FullModel(config, use_temperature=use_temperature, use_descriptors=use_descriptors).to(device)
            model = train_single_model(model, train_loader, val_loader, device, config, criterion)
            models.append(model)

            print(f"  Model {model_count}/{n_ensemble} trained")

    # Evaluate
    metrics = evaluate_ensemble(models, test_loader, device, scalers, config)

    # Calculate mean R²
    r2_values = [m['r2'] for m in metrics.values()]
    mean_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)

    print(f"\n  Results for {name}:")
    print(f"  {'Property':<12} {'R²':<8} {'RMSE':<10}")
    print(f"  {'-' * 32}")
    for task in config.TARGETS:
        if task in metrics:
            m = metrics[task]
            print(f"  {task:<12} {m['r2']:.4f}   {m['rmse']:.4f}")
    print(f"  {'-' * 32}")
    print(f"  {'Mean R²':<12} {mean_r2:.4f} ± {std_r2:.4f}")

    return {
        'name': name,
        'metrics': metrics,
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'config': {
            'use_temperature': use_temperature,
            'use_descriptors': use_descriptors,
            'task_weights': 'custom' if task_weights != config.EQUAL_WEIGHTS else 'equal',
            'n_ensemble': n_ensemble,
        }
    }


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(data_dir: str, output_dir: str = None, device: str = 'cuda'):
    """Run complete ablation study."""

    # Setup
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"experiments/ablation_study_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ABLATION STUDY FOR SurfMT-GNN")
    print("=" * 70)
    print(f"Output directory: {output_path}")

    config = Config()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print(f"\n📂 Loading data from {data_dir}...")
    train_df = pd.read_csv(f"{data_dir}/surfpro_train.csv")
    test_df = pd.read_csv(f"{data_dir}/surfpro_test.csv")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Setup scalers
    scalers = {}
    for task in config.TARGETS:
        if task in train_df.columns:
            scalers[task] = RobustScaler()
            valid_mask = train_df[task].notna()
            if valid_mask.sum() > 0:
                scalers[task].fit(train_df.loc[valid_mask, task].values.reshape(-1, 1))

    # Run ablation variants
    results = []

    # 1. Full Model (with larger ensemble for baseline)
    print("\n" + "=" * 70)
    print("Running ablation variants...")
    print("=" * 70)

    results.append(run_ablation_variant(
        "Full Model (Baseline)",
        train_df, test_df, scalers, config, device,
        use_temperature=True, use_descriptors=True,
        task_weights=config.TASK_WEIGHTS, n_ensemble=20
    ))

    # 2. No Temperature
    results.append(run_ablation_variant(
        "No Temperature",
        train_df, test_df, scalers, config, device,
        use_temperature=False, use_descriptors=True,
        task_weights=config.TASK_WEIGHTS, n_ensemble=20
    ))

    # 3. No Descriptors
    results.append(run_ablation_variant(
        "No Descriptors",
        train_df, test_df, scalers, config, device,
        use_temperature=True, use_descriptors=False,
        task_weights=config.TASK_WEIGHTS, n_ensemble=20
    ))

    # 4. Equal Weights
    results.append(run_ablation_variant(
        "Equal Weights",
        train_df, test_df, scalers, config, device,
        use_temperature=True, use_descriptors=True,
        task_weights=config.EQUAL_WEIGHTS, n_ensemble=20
    ))

    # 5. Single Model (no ensemble)
    results.append(run_ablation_variant(
        "Single Model",
        train_df, test_df, scalers, config, device,
        use_temperature=True, use_descriptors=True,
        task_weights=config.TASK_WEIGHTS, n_ensemble=1
    ))

    # Generate comparison table
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 70)

    baseline_r2 = results[0]['mean_r2']

    print(f"\n{'Variant':<25} {'Mean R²':<12} {'Change':<12} {'% Change':<12}")
    print("-" * 60)

    for r in results:
        change = r['mean_r2'] - baseline_r2
        pct_change = (change / baseline_r2) * 100 if baseline_r2 > 0 else 0
        sign = '+' if change >= 0 else ''
        print(f"{r['name']:<25} {r['mean_r2']:.4f}       {sign}{change:.4f}      {sign}{pct_change:.1f}%")

    print("-" * 60)

    # Per-property comparison
    print(f"\n{'Per-Property R² Comparison':^60}")
    print("-" * 60)

    header = f"{'Property':<12}"
    for r in results:
        short_name = r['name'][:10]
        header += f" {short_name:<10}"
    print(header)
    print("-" * 60)

    for task in config.TARGETS:
        row = f"{task:<12}"
        for r in results:
            if task in r['metrics']:
                row += f" {r['metrics'][task]['r2']:.4f}    "
            else:
                row += f" {'N/A':<10}"
        print(row)

    # Save results
    with open(output_path / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate LaTeX table
    print("\n" + "=" * 70)
    print("LATEX TABLE FOR PAPER (Table 4)")
    print("=" * 70)

    print(r"""
\begin{table}[t]
\caption{\textbf{Ablation study results.} Impact of removing key components.}
\label{tab:ablation}
\centering
\small
\begin{tabular}{lcc}
\toprule
\textbf{Variant} & \textbf{Mean $R^2$} & \textbf{vs. Full} \\
\midrule""")

    for r in results:
        change = r['mean_r2'] - baseline_r2
        pct = (change / baseline_r2) * 100
        sign = '+' if change >= 0 else ''

        if r['name'] == "Full Model (Baseline)":
            print(f"{r['name'].replace('(Baseline)', '(Ours)')} & \\textbf{{{r['mean_r2']:.3f}}} & --- \\\\")
        else:
            print(f"{r['name']} & {r['mean_r2']:.3f} & {sign}{pct:.1f}\\% \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

    print(f"\n✅ Ablation results saved to: {output_path}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run ablation study for SurfMT-GNN')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.device)