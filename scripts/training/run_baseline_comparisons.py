#!/usr/bin/env python3
"""
=============================================================================
BASELINE COMPARISONS FOR SurfMT-GNN
=============================================================================
This script runs all baseline comparisons required for JCIM paper:

1. Traditional ML Methods:
   - Random Forest + 200 RDKit descriptors
   - XGBoost + 200 RDKit descriptors
   - SVR (RBF kernel) + 200 RDKit descriptors

2. Single-Task GNN Methods:
   - Single-task AttentiveFP (trained per property)

Output:
- Baseline results table (plain text + LaTeX)
- Comparison with SurfMT-GNN
- JSON results for reproducibility

Author: Al-Futini Abdulhakim Nasser Ali
For: JCIM Paper - SurfMT-GNN

Run:
    python run_baseline_comparisons.py --data_dir data/raw --device cuda
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
from torch.optim import AdamW

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP

# Scikit-learn
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# XGBoost
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Will skip XGBoost baseline.")

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for baselines."""

    TARGETS = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']
    NUM_TASKS = 6

    # GNN settings
    HIDDEN_DIM = 128
    OUTPUT_DIM = 256
    NUM_LAYERS = 3
    DROPOUT = 0.15

    # Training
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 32
    MAX_EPOCHS = 300
    PATIENCE = 50

    # CV
    N_FOLDS = 5  # Fewer folds for baselines (faster)

    # Unit conversions
    UNIT_CONVERSIONS = {
        'pCMC': 1.0,
        'AW_ST_CMC': 1.0,
        'Gamma_max': 1e6,
        'Area_min': 1.0,
        'Pi_CMC': 1.0,
        'pC20': 1.0,
    }


# =============================================================================
# RDKit DESCRIPTOR CALCULATION
# =============================================================================

def get_all_descriptor_names():
    """Get list of all RDKit descriptor names."""
    return [x[0] for x in Descriptors._descList]


def compute_rdkit_descriptors(smiles: str) -> Optional[np.ndarray]:
    """Compute all RDKit descriptors for a molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Get all descriptor names
        desc_names = get_all_descriptor_names()
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

        # Calculate descriptors
        descriptors = calculator.CalcDescriptors(mol)
        descriptors = np.array(descriptors, dtype=np.float32)

        # Replace NaN/Inf with 0
        descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)

        return descriptors
    except:
        return None


def prepare_descriptor_data(df: pd.DataFrame, target: str) -> tuple:
    """Prepare X, y data for traditional ML."""
    X_list = []
    y_list = []

    for idx, row in df.iterrows():
        if pd.isna(row.get(target)):
            continue

        desc = compute_rdkit_descriptors(row['SMILES'])
        if desc is None:
            continue

        # Add temperature as feature
        temp = row.get('temp', 25.0)
        if pd.isna(temp):
            temp = 25.0

        features = np.append(desc, temp)
        X_list.append(features)
        y_list.append(row[target])

    return np.array(X_list), np.array(y_list)


# =============================================================================
# TRADITIONAL ML BASELINES
# =============================================================================

def run_random_forest(X_train, y_train, X_test, y_test):
    """Run Random Forest baseline."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    return y_pred


def run_xgboost(X_train, y_train, X_test, y_test):
    """Run XGBoost baseline."""
    if not HAS_XGBOOST:
        return None

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    return y_pred


def run_svr(X_train, y_train, X_test, y_test):
    """Run SVR baseline."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Scale target
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    # Train model (use subset for SVR due to O(n²) complexity)
    if len(X_train_scaled) > 1000:
        indices = np.random.choice(len(X_train_scaled), 1000, replace=False)
        X_train_subset = X_train_scaled[indices]
        y_train_subset = y_train_scaled[indices]
    else:
        X_train_subset = X_train_scaled
        y_train_subset = y_train_scaled

    model = SVR(kernel='rbf', C=10, gamma='scale')
    model.fit(X_train_subset, y_train_subset)

    # Predict
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    return y_pred


def run_traditional_ml_baselines(train_df, test_df, config):
    """Run all traditional ML baselines."""

    print("\n" + "=" * 70)
    print("TRADITIONAL ML BASELINES")
    print("=" * 70)

    results = {
        'Random Forest': {},
        'XGBoost': {},
        'SVR': {},
    }

    for target in config.TARGETS:
        print(f"\n📊 Processing {target}...")

        # Prepare data
        X_train, y_train = prepare_descriptor_data(train_df, target)
        X_test, y_test = prepare_descriptor_data(test_df, target)

        if len(X_train) < 10 or len(X_test) < 5:
            print(f"  Skipping {target}: insufficient data")
            continue

        print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")

        # Apply unit conversion
        conversion = config.UNIT_CONVERSIONS.get(target, 1.0)
        y_train_conv = y_train * conversion
        y_test_conv = y_test * conversion

        # Random Forest
        print(f"  Running Random Forest...")
        y_pred_rf = run_random_forest(X_train, y_train, X_test, y_test)
        y_pred_rf_conv = y_pred_rf * conversion

        r2_rf = r2_score(y_test_conv, y_pred_rf_conv)
        rmse_rf = np.sqrt(mean_squared_error(y_test_conv, y_pred_rf_conv))

        results['Random Forest'][target] = {
            'r2': float(r2_rf),
            'rmse': float(rmse_rf),
            'n': len(y_test)
        }
        print(f"    RF: R² = {r2_rf:.4f}, RMSE = {rmse_rf:.4f}")

        # XGBoost
        if HAS_XGBOOST:
            print(f"  Running XGBoost...")
            y_pred_xgb = run_xgboost(X_train, y_train, X_test, y_test)
            y_pred_xgb_conv = y_pred_xgb * conversion

            r2_xgb = r2_score(y_test_conv, y_pred_xgb_conv)
            rmse_xgb = np.sqrt(mean_squared_error(y_test_conv, y_pred_xgb_conv))

            results['XGBoost'][target] = {
                'r2': float(r2_xgb),
                'rmse': float(rmse_xgb),
                'n': len(y_test)
            }
            print(f"    XGB: R² = {r2_xgb:.4f}, RMSE = {rmse_xgb:.4f}")

        # SVR
        print(f"  Running SVR...")
        y_pred_svr = run_svr(X_train, y_train, X_test, y_test)
        y_pred_svr_conv = y_pred_svr * conversion

        r2_svr = r2_score(y_test_conv, y_pred_svr_conv)
        rmse_svr = np.sqrt(mean_squared_error(y_test_conv, y_pred_svr_conv))

        results['SVR'][target] = {
            'r2': float(r2_svr),
            'rmse': float(rmse_svr),
            'n': len(y_test)
        }
        print(f"    SVR: R² = {r2_svr:.4f}, RMSE = {rmse_svr:.4f}")

    return results


# =============================================================================
# SINGLE-TASK GNN BASELINE
# =============================================================================

def get_atom_features(atom):
    """Compute atom features."""
    elements = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si']
    element_features = [1.0 if atom.GetSymbol() == e else 0.0 for e in elements]

    degree = [0.0] * 7
    degree[min(atom.GetTotalDegree(), 6)] = 1.0

    charge = [0.0] * 5
    charge[min(max(atom.GetFormalCharge() + 2, 0), 4)] = 1.0

    hyb_types = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
    hybridization = [1.0 if atom.GetHybridization() == h else 0.0 for h in hyb_types]

    is_aromatic = [1.0 if atom.GetIsAromatic() else 0.0]

    n_hs = [0.0] * 5
    n_hs[min(atom.GetTotalNumHs(), 4)] = 1.0

    chiral_types = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER]
    chirality = [1.0 if atom.GetChiralTag() == c else 0.0 for c in chiral_types]

    is_in_ring = [1.0 if atom.IsInRing() else 0.0]
    atomic_mass = [atom.GetMass() / 100.0]

    return element_features + degree + charge + hybridization + is_aromatic + n_hs + chirality + is_in_ring + atomic_mass


def get_bond_features(bond):
    """Compute bond features."""
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bt = [1.0 if bond.GetBondType() == b else 0.0 for b in bond_types]

    is_conjugated = [1.0 if bond.GetIsConjugated() else 0.0]
    is_in_ring = [1.0 if bond.IsInRing() else 0.0]

    stereo_types = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOCIS]
    stereo = [1.0 if bond.GetStereo() == s else 0.0 for s in stereo_types]

    return bt + is_conjugated + is_in_ring + stereo


def smiles_to_graph_single_task(smiles, target_value, temperature=25.0):
    """Convert SMILES to graph for single-task."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        if len(atom_features) == 0:
            return None

        x = torch.tensor(atom_features, dtype=torch.float)

        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = get_bond_features(bond)
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([bond_feat, bond_feat])

        if len(edge_index) == 0:
            edge_index = [[0, 0]]
            edge_attr = [[0.0] * 10]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.y = torch.tensor([target_value], dtype=torch.float)
        data.temperature = torch.tensor([temperature], dtype=torch.float)

        return data
    except:
        return None


class SingleTaskGNN(nn.Module):
    """Single-task AttentiveFP model."""

    def __init__(self, config):
        super().__init__()

        self.encoder = AttentiveFP(
            in_channels=39,
            hidden_channels=config.HIDDEN_DIM,
            out_channels=config.OUTPUT_DIM,
            edge_dim=10,
            num_layers=config.NUM_LAYERS,
            num_timesteps=2,
            dropout=config.DROPOUT,
        )

        self.head = nn.Sequential(
            nn.Linear(config.OUTPUT_DIM, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, batch):
        embed = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return self.head(embed)


def train_single_task_gnn(train_df, test_df, target, config, device, n_models=5):
    """Train single-task GNN for one property."""

    # Prepare data
    scaler = RobustScaler()

    # Training data
    train_graphs = []
    train_values = []
    for idx, row in train_df.iterrows():
        if pd.isna(row.get(target)):
            continue
        temp = row.get('temp', 25.0)
        if pd.isna(temp):
            temp = 25.0
        graph = smiles_to_graph_single_task(row['SMILES'], row[target], temp)
        if graph is not None:
            train_graphs.append(graph)
            train_values.append(row[target])

    if len(train_graphs) < 10:
        return None

    # Fit scaler and update targets
    train_values = np.array(train_values).reshape(-1, 1)
    scaler.fit(train_values)

    for i, g in enumerate(train_graphs):
        g.y = torch.tensor(scaler.transform([[train_values[i, 0]]])[0], dtype=torch.float)

    # Test data
    test_graphs = []
    test_values = []
    for idx, row in test_df.iterrows():
        if pd.isna(row.get(target)):
            continue
        temp = row.get('temp', 25.0)
        if pd.isna(temp):
            temp = 25.0
        graph = smiles_to_graph_single_task(row['SMILES'], row[target], temp)
        if graph is not None:
            test_graphs.append(graph)
            test_values.append(row[target])

    if len(test_graphs) < 5:
        return None

    test_values = np.array(test_values)

    # Train ensemble
    all_preds = []
    kfold = KFold(n_splits=min(config.N_FOLDS, len(train_graphs) // 10), shuffle=True, random_state=42)

    model_count = 0
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_graphs)):
        if model_count >= n_models:
            break

        train_subset = [train_graphs[i] for i in train_idx]
        val_subset = [train_graphs[i] for i in val_idx]

        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=config.BATCH_SIZE, shuffle=False)

        # Train model
        torch.manual_seed(fold)
        model = SingleTaskGNN(config).to(device)
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience = 0

        for epoch in range(config.MAX_EPOCHS):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch).squeeze()
                loss = criterion(pred, batch.y.squeeze())
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch).squeeze()
                    val_loss += criterion(pred, batch.y.squeeze()).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= config.PATIENCE:
                    break

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Predict
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch).cpu().numpy()
                preds.append(pred)

        preds = np.vstack(preds).flatten()
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        all_preds.append(preds)

        model_count += 1

    # Ensemble prediction
    ensemble_preds = np.mean(all_preds, axis=0)

    # Apply unit conversion
    conversion = config.UNIT_CONVERSIONS.get(target, 1.0)
    test_values_conv = test_values * conversion
    ensemble_preds_conv = ensemble_preds * conversion

    r2 = r2_score(test_values_conv, ensemble_preds_conv)
    rmse = np.sqrt(mean_squared_error(test_values_conv, ensemble_preds_conv))

    return {
        'r2': float(r2),
        'rmse': float(rmse),
        'n': len(test_values)
    }


def run_single_task_gnn_baselines(train_df, test_df, config, device):
    """Run single-task GNN baselines."""

    print("\n" + "=" * 70)
    print("SINGLE-TASK GNN BASELINES (AttentiveFP)")
    print("=" * 70)

    results = {}

    for target in config.TARGETS:
        print(f"\n📊 Training single-task model for {target}...")

        metrics = train_single_task_gnn(train_df, test_df, target, config, device, n_models=5)

        if metrics is not None:
            results[target] = metrics
            print(f"  R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
        else:
            print(f"  Skipped: insufficient data")

    return results


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(data_dir: str, output_dir: str = None, device: str = 'cuda'):
    """Run all baseline comparisons."""

    # Setup
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"experiments/baselines_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BASELINE COMPARISONS FOR SurfMT-GNN")
    print("=" * 70)

    config = Config()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print(f"\n📂 Loading data from {data_dir}...")
    train_df = pd.read_csv(f"{data_dir}/surfpro_train.csv")
    test_df = pd.read_csv(f"{data_dir}/surfpro_test.csv")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Run traditional ML baselines
    ml_results = run_traditional_ml_baselines(train_df, test_df, config)

    # Run single-task GNN baselines
    gnn_results = run_single_task_gnn_baselines(train_df, test_df, config, device)

    # Compile all results
    all_results = {
        'Traditional ML': ml_results,
        'Single-Task GNN': {'AttentiveFP': gnn_results}
    }

    # Calculate mean R² for each method
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 70)

    summary = {}

    # Traditional ML
    for method in ['Random Forest', 'XGBoost', 'SVR']:
        if method in ml_results and ml_results[method]:
            r2_values = [ml_results[method][t]['r2'] for t in config.TARGETS if t in ml_results[method]]
            if r2_values:
                summary[method] = {
                    'mean_r2': np.mean(r2_values),
                    'std_r2': np.std(r2_values)
                }

    # Single-task GNN
    if gnn_results:
        r2_values = [gnn_results[t]['r2'] for t in config.TARGETS if t in gnn_results]
        if r2_values:
            summary['Single-Task AttentiveFP'] = {
                'mean_r2': np.mean(r2_values),
                'std_r2': np.std(r2_values)
            }

    # Your SurfMT-GNN results (from enhanced training)
    surfmt_results = {
        'pCMC': 0.902, 'AW_ST_CMC': 0.809, 'Gamma_max': 0.805,
        'Area_min': 0.868, 'Pi_CMC': 0.808, 'pC20': 0.898
    }
    summary['SurfMT-GNN (Ours)'] = {
        'mean_r2': np.mean(list(surfmt_results.values())),
        'std_r2': np.std(list(surfmt_results.values()))
    }

    # Print summary
    print(f"\n{'Method':<30} {'Mean R²':<12} {'vs. Ours':<12}")
    print("-" * 55)

    our_r2 = summary['SurfMT-GNN (Ours)']['mean_r2']

    for method, stats in summary.items():
        diff = ((stats['mean_r2'] - our_r2) / our_r2) * 100
        sign = '+' if diff >= 0 else ''
        print(f"{method:<30} {stats['mean_r2']:.4f}       {sign}{diff:.1f}%")

    # Per-property comparison
    print(f"\n{'Per-Property R² Comparison':^70}")
    print("-" * 70)

    print(f"{'Property':<12} {'RF':<8} {'XGB':<8} {'SVR':<8} {'ST-GNN':<8} {'Ours':<8}")
    print("-" * 70)

    for target in config.TARGETS:
        row = f"{target:<12}"

        # RF
        if 'Random Forest' in ml_results and target in ml_results['Random Forest']:
            row += f" {ml_results['Random Forest'][target]['r2']:.4f}  "
        else:
            row += f" {'N/A':<8}"

        # XGB
        if 'XGBoost' in ml_results and target in ml_results['XGBoost']:
            row += f" {ml_results['XGBoost'][target]['r2']:.4f}  "
        else:
            row += f" {'N/A':<8}"

        # SVR
        if 'SVR' in ml_results and target in ml_results['SVR']:
            row += f" {ml_results['SVR'][target]['r2']:.4f}  "
        else:
            row += f" {'N/A':<8}"

        # ST-GNN
        if target in gnn_results:
            row += f" {gnn_results[target]['r2']:.4f}  "
        else:
            row += f" {'N/A':<8}"

        # Ours
        row += f" {surfmt_results.get(target, 'N/A'):.4f}"

        print(row)

    # Save results
    with open(output_path / 'baseline_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate LaTeX table
    print("\n" + "=" * 70)
    print("LATEX TABLE FOR PAPER (Table 4 - Baselines)")
    print("=" * 70)

    print(r"""
\begin{table}[t]
\caption{\textbf{Comprehensive baseline comparison (mean $R^2$ across 6 properties).}}
\label{tab:baselines}
\centering
\small
\begin{tabular}{lcc}
\toprule
\textbf{Method} & \textbf{Mean $R^2$} & \textbf{vs. Ours} \\
\midrule
\multicolumn{3}{l}{\textit{Traditional ML (200 descriptors)}} \\""")

    for method in ['Random Forest', 'XGBoost', 'SVR']:
        if method in summary:
            diff = ((summary[method]['mean_r2'] - our_r2) / our_r2) * 100
            print(f"\\quad {method} & {summary[method]['mean_r2']:.3f} & {diff:.1f}\\% \\\\")

    print(r"""\midrule
\multicolumn{3}{l}{\textit{Single-task GNNs}} \\""")

    if 'Single-Task AttentiveFP' in summary:
        diff = ((summary['Single-Task AttentiveFP']['mean_r2'] - our_r2) / our_r2) * 100
        print(f"\\quad AttentiveFP & {summary['Single-Task AttentiveFP']['mean_r2']:.3f} & {diff:.1f}\\% \\\\")

    print(r"""\midrule
\textbf{SurfMT-GNN (Ours)} & \textbf{""" + f"{our_r2:.3f}" + r"""} & --- \\
\bottomrule
\end{tabular}
\end{table}
""")

    print(f"\n✅ Baseline results saved to: {output_path}")

    return all_results, summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run baseline comparisons')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.device)