#!/usr/bin/env python3
"""
SurfPro Data Pipeline - Complete Test Script
=============================================
Run this script to test the entire data preprocessing pipeline.

Prerequisites:
    pip install torch torch-geometric rdkit pandas numpy scikit-learn tqdm

Usage:
    python scripts/data_preparation/test_pipeline.py

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Target Journal: Journal of Chemical Information and Modeling (JCIM)
"""

import os
import sys
from pathlib import Path

# =============================================================================
# Setup Project Root
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project Root: {PROJECT_ROOT}")


# =============================================================================
# Check Dependencies
# =============================================================================

def check_dependencies():
    """Check if all required packages are installed."""
    print("\n" + "=" * 70)
    print("Checking Dependencies")
    print("=" * 70)

    all_ok = True

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found. Install: pip install torch")
        all_ok = False

    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError:
        print("✗ PyTorch Geometric not found. Install: pip install torch-geometric")
        all_ok = False

    try:
        from rdkit import Chem
        print("✓ RDKit: available")
    except ImportError:
        print("✗ RDKit not found. Install: pip install rdkit")
        all_ok = False

    try:
        import pandas
        import numpy
        import sklearn
        print("✓ Pandas, NumPy, Scikit-learn: available")
    except ImportError as e:
        print(f"✗ Missing: {e}")
        all_ok = False

    return all_ok


# =============================================================================
# Import Project Modules
# =============================================================================

def import_project_modules():
    """Import project modules and return them."""
    print("\n" + "=" * 70)
    print("Importing Project Modules")
    print("=" * 70)

    modules = {}

    try:
        from src.data.featurizer import (
            AtomFeaturizer,
            BondFeaturizer,
            MoleculeFeaturizer,
            FeaturizationConfig,
            validate_featurization
        )
        modules['AtomFeaturizer'] = AtomFeaturizer
        modules['BondFeaturizer'] = BondFeaturizer
        modules['MoleculeFeaturizer'] = MoleculeFeaturizer
        modules['FeaturizationConfig'] = FeaturizationConfig
        modules['validate_featurization'] = validate_featurization
        print("✓ Featurizer module")
    except ImportError as e:
        print(f"✗ Featurizer: {e}")
        return None

    try:
        from src.data.dataset import (
            SurfProDataset,
            get_cv_splits,
            create_dataloaders,
            create_test_dataloader,
            print_dataset_info,
            TARGET_COLUMNS
        )
        modules['SurfProDataset'] = SurfProDataset
        modules['get_cv_splits'] = get_cv_splits
        modules['create_dataloaders'] = create_dataloaders
        modules['create_test_dataloader'] = create_test_dataloader
        modules['print_dataset_info'] = print_dataset_info
        modules['TARGET_COLUMNS'] = TARGET_COLUMNS
        print("✓ Dataset module")
    except ImportError as e:
        print(f"✗ Dataset: {e}")
        return None

    try:
        from src.data.transforms import (
            TargetScaler,
            AddNoise,
            NormalizeFeatures
        )
        modules['TargetScaler'] = TargetScaler
        modules['AddNoise'] = AddNoise
        modules['NormalizeFeatures'] = NormalizeFeatures
        print("✓ Transforms module")
    except ImportError as e:
        print(f"✗ Transforms: {e}")
        return None

    return modules


# =============================================================================
# Test 1: Featurizer
# =============================================================================

def run_test_featurizer(modules):
    """Test molecular featurization."""
    print("\n" + "=" * 70)
    print("Test 1: Molecular Featurizer")
    print("=" * 70)

    AtomFeaturizer = modules['AtomFeaturizer']
    BondFeaturizer = modules['BondFeaturizer']
    MoleculeFeaturizer = modules['MoleculeFeaturizer']
    validate_featurization = modules['validate_featurization']

    atom_feat = AtomFeaturizer()
    bond_feat = BondFeaturizer()
    mol_feat = MoleculeFeaturizer()

    print(f"\nFeature Dimensions:")
    print(f"  Atom: {atom_feat.feature_dim}")
    print(f"  Bond: {bond_feat.feature_dim}")
    print(f"  Global: {mol_feat.global_dim}")

    smiles_examples = [
        ("SDS (anionic)", "CCCCCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"),
        ("DTAC (cationic)", "CCCCCCCCCCCC[N+](C)(C)C.[Cl-]"),
        ("Non-ionic", "CCCCCCCCCCCC(=O)OCC(O)CO"),
        ("Invalid", "INVALID_SMILES"),
    ]

    print(f"\nSMILES Featurization:")
    for name, smiles in smiles_examples:
        result = validate_featurization(smiles, verbose=False)
        if result['valid']:
            print(f"  ✓ {name}: {result['num_atoms']} atoms, {result['num_bonds']} bonds")
        else:
            print(f"  ✗ {name}: {result.get('error', 'Failed')}")

    print("\n✓ Featurizer test completed")
    return True


# =============================================================================
# Test 2: Dataset
# =============================================================================

def run_test_dataset(modules):
    """Test dataset creation."""
    print("\n" + "=" * 70)
    print("Test 2: SurfPro Dataset")
    print("=" * 70)

    SurfProDataset = modules['SurfProDataset']
    print_dataset_info = modules['print_dataset_info']

    data_root = PROJECT_ROOT / 'data'
    train_csv = data_root / 'raw' / 'surfpro_train.csv'

    if not train_csv.exists():
        print(f"\n⚠ Data not found: {train_csv}")
        print("  Please ensure CSV files are in data/raw/")
        return None, None

    print(f"\n✓ Raw data found at: {data_root / 'raw'}")

    # Create training dataset
    print("\nProcessing training data...")
    train_ds = SurfProDataset(root=str(data_root), split='train')

    # Create test dataset
    print("\nProcessing test data...")
    test_ds = SurfProDataset(root=str(data_root), split='test')

    # Print info
    print_dataset_info(train_ds)

    print("\n✓ Dataset test completed")
    return train_ds, test_ds


# =============================================================================
# Test 3: CV Splits
# =============================================================================

def run_test_cv_splits(modules, train_ds):
    """Test cross-validation split generation."""
    print("\n" + "=" * 70)
    print("Test 3: Cross-Validation Splits")
    print("=" * 70)

    get_cv_splits = modules['get_cv_splits']

    splits_dir = PROJECT_ROOT / 'data' / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_path = splits_dir / 'cv_splits.json'

    splits = get_cv_splits(
        train_ds,
        n_folds=10,
        seed=42,
        save_path=str(splits_path)
    )

    print(f"\n✓ CV splits saved to: {splits_path}")
    print("✓ CV splits test completed")
    return splits


# =============================================================================
# Test 4: DataLoaders
# =============================================================================

def run_test_dataloaders(modules, train_ds, test_ds, splits):
    """Test DataLoader creation."""
    print("\n" + "=" * 70)
    print("Test 4: DataLoaders")
    print("=" * 70)

    create_dataloaders = modules['create_dataloaders']
    create_test_dataloader = modules['create_test_dataloader']

    # Use first fold
    train_idx, val_idx = splits[0]

    print(f"\nUsing Fold 1: train={len(train_idx)}, val={len(val_idx)}")

    train_loader, val_loader = create_dataloaders(
        train_ds,
        train_idx,
        val_idx,
        batch_size=32,
        num_workers=0
    )

    test_loader = create_test_dataloader(
        test_ds,
        batch_size=32,
        num_workers=0
    )

    print(f"\nDataLoader sizes:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    # Inspect a batch
    batch = next(iter(train_loader))
    print(f"\nBatch inspection:")
    print(f"  num_graphs: {batch.num_graphs}")
    print(f"  x shape: {batch.x.shape}")
    print(f"  edge_index shape: {batch.edge_index.shape}")
    print(f"  y shape: {batch.y.shape}")
    print(f"  mask shape: {batch.mask.shape}")

    print("\n✓ DataLoader test completed")
    return train_loader, val_loader, test_loader


# =============================================================================
# Test 5: Target Scaling
# =============================================================================

def run_test_scaling(modules, train_ds, train_idx):
    """Test target scaling."""
    print("\n" + "=" * 70)
    print("Test 5: Target Scaling")
    print("=" * 70)

    TargetScaler = modules['TargetScaler']
    TARGET_COLUMNS = modules['TARGET_COLUMNS']

    # Create scaler
    scaler = TargetScaler(task_names=TARGET_COLUMNS)

    # Fit on training data
    print("\nFitting scaler on training data...")
    scaler.fit(train_ds, indices=train_idx)

    # Save scaler
    scaler_dir = PROJECT_ROOT / 'data' / 'processed'
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = scaler_dir / 'target_scaler.json'
    scaler.save(str(scaler_path))

    # Test transform/inverse_transform
    sample = train_ds[train_idx[0]]
    original = sample.y.clone()

    print(f"\nScaling test:")
    print(f"  Original:  {original.numpy()}")

    scaled = scaler.transform(original)
    print(f"  Scaled:    {scaled.numpy()}")

    recovered = scaler.inverse_transform(scaled)
    print(f"  Recovered: {recovered.numpy()}")

    error = (original - recovered).abs().max().item()
    print(f"  Max error: {error:.10f}")

    print("\n✓ Scaling test completed")
    return scaler


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("SurfPro Data Pipeline - Complete Test Suite")
    print("=" * 70)

    # Step 0: Check dependencies
    if not check_dependencies():
        print("\n✗ Missing dependencies. Please install required packages.")
        sys.exit(1)

    # Step 1: Import modules
    modules = import_project_modules()
    if modules is None:
        print("\n✗ Failed to import project modules.")
        sys.exit(1)

    # Test 1: Featurizer
    run_test_featurizer(modules)

    # Test 2: Dataset
    train_ds, test_ds = run_test_dataset(modules)
    if train_ds is None:
        print("\n⚠ Stopping tests - data not found")
        sys.exit(1)

    # Test 3: CV Splits
    splits = run_test_cv_splits(modules, train_ds)

    # Test 4: DataLoaders
    run_test_dataloaders(modules, train_ds, test_ds, splits)

    # Test 5: Scaling
    train_idx, _ = splits[0]
    run_test_scaling(modules, train_ds, train_idx)

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)

    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    FILES CREATED                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  • data/processed/surfpro_train.pt                               ║
    ║  • data/processed/surfpro_test.pt                                ║
    ║  • data/splits/cv_splits.json                                    ║
    ║  • data/processed/target_scaler.json                             ║
    ╚══════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════╗
    ║                    DATASET SUMMARY                                ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Training samples:  {len(train_ds):>6}                                       ║
    ║  Test samples:      {len(test_ds):>6}                                       ║
    ║  Atom features:     {train_ds.atom_dim:>6}                                       ║
    ║  Bond features:     {train_ds.bond_dim:>6}                                       ║
    ║  Global features:   {train_ds.global_dim:>6}                                       ║
    ║  Number of tasks:   {train_ds.num_tasks:>6}                                       ║
    ╚══════════════════════════════════════════════════════════════════╝

    🚀 Ready for Phase 3: Model Training!
    """)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    main()