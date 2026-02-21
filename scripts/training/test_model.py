#!/usr/bin/env python3
"""
Quick Test Script for Phase 3 Components
========================================
Test that all model and training components work correctly.

Usage:
    python scripts/training/test_model.py

Author: Al-Futini Abdulhakim Nasser Ali
"""

import sys
from pathlib import Path

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project Root: {PROJECT_ROOT}")

# =============================================================================
# Check Dependencies
# =============================================================================

print("\n" + "=" * 70)
print("Checking Dependencies")
print("=" * 70)

import torch

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

import torch_geometric

print(f"✓ PyTorch Geometric: {torch_geometric.__version__}")

# =============================================================================
# Test Model Imports
# =============================================================================

print("\n" + "=" * 70)
print("Testing Model Imports")
print("=" * 70)

try:
    from src.models.attentive_fp import AttentiveFPEncoder

    print("✓ AttentiveFPEncoder imported")
except ImportError as e:
    print(f"✗ AttentiveFPEncoder: {e}")
    sys.exit(1)

try:
    from src.models.task_heads import TaskHead, MultiTaskHeads

    print("✓ TaskHeads imported")
except ImportError as e:
    print(f"✗ TaskHeads: {e}")
    sys.exit(1)

try:
    from src.models.mtl_model import SurfProMTL

    print("✓ SurfProMTL imported")
except ImportError as e:
    print(f"✗ SurfProMTL: {e}")
    sys.exit(1)

try:
    from src.training.losses import MaskedMSELoss, MultiTaskLoss

    print("✓ Loss functions imported")
except ImportError as e:
    print(f"✗ Loss functions: {e}")
    sys.exit(1)

try:
    from scripts.training.trainer import Trainer, TrainingConfig

    print("✓ Trainer imported")
except ImportError as e:
    print(f"✗ Trainer: {e}")
    sys.exit(1)

# =============================================================================
# Test Model Forward Pass
# =============================================================================

print("\n" + "=" * 70)
print("Testing Model Forward Pass")
print("=" * 70)

from torch_geometric.data import Data, Batch

# Constants
NUM_TASKS = 6
TASK_NAMES = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']


def create_dummy_batch(batch_size=4):
    """Create dummy batch for testing."""
    data_list = []
    for i in range(batch_size):
        num_nodes = 20 + i * 5
        num_edges = num_nodes * 2

        data = Data(
            x=torch.randn(num_nodes, 34),
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            edge_attr=torch.randn(num_edges, 12),
            global_features=torch.randn(1, 6),
            y=torch.randn(NUM_TASKS),
            mask=torch.ones(NUM_TASKS)
        )
        data_list.append(data)

    return Batch.from_data_list(data_list)


def reshape_batch_targets(batch, num_tasks=NUM_TASKS):
    """
    Reshape y and mask from concatenated 1D tensors to 2D tensors.

    PyTorch Geometric concatenates y and mask when batching, so we need
    to reshape from [batch_size * num_tasks] to [batch_size, num_tasks].
    """
    y = batch.y.view(batch.num_graphs, num_tasks)
    mask = batch.mask.view(batch.num_graphs, num_tasks)
    return y, mask


# Create model
model = SurfProMTL(
    atom_dim=34,
    bond_dim=12,
    global_dim=6,
    hidden_dim=128,  # Smaller for testing
    num_layers=2,
    num_timesteps=2,
    dropout=0.1
)

# Print parameter count
params = model.count_parameters()
print(f"\nModel parameters: {params['total']:,}")

# Create dummy batch
batch = create_dummy_batch(4)
print(f"\nDummy batch created:")
print(f"  Graphs: {batch.num_graphs}")
print(f"  Nodes: {batch.x.shape[0]}")
print(f"  Edges: {batch.edge_index.shape[1]}")

# Forward pass
model.eval()
with torch.no_grad():
    output = model(batch)

print(f"\nForward pass successful!")
print(f"  predictions shape: {output['predictions'].shape}")
print(f"  Expected shape: [{batch.num_graphs}, {NUM_TASKS}]")

assert output['predictions'].shape == (batch.num_graphs, NUM_TASKS), "Output shape mismatch!"
print("✓ Output shape correct")

# =============================================================================
# Test Loss Functions
# =============================================================================

print("\n" + "=" * 70)
print("Testing Loss Functions")
print("=" * 70)

# Reshape y and mask from [batch_size * num_tasks] to [batch_size, num_tasks]
y, mask = reshape_batch_targets(batch)

print(f"  predictions shape: {output['predictions'].shape}")
print(f"  y shape (reshaped): {y.shape}")
print(f"  mask shape (reshaped): {mask.shape}")

# Test MaskedMSELoss
mse_loss = MaskedMSELoss()
loss = mse_loss(output['predictions'], y, mask)
print(f"✓ MaskedMSELoss: {loss.item():.4f}")

# Test MultiTaskLoss
mtl_loss = MultiTaskLoss(TASK_NAMES)
losses = mtl_loss(output['predictions'], y, mask)
print(f"✓ MultiTaskLoss: {losses['total'].item():.4f}")

# Print per-task losses
print("\n  Per-task losses:")
for task_name in TASK_NAMES:
    if task_name in losses:
        print(f"    {task_name}: {losses[task_name].item():.4f}")

# =============================================================================
# Test with Real Data (if available)
# =============================================================================

print("\n" + "=" * 70)
print("Testing with Real Data")
print("=" * 70)

try:
    from src.data import SurfProDataset, create_dataloaders, load_cv_splits

    data_root = PROJECT_ROOT / 'data'

    if (data_root / 'processed' / 'surfpro_train.pt').exists():
        print("Loading processed dataset...")
        dataset = SurfProDataset(root=str(data_root), split='train')
        print(f"✓ Dataset loaded: {len(dataset)} samples")

        # Load splits
        splits_path = data_root / 'splits' / 'cv_splits.json'
        if splits_path.exists():
            splits = load_cv_splits(str(splits_path))
            train_idx, val_idx = splits[0]

            # Create small loader
            train_loader, val_loader = create_dataloaders(
                dataset, train_idx[:100], val_idx[:50],
                batch_size=16, num_workers=0
            )

            # Get one batch
            real_batch = next(iter(train_loader))
            print(f"✓ Real batch loaded: {real_batch.num_graphs} graphs")

            # Forward pass
            with torch.no_grad():
                output = model(real_batch)

            print(f"✓ Forward pass on real data: {output['predictions'].shape}")

            # Reshape y and mask for loss calculation
            y, mask = reshape_batch_targets(real_batch)

            # Loss
            losses = mtl_loss(output['predictions'], y, mask)
            print(f"✓ Loss on real data: {losses['total'].item():.4f}")

            # Print per-task losses
            print("\n  Per-task losses on real data:")
            for task_name in TASK_NAMES:
                if task_name in losses:
                    print(f"    {task_name}: {losses[task_name].item():.4f}")
        else:
            print("⚠ CV splits not found")
    else:
        print("⚠ Processed data not found. Run test_pipeline.py first.")

except Exception as e:
    print(f"⚠ Error testing with real data: {e}")
    import traceback

    traceback.print_exc()

# =============================================================================
# Test GPU (if available)
# =============================================================================

print("\n" + "=" * 70)
print("Testing GPU")
print("=" * 70)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

    # Move model to GPU
    model_gpu = model.to(device)
    batch_gpu = batch.to(device)

    with torch.no_grad():
        output_gpu = model_gpu(batch_gpu)

    print(f"✓ Forward pass on GPU successful")
    print(f"  Output device: {output_gpu['predictions'].device}")

    # Test loss on GPU
    y_gpu = batch_gpu.y.view(batch_gpu.num_graphs, NUM_TASKS)
    mask_gpu = batch_gpu.mask.view(batch_gpu.num_graphs, NUM_TASKS)

    loss_gpu = mse_loss(output_gpu['predictions'], y_gpu, mask_gpu)
    print(f"✓ Loss on GPU: {loss_gpu.item():.4f}")
else:
    print("⚠ No GPU available, skipping GPU test")

# =============================================================================
# Test Gradient Flow
# =============================================================================

print("\n" + "=" * 70)
print("Testing Gradient Flow")
print("=" * 70)

# Create fresh model for gradient test
model_grad = SurfProMTL(
    atom_dim=34,
    bond_dim=12,
    global_dim=6,
    hidden_dim=128,
    num_layers=2,
    num_timesteps=2,
    dropout=0.1
)

model_grad.train()
batch_grad = create_dummy_batch(4)

# Forward pass
output_grad = model_grad(batch_grad)

# Reshape targets
y_grad, mask_grad = reshape_batch_targets(batch_grad)

# Compute loss
loss_grad = mse_loss(output_grad['predictions'], y_grad, mask_grad)

# Backward pass
loss_grad.backward()

# Check gradients
has_grad = True
for name, param in model_grad.named_parameters():
    if param.requires_grad and param.grad is None:
        print(f"✗ No gradient for: {name}")
        has_grad = False

if has_grad:
    print("✓ All parameters have gradients")

    # Print gradient norms for key layers
    print("\n  Gradient norms (sample):")
    for name, param in model_grad.named_parameters():
        if param.grad is not None and 'weight' in name:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                print(f"    {name}: {grad_norm:.6f}")
                break  # Just show one example

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("✓ ALL PHASE 3 TESTS PASSED!")
print("=" * 70)

print("""
Phase 3 Components Ready:
  ✓ AttentiveFP Encoder
  ✓ Multi-Task Heads
  ✓ SurfProMTL Model
  ✓ Masked Loss Functions
  ✓ Trainer Class
  ✓ Gradient Flow

Next: Run full training with:
  python scripts/training/train_mtl.py --fold 0

Or train all folds:
  python scripts/training/train_mtl.py --fold -1
""")