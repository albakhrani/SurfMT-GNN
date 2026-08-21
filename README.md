# SurfMT-GNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Multi-Task Graph Neural Networks for Comprehensive Surfactant Property Prediction**

> Accompanying code for the paper submitted to *Journal of Chemical Information and Modeling* (JCIM)

---

## 📋 Overview

SurfMT-GNN is a unified multi-task graph neural network framework for simultaneous prediction of six key surfactant properties:

| Property | Symbol | Unit | Description |
|----------|--------|------|-------------|
| Critical Micelle Concentration | pCMC | -log₁₀(M) | Concentration threshold for micelle formation |
| Surface Tension at CMC | γ_CMC | mN/m | Interfacial tension reduction at saturation |
| Maximum Surface Excess | Γ_max | μmol/m² | Interfacial packing density |
| Minimum Molecular Area | A_min | nm² | Molecular footprint at interface |
| Surface Pressure at CMC | π_CMC | mN/m | Surface pressure at saturation |
| Surfactant Efficiency | pC₂₀ | -log₁₀(M) | Concentration for 20 mN/m reduction |

## 🎯 Key Features

- **Multi-Task Learning**: Simultaneous prediction of 6 properties with knowledge transfer
- **Temperature-Aware**: Dedicated temperature encoder for thermal effects (15-60°C)
- **Hybrid Architecture**: Combines AttentiveFP graph encoder + molecular descriptors
- **Uncertainty Quantification**: 60-model deep ensemble with calibrated confidence intervals
- **High Throughput**: ~50ms per molecule inference (>1,000 molecules/min)

## 📊 Performance

| Property | R² | RMSE | Improvement vs Single-Task |
|----------|-----|------|---------------------------|
| pCMC | 0.902 | 0.336 | +12.7% |
| γ_CMC | 0.809 | 3.52 mN/m | +19.0% |
| Γ_max | 0.805 | 0.62 μmol/m² | +29.8% |
| A_min | 0.868 | 0.24 nm² | +20.6% |
| π_CMC | 0.808 | 3.55 mN/m | +24.3% |
| pC₂₀ | 0.898 | 0.36 | +15.1% |
| **Mean** | **0.848** | - | **+20.3%** |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SurfMT-GNN                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │  AttentiveFP  │  │  Temperature  │  │  Molecular    │       │
│  │  Graph Encoder│  │  Encoder MLP  │  │  Descriptors  │       │
│  │  (256-dim)    │  │  (64-dim)     │  │  (64-dim)     │       │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘       │
│          │                  │                  │                │
│          └──────────────────┼──────────────────┘                │
│                             ▼                                   │
│                    ┌─────────────────┐                          │
│                    │  Fusion Layer   │                          │
│                    │  (384→256→128)  │                          │
│                    └────────┬────────┘                          │
│                             │                                   │
│     ┌───────┬───────┬───────┼───────┬───────┬───────┐          │
│     ▼       ▼       ▼       ▼       ▼       ▼       │          │
│  ┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐                    │
│  │pCMC ││γ_CMC││Γ_max││A_min││π_CMC││pC₂₀ │  Task Heads       │
│  └─────┘└─────┘└─────┘└─────┘└─────┘└─────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Repository Structure

```
SurfMT-GNN/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment
├── CITATION.cff             # Citation information
│
├── configs/                  # Configuration files
│   ├── model/
│   │   └── attentive_fp.yaml
│   ├── training/
│   │   └── default.yaml
│   └── experiment/
│       └── default.yaml
│
├── data/                     # Data files
│   ├── processed/           # Processed PyTorch files
│   │   ├── surfpro_train_with_temp.pt
│   │   ├── surfpro_test_with_temp.pt
│   │   └── target_scaler.json
│   └── splits/              # Cross-validation splits
│       └── cv_splits.json
│
├── src/                      # Source code
│   ├── data/                # Data loading
│   │   ├── __init__.py
│   │   ├── dataset.py       # PyTorch Dataset
│   │   ├── featurizer.py    # SMILES → Graph
│   │   └── transforms.py    # Data transforms
│   │
│   ├── models/              # Model architectures
│   │   ├── __init__.py
│   │   ├── attentive_fp.py  # AttentiveFP encoder
│   │   ├── mtl_model.py     # Multi-task model
│   │   ├── task_heads.py    # Prediction heads
│   │   └── temperature_model.py
│   │
│   ├── features/            # Feature extraction
│   │   ├── __init__.py
│   │   ├── atom_features.py
│   │   ├── bond_features.py
│   │   └── graph_construction.py
│   │
│   ├── training/            # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── losses.py
│   │
│   ├── evaluation/          # Evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   └── analysis/            # Analysis tools
│       ├── __init__.py
│       ├── attention_analysis.py
│       ├── task_correlation.py
│       └── uncertainty.py
│
└── scripts/                  # Executable scripts
    ├── data_preparation/
    │   ├── prepare_data.py
    │   ├── surfpro_data_exploration.py
    │   └── test_pipeline.py
    │
    ├── training/
    │   ├── train_mtl.py           # Main training script
    │   ├── run_ablation_study.py
    │   ├── run_baseline_comparisons.py
    │   ├── run_enhanced_training.py
    │   └── test_model.py
    │
    └── experiments/
        ├── run_ensemble.py        # Ensemble training
        ├── run_analysis.py
        └── generate_paper_figures.py
```

## 🚀 Installation

### Option 1: Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/albakhrani/SurfMT-GNN.git
cd SurfMT-GNN

# Create conda environment
conda env create -f environment.yml
conda activate surfmt-gnn
```

### Option 2: Pip

```bash
# Clone repository
git clone https://github.com/albakhrani/SurfMT-GNN.git
cd SurfMT-GNN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch 2.0.1+
- PyTorch Geometric 2.3.0+
- RDKit 2022.09.5+
- CUDA 11.7+ (for GPU acceleration)

## 📖 Usage

### 1. Data Preparation

```bash
# Prepare data from raw CSV files
python scripts/data_preparation/prepare_data.py

# Run data exploration
python scripts/data_preparation/surfpro_data_exploration.py
```

### 2. Training

```bash
# Train single fold
python scripts/training/train_mtl.py --fold 0 --epochs 500

# Train all 10 folds (cross-validation)
python scripts/training/train_mtl.py --fold -1 --epochs 500

# Train with custom learning rate
python scripts/training/train_mtl.py --fold -1 --lr 5e-4 --epochs 500
```

### 3. Ensemble Training

```bash
# Train 60-model ensemble (10 folds × 6 seeds)
python scripts/experiments/run_ensemble.py --n_models 60 --epochs 500
```

### 4. Evaluation

```bash
# Test trained model
python scripts/training/test_model.py --checkpoint experiments/best_model.pt
```

### 5. Prediction on New Molecules

```python
from src.models import SurfMTGNN
from src.data import MoleculeFeaturizer

# Load trained model
model = SurfMTGNN.load_from_checkpoint('experiments/best_model.pt')
featurizer = MoleculeFeaturizer()

# Predict for new SMILES
smiles = "CCCCCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"  # SDS
graph = featurizer(smiles, temperature=25.0)
predictions = model.predict(graph)

print(f"pCMC: {predictions['pCMC']:.3f}")
print(f"γ_CMC: {predictions['gamma_CMC']:.2f} mN/m")
print(f"Uncertainty: {predictions['uncertainty']:.3f}")
```

## 📊 Dataset

This work uses the **SurfPro database** from:

> Hödl, Stefan L., Luc Hermans, Pim F. J. Dankloff, Aigars Piruska, Wilhelm T. S. Huck, and William E. Robinson. ‘SurfPro – a Curated Database and Predictive Model of Experimental Properties of Surfactants’. Digital Discovery 4, no. 5 (2025): 1176–87. https://doi.org/10.1039/D4DD00393D.

**Original dataset**: https://github.com/BigChemistry-RobotLab/SurfPro

### Dataset Statistics

| Split | Samples | pCMC | γ_CMC | Γ_max | A_min | π_CMC | pC₂₀ |
|-------|---------|------|-------|-------|-------|-------|------|
| Train | 1,484 | 1,255 | 902 | 602 | 608 | 674 | 587 |
| Test | 140 | 140 | 70 | 70 | 70 | 70 | 70 |

### Surfactant Classes

| Class | Count | Percentage |
|-------|-------|------------|
| Anionic | 512 | 33.6% |
| Nonionic | 487 | 32.0% |
| Cationic | 298 | 19.6% |
| Gemini | 138 | 9.1% |
| Zwitterionic | 89 | 5.8% |

## 📈 Results Reproduction

To reproduce the results reported in the paper:

```bash
# 1. Prepare data
python scripts/data_preparation/prepare_data.py

# 2. Run full ensemble training
python scripts/experiments/run_ensemble.py --n_models 60 --epochs 500 --patience 80

# 3. Generate figures
python scripts/experiments/generate_paper_figures.py
```

## 🔬 Ablation Studies

```bash
# Run ablation studies
python scripts/training/run_ablation_study.py

# Compare with baselines
python scripts/training/run_baseline_comparisons.py
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{alfutini2026surfmtgnn,
  title={Multi-Task Graph Neural Networks for Comprehensive Surfactant Property Prediction},
  author={Al-Futini, Abdulhakim Nasser Ali and Huang, Hexin and AL-Bakhrani, Ali A.},
  journal={Digital Discovery},
  year={2026},
  publisher={Royal Society of Chemistry}
  doi={10.1039/d6dd00135a}
}
```

Also cite the SurfPro dataset:

```bibtex
@article{hoedl2025,
  title={{{SurfPro}} – a Curated Database and Predictive Model of Experimental Properties of Surfactants},
  author={Hödl, Stefan L. and Hermans, Luc and Dankloff, Pim F. J. and Piruska, Aigars and Huck, Wilhelm T. S. and Robinson, William E.},
  journal={Digital Discovery},
  volume={4},
  number={5},
  pages={1176--1187},
  year={2025},
  doi={10.1039/D4DD00393D},
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Al-Futini Abdulhakim Nasser Ali** - School of Earth Science and Resources, Chang'an University
- **Huang Hexin** (Corresponding Author) - Chang'an University - hx.huang@outlook.com
- **Ali A. AL-Bakhrani** - School of Software, Dalian University of Technology

## 🙏 Acknowledgments

- Hödl et al. for the SurfPro database
- Chang'an University High-Performance Computing Center for computational resources
- PyTorch Geometric team for the excellent GNN library

## 📧 Contact

For questions or issues, please:
1. Open an issue on GitHub
2. Contact: albakhrani2017@gmail.com

---

**⭐ If you find this work useful, please consider giving it a star!**
