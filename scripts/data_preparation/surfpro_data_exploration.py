#!/usr/bin/env python3
"""
SurfPro Dataset Comprehensive Data Exploration
==============================================
Paper 1: Multi-Task Graph Neural Networks for Comprehensive Surfactant Property Prediction

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
Department: Geological Resources and Geological Engineering

Target Journal: Journal of Chemical Information and Modeling (JCIM)

This script performs comprehensive data exploration on the SurfPro dataset:
- surfpro_train.csv: 1,484 training samples
- surfpro_test.csv: 140 held-out test samples  
- surfpro_literature.csv: Full dataset with DOI references

Target Properties:
- pCMC: Critical micelle concentration (-log₁₀M)
- γCMC (AW_ST_CMC): Surface tension at CMC (mN/m)
- Γmax (Gamma_max): Maximum surface excess (μmol/m²)
- Amin (Area_min): Minimum area per molecule (Ų)
- πCMC (Pi_CMC): Surface pressure at CMC (mN/m)
- pC20: Surfactant efficiency (-log₁₀M)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from collections import Counter
from typing import Dict, List, Tuple, Optional
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
    print("✓ RDKit imported successfully")
except ImportError:
    RDKIT_AVAILABLE = False
    print("✗ RDKit not available - SMILES validation will be skipped")

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif'
})

# Define paths
PROJECT_DIR = Path('/mnt/project')
OUTPUT_DIR = Path('/mnt/user-data/outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Target property columns
TARGET_COLUMNS = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20']
TARGET_NAMES = {
    'pCMC': 'pCMC (-log₁₀M)',
    'AW_ST_CMC': 'γCMC (mN/m)',
    'Gamma_max': 'Γmax (μmol/m²)',
    'Area_min': 'Amin (Ų)',
    'Pi_CMC': 'πCMC (mN/m)',
    'pC20': 'pC20 (-log₁₀M)'
}


class SurfProDataExplorer:
    """Comprehensive data explorer for SurfPro dataset."""
    
    def __init__(self):
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.literature_df: Optional[pd.DataFrame] = None
        self.report_lines: List[str] = []
        
    def log(self, message: str, header: bool = False):
        """Log message to report and print."""
        if header:
            separator = "=" * 80
            self.report_lines.append(f"\n{separator}")
            self.report_lines.append(message)
            self.report_lines.append(separator)
            print(f"\n{'='*80}")
            print(message)
            print('='*80)
        else:
            self.report_lines.append(message)
            print(message)
    
    def load_data(self) -> bool:
        """Load all three CSV files."""
        self.log("1. LOADING DATA", header=True)
        
        try:
            # Load training data
            train_path = PROJECT_DIR / 'surfpro_train.csv'
            self.train_df = pd.read_csv(train_path)
            self.log(f"✓ Loaded surfpro_train.csv: {self.train_df.shape[0]:,} rows × {self.train_df.shape[1]} columns")
            
            # Load test data
            test_path = PROJECT_DIR / 'surfpro_test.csv'
            self.test_df = pd.read_csv(test_path)
            self.log(f"✓ Loaded surfpro_test.csv: {self.test_df.shape[0]:,} rows × {self.test_df.shape[1]} columns")
            
            # Load literature data
            lit_path = PROJECT_DIR / 'surfpro_literature.csv'
            self.literature_df = pd.read_csv(lit_path)
            self.log(f"✓ Loaded surfpro_literature.csv: {self.literature_df.shape[0]:,} rows × {self.literature_df.shape[1]} columns")
            
            return True
            
        except Exception as e:
            self.log(f"✗ Error loading data: {e}")
            return False
    
    def show_basic_stats(self):
        """Display basic statistics for each dataset."""
        self.log("2. BASIC STATISTICS", header=True)
        
        datasets = {
            'Training': self.train_df,
            'Test': self.test_df,
            'Literature': self.literature_df
        }
        
        for name, df in datasets.items():
            self.log(f"\n--- {name} Dataset ---")
            self.log(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            self.log(f"\nColumns: {list(df.columns)}")
            self.log(f"\nData Types:\n{df.dtypes.to_string()}")
            self.log(f"\nFirst 5 rows:\n{df.head().to_string()}")
            self.log("")
    
    def analyze_target_properties(self):
        """Analyze target properties statistics."""
        self.log("3. TARGET PROPERTIES ANALYSIS", header=True)
        
        # Map column names for literature dataset
        lit_column_map = {'Surfactant_Type': 'type', 'Temp_Celsius': 'temp'}
        
        # Combined train/test for analysis
        combined = pd.concat([self.train_df, self.test_df], ignore_index=True)
        
        # Statistics table
        stats_data = []
        
        for col in TARGET_COLUMNS:
            train_vals = self.train_df[col].dropna()
            test_vals = self.test_df[col].dropna()
            combined_vals = combined[col].dropna()
            
            train_missing = (self.train_df[col].isna().sum() / len(self.train_df)) * 100
            test_missing = (self.test_df[col].isna().sum() / len(self.test_df)) * 100
            
            stats_data.append({
                'Property': TARGET_NAMES[col],
                'Train N': len(train_vals),
                'Train Missing %': f"{train_missing:.1f}%",
                'Test N': len(test_vals),
                'Test Missing %': f"{test_missing:.1f}%",
                'Mean': f"{combined_vals.mean():.3f}",
                'Std': f"{combined_vals.std():.3f}",
                'Min': f"{combined_vals.min():.3f}",
                'Max': f"{combined_vals.max():.3f}",
                'Range': f"{combined_vals.max() - combined_vals.min():.3f}"
            })
        
        stats_df = pd.DataFrame(stats_data)
        self.log(f"\n{stats_df.to_string(index=False)}")
        
        # Save statistics to CSV
        stats_df.to_csv(OUTPUT_DIR / 'target_properties_statistics.csv', index=False)
        self.log(f"\n✓ Saved statistics to target_properties_statistics.csv")
        
        return stats_df
    
    def analyze_surfactant_classes(self):
        """Analyze surfactant class distribution."""
        self.log("4. SURFACTANT CLASS DISTRIBUTION", header=True)
        
        # Get class distributions
        train_classes = self.train_df['type'].value_counts()
        test_classes = self.test_df['type'].value_counts()
        
        # Literature dataset uses 'Surfactant_Type'
        lit_classes = self.literature_df['Surfactant_Type'].value_counts()
        
        self.log("\n--- Training Set Class Distribution ---")
        for cls, count in train_classes.items():
            pct = (count / len(self.train_df)) * 100
            self.log(f"  {cls}: {count:,} ({pct:.1f}%)")
        
        self.log("\n--- Test Set Class Distribution ---")
        for cls, count in test_classes.items():
            pct = (count / len(self.test_df)) * 100
            self.log(f"  {cls}: {count:,} ({pct:.1f}%)")
        
        self.log("\n--- Literature Dataset Class Distribution ---")
        for cls, count in lit_classes.items():
            pct = (count / len(self.literature_df)) * 100
            self.log(f"  {cls}: {count:,} ({pct:.1f}%)")
        
        # Comparison table
        all_classes = sorted(set(train_classes.index) | set(test_classes.index))
        comparison_data = []
        for cls in all_classes:
            train_n = train_classes.get(cls, 0)
            test_n = test_classes.get(cls, 0)
            total = train_n + test_n
            train_pct = (train_n / total * 100) if total > 0 else 0
            test_pct = (test_n / total * 100) if total > 0 else 0
            comparison_data.append({
                'Class': cls,
                'Train N': train_n,
                'Test N': test_n,
                'Total': total,
                'Train %': f"{train_pct:.1f}%",
                'Test %': f"{test_pct:.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        self.log(f"\n--- Train/Test Split by Class ---\n{comparison_df.to_string(index=False)}")
        
        return train_classes, test_classes
    
    def check_temperature_distribution(self):
        """Analyze temperature distribution."""
        self.log("5. TEMPERATURE DISTRIBUTION", header=True)
        
        # Training set
        train_temps = self.train_df['temp'].value_counts().sort_index()
        self.log("\n--- Training Set Temperature Distribution ---")
        for temp, count in train_temps.items():
            pct = (count / len(self.train_df)) * 100
            self.log(f"  {temp}°C: {count:,} ({pct:.1f}%)")
        
        # Test set
        test_temps = self.test_df['temp'].value_counts().sort_index()
        self.log("\n--- Test Set Temperature Distribution ---")
        for temp, count in test_temps.items():
            pct = (count / len(self.test_df)) * 100
            self.log(f"  {temp}°C: {count:,} ({pct:.1f}%)")
    
    def check_data_quality(self):
        """Perform data quality checks."""
        self.log("6. DATA QUALITY CHECKS", header=True)
        
        combined = pd.concat([self.train_df, self.test_df], ignore_index=True)
        
        # Check for duplicate SMILES
        self.log("\n--- Duplicate SMILES Check ---")
        train_dupes = self.train_df['SMILES'].duplicated().sum()
        test_dupes = self.test_df['SMILES'].duplicated().sum()
        
        # Check overlap between train and test
        train_smiles = set(self.train_df['SMILES'])
        test_smiles = set(self.test_df['SMILES'])
        overlap = train_smiles & test_smiles
        
        self.log(f"  Training set duplicates: {train_dupes}")
        self.log(f"  Test set duplicates: {test_dupes}")
        self.log(f"  Train-Test overlap (data leakage check): {len(overlap)} molecules")
        
        if len(overlap) > 0:
            self.log(f"  ⚠ WARNING: {len(overlap)} molecules appear in both train and test sets!")
            self.log(f"  Overlapping SMILES sample: {list(overlap)[:3]}")
        else:
            self.log(f"  ✓ No data leakage between train and test sets")
        
        # SMILES validity check
        if RDKIT_AVAILABLE:
            self.log("\n--- SMILES Validity Check (RDKit) ---")
            invalid_smiles = []
            
            for idx, smiles in enumerate(combined['SMILES']):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    invalid_smiles.append((idx, smiles))
            
            if invalid_smiles:
                self.log(f"  ⚠ Found {len(invalid_smiles)} invalid SMILES:")
                for idx, smiles in invalid_smiles[:5]:
                    self.log(f"    Row {idx}: {smiles[:50]}...")
            else:
                self.log(f"  ✓ All {len(combined)} SMILES are valid")
        
        # Outlier detection (>3 std from mean)
        self.log("\n--- Outlier Detection (>3σ from mean) ---")
        outlier_summary = []
        
        for col in TARGET_COLUMNS:
            vals = combined[col].dropna()
            if len(vals) > 0:
                mean = vals.mean()
                std = vals.std()
                lower = mean - 3 * std
                upper = mean + 3 * std
                
                outliers = vals[(vals < lower) | (vals > upper)]
                outlier_pct = (len(outliers) / len(vals)) * 100
                
                outlier_summary.append({
                    'Property': TARGET_NAMES[col],
                    'N Valid': len(vals),
                    'Mean': f"{mean:.3f}",
                    'Std': f"{std:.3f}",
                    '3σ Range': f"[{lower:.3f}, {upper:.3f}]",
                    'Outliers': len(outliers),
                    'Outlier %': f"{outlier_pct:.2f}%"
                })
        
        outlier_df = pd.DataFrame(outlier_summary)
        self.log(f"\n{outlier_df.to_string(index=False)}")
        
        return outlier_df
    
    def create_visualizations(self):
        """Create all visualizations."""
        self.log("7. CREATING VISUALIZATIONS", header=True)
        
        combined = pd.concat([self.train_df, self.test_df], ignore_index=True)
        
        # =========================================
        # Figure 1: Histograms of target properties
        # =========================================
        self.log("\n  Creating histograms of target properties...")
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()
        
        colors = plt.cm.Set2(np.linspace(0, 1, 6))
        
        for idx, col in enumerate(TARGET_COLUMNS):
            ax = axes[idx]
            data = combined[col].dropna()
            
            ax.hist(data, bins=30, color=colors[idx], edgecolor='black', alpha=0.7)
            ax.set_xlabel(TARGET_NAMES[col])
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {TARGET_NAMES[col]}\n(n={len(data):,})')
            
            # Add statistics text
            stats_text = f'μ={data.mean():.2f}\nσ={data.std():.2f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('SurfPro Dataset: Target Property Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'fig1_property_histograms.png', dpi=300)
        plt.close()
        self.log("  ✓ Saved fig1_property_histograms.png")
        
        # =========================================
        # Figure 2: Box plots by surfactant class
        # =========================================
        self.log("\n  Creating box plots by surfactant class...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        class_order = combined['type'].value_counts().index.tolist()
        
        for idx, col in enumerate(TARGET_COLUMNS):
            ax = axes[idx]
            
            # Filter to classes with enough data
            plot_data = combined[['type', col]].dropna()
            
            sns.boxplot(data=plot_data, x='type', y=col, order=class_order,
                       palette='Set2', ax=ax)
            ax.set_xlabel('Surfactant Class')
            ax.set_ylabel(TARGET_NAMES[col])
            ax.set_title(f'{TARGET_NAMES[col]} by Class')
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('SurfPro Dataset: Properties by Surfactant Class', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'fig2_boxplots_by_class.png', dpi=300)
        plt.close()
        self.log("  ✓ Saved fig2_boxplots_by_class.png")
        
        # =========================================
        # Figure 3: Correlation heatmap
        # =========================================
        self.log("\n  Creating correlation heatmap...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_data = combined[TARGET_COLUMNS].dropna()
        corr_matrix = corr_data.corr()
        
        # Rename columns for display
        corr_matrix_display = corr_matrix.copy()
        corr_matrix_display.columns = [TARGET_NAMES[c].split(' ')[0] for c in TARGET_COLUMNS]
        corr_matrix_display.index = [TARGET_NAMES[c].split(' ')[0] for c in TARGET_COLUMNS]
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(corr_matrix_display, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, square=True, ax=ax,
                   mask=mask, linewidths=0.5,
                   annot_kws={'size': 11})
        
        ax.set_title('Correlation Matrix of Target Properties\n(Pearson r)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'fig3_correlation_heatmap.png', dpi=300)
        plt.close()
        self.log("  ✓ Saved fig3_correlation_heatmap.png")
        
        # =========================================
        # Figure 4: Missing values heatmap
        # =========================================
        self.log("\n  Creating missing values heatmap...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Train missing values
        train_missing = self.train_df[TARGET_COLUMNS].isna()
        train_missing_pct = train_missing.mean() * 100
        
        # Test missing values
        test_missing = self.test_df[TARGET_COLUMNS].isna()
        test_missing_pct = test_missing.mean() * 100
        
        # Bar chart of missing percentages
        x = np.arange(len(TARGET_COLUMNS))
        width = 0.35
        
        ax = axes[0]
        bars1 = ax.bar(x - width/2, train_missing_pct, width, label='Train', color='steelblue')
        bars2 = ax.bar(x + width/2, test_missing_pct, width, label='Test', color='coral')
        
        ax.set_xlabel('Property')
        ax.set_ylabel('Missing %')
        ax.set_title('Missing Values by Property')
        ax.set_xticks(x)
        ax.set_xticklabels([TARGET_NAMES[c].split(' ')[0] for c in TARGET_COLUMNS], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, max(max(train_missing_pct), max(test_missing_pct)) * 1.2 + 5)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        # Missing pattern heatmap (sample of data)
        ax = axes[1]
        sample_idx = np.random.choice(len(combined), min(100, len(combined)), replace=False)
        sample_missing = combined.iloc[sample_idx][TARGET_COLUMNS].isna().astype(int)
        
        sns.heatmap(sample_missing, cmap='YlOrRd', cbar_kws={'label': 'Missing'}, ax=ax,
                   xticklabels=[TARGET_NAMES[c].split(' ')[0] for c in TARGET_COLUMNS])
        ax.set_xlabel('Property')
        ax.set_ylabel('Sample Index')
        ax.set_title('Missing Value Pattern (100 random samples)')
        
        plt.suptitle('SurfPro Dataset: Missing Values Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'fig4_missing_values.png', dpi=300)
        plt.close()
        self.log("  ✓ Saved fig4_missing_values.png")
        
        # =========================================
        # Figure 5: Class distribution comparison
        # =========================================
        self.log("\n  Creating class distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart for combined dataset
        ax = axes[0]
        combined_classes = combined['type'].value_counts()
        colors_pie = plt.cm.Set2(np.linspace(0, 1, len(combined_classes)))
        
        wedges, texts, autotexts = ax.pie(combined_classes.values, labels=combined_classes.index,
                                          autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax.set_title('Overall Class Distribution\n(Train + Test)')
        
        # Bar chart comparing train vs test
        ax = axes[1]
        train_classes = self.train_df['type'].value_counts()
        test_classes = self.test_df['type'].value_counts()
        
        all_classes = sorted(set(train_classes.index) | set(test_classes.index))
        x = np.arange(len(all_classes))
        width = 0.35
        
        train_vals = [train_classes.get(c, 0) for c in all_classes]
        test_vals = [test_classes.get(c, 0) for c in all_classes]
        
        ax.bar(x - width/2, train_vals, width, label=f'Train (n={len(self.train_df)})', color='steelblue')
        ax.bar(x + width/2, test_vals, width, label=f'Test (n={len(self.test_df)})', color='coral')
        
        ax.set_xlabel('Surfactant Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution: Train vs Test')
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, rotation=45, ha='right')
        ax.legend()
        
        plt.suptitle('SurfPro Dataset: Surfactant Class Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'fig5_class_distribution.png', dpi=300)
        plt.close()
        self.log("  ✓ Saved fig5_class_distribution.png")
        
        # =========================================
        # Figure 6: Pairplot of key properties
        # =========================================
        self.log("\n  Creating pairplot of key properties...")
        
        # Select key properties for pairplot (reduce to 4 for clarity)
        key_props = ['pCMC', 'AW_ST_CMC', 'Gamma_max', 'pC20']
        plot_data = combined[key_props + ['type']].dropna()
        
        g = sns.pairplot(plot_data, hue='type', palette='Set2', diag_kind='kde',
                        plot_kws={'alpha': 0.6, 's': 30}, height=2.5)
        g.fig.suptitle('SurfPro Dataset: Property Relationships by Class', y=1.02, fontsize=14, fontweight='bold')
        
        plt.savefig(OUTPUT_DIR / 'fig6_pairplot.png', dpi=300)
        plt.close()
        self.log("  ✓ Saved fig6_pairplot.png")
        
        self.log("\n  All visualizations saved successfully!")
    
    def generate_summary_report(self):
        """Generate summary report with key findings."""
        self.log("8. SUMMARY REPORT & RECOMMENDATIONS", header=True)
        
        combined = pd.concat([self.train_df, self.test_df], ignore_index=True)
        
        self.log("\n" + "="*60)
        self.log("KEY FINDINGS")
        self.log("="*60)
        
        # Dataset size
        self.log(f"\n📊 DATASET SIZE:")
        self.log(f"   • Training samples: {len(self.train_df):,}")
        self.log(f"   • Test samples: {len(self.test_df):,}")
        self.log(f"   • Train/Test ratio: {len(self.train_df)/len(self.test_df):.1f}:1")
        self.log(f"   • Total unique surfactants: {combined['SMILES'].nunique():,}")
        
        # Class imbalance
        self.log(f"\n📈 CLASS DISTRIBUTION:")
        class_counts = combined['type'].value_counts()
        majority_class = class_counts.index[0]
        minority_class = class_counts.index[-1]
        imbalance_ratio = class_counts.iloc[0] / class_counts.iloc[-1]
        self.log(f"   • Majority class: {majority_class} ({class_counts.iloc[0]:,} samples)")
        self.log(f"   • Minority class: {minority_class} ({class_counts.iloc[-1]:,} samples)")
        self.log(f"   • Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        # Missing values
        self.log(f"\n❓ MISSING VALUES (combined train+test):")
        for col in TARGET_COLUMNS:
            missing_pct = (combined[col].isna().sum() / len(combined)) * 100
            self.log(f"   • {TARGET_NAMES[col]}: {missing_pct:.1f}% missing")
        
        # Property correlations
        self.log(f"\n🔗 NOTABLE PROPERTY CORRELATIONS:")
        corr_matrix = combined[TARGET_COLUMNS].corr()
        strong_corrs = []
        for i, col1 in enumerate(TARGET_COLUMNS):
            for j, col2 in enumerate(TARGET_COLUMNS):
                if i < j:
                    r = corr_matrix.loc[col1, col2]
                    if abs(r) > 0.5:
                        strong_corrs.append((col1, col2, r))
        
        for col1, col2, r in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
            direction = "positive" if r > 0 else "negative"
            self.log(f"   • {TARGET_NAMES[col1].split(' ')[0]} ↔ {TARGET_NAMES[col2].split(' ')[0]}: r={r:.2f} ({direction})")
        
        self.log("\n" + "="*60)
        self.log("DATA CHALLENGES IDENTIFIED")
        self.log("="*60)
        
        challenges = []
        
        # Check missing values
        max_missing = max(combined[col].isna().sum() / len(combined) * 100 for col in TARGET_COLUMNS)
        if max_missing > 30:
            challenges.append(f"1. HIGH MISSING VALUES: Up to {max_missing:.1f}% missing for some properties")
            challenges.append("   → Consider: Multi-task learning can help by sharing information across tasks")
        
        # Check class imbalance
        if imbalance_ratio > 5:
            challenges.append(f"2. CLASS IMBALANCE: {imbalance_ratio:.1f}:1 ratio between majority/minority classes")
            challenges.append("   → Consider: Weighted sampling, class-balanced loss, or stratified splits")
        
        # Check data leakage
        train_smiles = set(self.train_df['SMILES'])
        test_smiles = set(self.test_df['SMILES'])
        overlap = len(train_smiles & test_smiles)
        if overlap > 0:
            challenges.append(f"3. DATA LEAKAGE: {overlap} molecules appear in both train and test")
            challenges.append("   → Consider: Remove duplicates or use scaffold split")
        else:
            challenges.append("3. NO DATA LEAKAGE: Train/test sets are properly separated ✓")
        
        for challenge in challenges:
            self.log(challenge)
        
        self.log("\n" + "="*60)
        self.log("RECOMMENDATIONS FOR PREPROCESSING")
        self.log("="*60)
        
        recommendations = [
            "1. MOLECULAR FEATURIZATION:",
            "   • Use AttentiveFP for graph-based molecular representation",
            "   • Include temperature as an auxiliary feature",
            "   • Consider adding surfactant-specific descriptors (HLB, LogP)",
            "",
            "2. HANDLING MISSING VALUES:",
            "   • Use multi-task learning (MTL) framework",
            "   • Implement masked loss for missing targets",
            "   • Train on all available data per property",
            "",
            "3. CLASS IMBALANCE:",
            "   • Use stratified k-fold cross-validation (fold column exists)",
            "   • Consider class-weighted loss function",
            "   • Ensure each class is represented in train/test",
            "",
            "4. DATA NORMALIZATION:",
            "   • Standardize targets (z-score normalization)",
            "   • Use separate scalers per property",
            "   • Log-transform skewed properties if needed",
            "",
            "5. CROSS-VALIDATION STRATEGY:",
            "   • Use existing 10-fold splits (fold column in train data)",
            "   • Report mean ± std across folds",
            "   • Final evaluation on held-out test set"
        ]
        
        for rec in recommendations:
            self.log(rec)
        
        self.log("\n" + "="*60)
        self.log("NEXT STEPS")
        self.log("="*60)
        
        next_steps = [
            "1. Implement data preprocessing pipeline",
            "2. Build molecular graph featurizer using RDKit + PyTorch Geometric",
            "3. Create data loaders with proper batching",
            "4. Implement baseline single-task GNN",
            "5. Design multi-task architecture with shared encoder",
            "6. Train and evaluate on 10-fold CV",
            "7. Final evaluation on test set"
        ]
        
        for step in next_steps:
            self.log(f"   {step}")
        
        # Save report to file
        report_path = OUTPUT_DIR / 'surfpro_exploration_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.report_lines))
        
        self.log(f"\n✓ Full report saved to: {report_path}")
    
    def run_full_exploration(self):
        """Run complete data exploration pipeline."""
        print("\n" + "="*80)
        print("SURFPRO DATASET COMPREHENSIVE EXPLORATION")
        print("Paper 1: Multi-Task GNN for Surfactant Property Prediction")
        print("="*80 + "\n")
        
        if not self.load_data():
            return False
        
        self.show_basic_stats()
        self.analyze_target_properties()
        self.analyze_surfactant_classes()
        self.check_temperature_distribution()
        self.check_data_quality()
        self.create_visualizations()
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("EXPLORATION COMPLETE!")
        print("="*80)
        print(f"\nOutput files saved to: {OUTPUT_DIR}")
        print("  • fig1_property_histograms.png")
        print("  • fig2_boxplots_by_class.png")
        print("  • fig3_correlation_heatmap.png")
        print("  • fig4_missing_values.png")
        print("  • fig5_class_distribution.png")
        print("  • fig6_pairplot.png")
        print("  • target_properties_statistics.csv")
        print("  • surfpro_exploration_report.txt")
        
        return True


def main():
    """Main entry point."""
    explorer = SurfProDataExplorer()
    success = explorer.run_full_exploration()
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
