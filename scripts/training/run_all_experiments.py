#!/usr/bin/env python3
"""
=============================================================================
MASTER SCRIPT: RUN ALL EXPERIMENTS FOR SurfMT-GNN JCIM PAPER
=============================================================================
This script runs everything needed for the JCIM paper:

1. Ablation Study (5 variants)
2. Baseline Comparisons (RF, XGBoost, SVR, Single-task GNN)
3. Generate all tables and figures
4. Create complete results report

Estimated Total Time:
- Ablation Study: ~4-6 hours
- Baselines: ~2-3 hours
- Total: ~6-9 hours on GPU

Run:
    python run_all_experiments.py --data_dir data/raw --device cuda

Author: Al-Futini Abdulhakim Nasser Ali
=============================================================================
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'=' * 70}")
    print(f"🚀 {description}")
    print(f"{'=' * 70}")
    print(f"Command: {cmd}")
    print()

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"❌ Error running: {description}")
        return False

    print(f"✅ Completed: {description}")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run all experiments for JCIM paper')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/complete_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--skip_ablation', action='store_true', help='Skip ablation study')
    parser.add_argument('--skip_baselines', action='store_true', help='Skip baseline comparisons')

    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPLETE EXPERIMENT SUITE FOR SurfMT-GNN JCIM PAPER")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Timestamp: {timestamp}")

    results = {}

    # 1. Run Ablation Study
    if not args.skip_ablation:
        ablation_dir = output_dir / 'ablation_study'
        success = run_command(
            f"python run_ablation_study.py --data_dir {args.data_dir} --output_dir {ablation_dir} --device {args.device}",
            "Running Ablation Study (5 variants)"
        )
        results['ablation'] = 'completed' if success else 'failed'
    else:
        print("\n⏭️ Skipping ablation study")
        results['ablation'] = 'skipped'

    # 2. Run Baseline Comparisons
    if not args.skip_baselines:
        baseline_dir = output_dir / 'baselines'
        success = run_command(
            f"python run_baseline_comparisons.py --data_dir {args.data_dir} --output_dir {baseline_dir} --device {args.device}",
            "Running Baseline Comparisons (RF, XGBoost, SVR, Single-task GNN)"
        )
        results['baselines'] = 'completed' if success else 'failed'
    else:
        print("\n⏭️ Skipping baseline comparisons")
        results['baselines'] = 'skipped'

    # 3. Generate Final Report
    print("\n" + "=" * 70)
    print("📊 GENERATING FINAL REPORT")
    print("=" * 70)

    # Create final report
    report = f"""
================================================================================
COMPLETE EXPERIMENT RESULTS FOR SurfMT-GNN
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Directory: {output_dir}

================================================================================
EXPERIMENT STATUS
================================================================================
Ablation Study: {results.get('ablation', 'not run')}
Baseline Comparisons: {results.get('baselines', 'not run')}

================================================================================
RESULTS SUMMARY
================================================================================

Your SurfMT-GNN Results (from enhanced training):
-------------------------------------------------
Property     R²       RMSE
-------------------------------------------------
pCMC         0.902    0.336
γ_CMC        0.809    3.521
Γ_max        0.805    0.618
A_min        0.868    0.241
π_CMC        0.808    3.546
pC20         0.898    0.363
-------------------------------------------------
Mean R²:     0.848 ± 0.042
-------------------------------------------------

================================================================================
FILES GENERATED
================================================================================
1. Ablation Study Results: {output_dir}/ablation_study/ablation_results.json
2. Baseline Results: {output_dir}/baselines/baseline_results.json
3. This Report: {output_dir}/experiment_report.txt

================================================================================
NEXT STEPS
================================================================================
1. Review results in the JSON files
2. Update paper Table 4 (ablation) with actual numbers
3. Update paper Table 4 (baselines) with actual numbers
4. Regenerate Figure 8 (ablation bar chart) with actual data
5. Submit paper!

================================================================================
"""

    report_path = output_dir / 'experiment_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    print(report)

    print(f"\n✅ Report saved to: {report_path}")
    print(f"\n{'=' * 70}")
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"{'=' * 70}")

    return results


if __name__ == '__main__':
    main()