#!/usr/bin/env python3
"""
Analysis Module for SurfPro MTL-GNN
===================================
Interpretability and analysis tools for understanding model behavior.

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
"""

from .attention_analysis import (
    AttentionAnalyzer,
    visualize_molecule_attention,
    get_important_atoms,
    aggregate_attention_by_atom_type,
    plot_atom_type_importance,
)

from .task_correlation import (
    TaskCorrelationAnalyzer,
    plot_correlation_heatmap,
    plot_task_weights,
    plot_correlation_comparison,
    validate_physical_relationships,
)

from .uncertainty import (
    UncertaintyAnalyzer,
    calibration_analysis,
    compute_confidence_intervals,
    plot_calibration_curve,
    plot_coverage_comparison,
)

__all__ = [
    # Attention analysis
    'AttentionAnalyzer',
    'visualize_molecule_attention',
    'get_important_atoms',
    'aggregate_attention_by_atom_type',
    'plot_atom_type_importance',

    # Task correlation
    'TaskCorrelationAnalyzer',
    'plot_correlation_heatmap',
    'plot_task_weights',
    'plot_correlation_comparison',
    'validate_physical_relationships',

    # Uncertainty
    'UncertaintyAnalyzer',
    'calibration_analysis',
    'compute_confidence_intervals',
    'plot_calibration_curve',
    'plot_coverage_comparison',
]