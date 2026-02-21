#!/usr/bin/env python3
"""
Models Module for SurfPro MTL-GNN
=================================
Graph neural network models for surfactant property prediction.

Components:
    - AttentiveFPEncoder: Graph encoder with attention
    - TemperatureAwareMTL: Temperature-aware multi-task model
    - SurfProMTL: Standard multi-task model (without temperature)
    - Task heads and utilities

Author: Al-Futini Abdulhakim Nasser Ali
Supervisor: Prof. Huang Hexin
"""

from .attentive_fp import (
    AttentiveFPEncoder,
    AttentiveLayer,
    AttentiveReadout,
    aggregate_node_attention,
)

from .task_heads import (
    TaskHead,
    MultiTaskHeads,
    UncertaintyWeightedHeads,
)

from .mtl_model import (
    SurfProMTL,
)

from .temperature_model import (
    TemperatureAwareMTL,
    TemperatureEncoder,
    create_temperature_model,
)

__all__ = [
    # Encoder
    'AttentiveFPEncoder',
    'AttentiveLayer',
    'AttentiveReadout',
    'aggregate_node_attention',
    
    # Task heads
    'TaskHead',
    'MultiTaskHeads',
    'UncertaintyWeightedHeads',
    
    # Models
    'SurfProMTL',
    'TemperatureAwareMTL',
    'TemperatureEncoder',
    'create_temperature_model',
]
