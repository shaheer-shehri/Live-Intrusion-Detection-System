"""
UNSW-NB15 Preprocessing Pipeline
================================
A modular preprocessing pipeline for the UNSW-NB15 network intrusion detection dataset.

Modules:
    - config: Configuration parameters and design decisions
    - data_loader: Data loading and initial inspection
    - feature_engineer: Feature engineering and selection
    - encoders: Categorical encoding strategies
    - scalers: Numerical scaling strategies
    - pipeline: Main pipeline orchestrator
"""

from .config import PreprocessingConfig
from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .encoders import CategoricalEncoder
from .scalers import NumericalScaler
from .pipeline import PreprocessingPipeline

__all__ = [
    'PreprocessingConfig',
    'DataLoader',
    'FeatureEngineer',
    'CategoricalEncoder',
    'NumericalScaler',
    'PreprocessingPipeline'
]

__version__ = '1.0.0'
