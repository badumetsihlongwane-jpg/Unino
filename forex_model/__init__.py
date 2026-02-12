"""
MRAT Forex Model Package
Macro-Regime Adaptive Transformer for Forex Prediction
"""

__version__ = "1.0.0"
__author__ = "MRAT Development Team"

from .models.mrat_model import MRATForexModel, MRATConfig, create_mrat_model
from .transformers.data_transformer import DataPipeline
from .utils.training import train_mrat_model, generate_prediction_output

__all__ = [
    'MRATForexModel',
    'MRATConfig',
    'create_mrat_model',
    'DataPipeline',
    'train_mrat_model',
    'generate_prediction_output'
]
