"""
SmolState - Minimal reimplementation of state training loop.
"""

from .model import StateTransitionPerturbationModel
from .config import SmolStateConfig, create_config
from .data import DataModuleWrapper, create_data_module  
from .train import SmolTrainer, create_trainer
from .checkpoint import CheckpointManager

__version__ = "0.1.0"
__all__ = [
    "StateTransitionPerturbationModel",
    "SmolStateConfig", 
    "create_config",
    "DataModuleWrapper",
    "create_data_module",
    "SmolTrainer",
    "create_trainer", 
    "CheckpointManager",
]