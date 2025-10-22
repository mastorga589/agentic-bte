"""
Models module for shared NLP model management
"""

from .model_manager import (
    get_model_manager,
    get_scispacy_model,
    get_entity_linker,
    get_ner_model
)

__all__ = [
    'get_model_manager',
    'get_scispacy_model',
    'get_entity_linker', 
    'get_ner_model'
]