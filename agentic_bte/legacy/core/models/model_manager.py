"""
Model Manager - Singleton for sharing heavy NLP models

This module ensures that expensive NLP models are loaded only once and shared
across all components, eliminating the performance issue of multiple model loading.
"""

import logging
import threading
from typing import Optional, Any
import warnings

# Suppress model loading warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class ModelManager:
    """Singleton manager for heavy NLP models"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Set up logging to be less verbose
        spacy_logger = logging.getLogger("spacy")
        spacy_logger.setLevel(logging.ERROR)
        
        nmslib_logger = logging.getLogger("nmslib")
        nmslib_logger.setLevel(logging.ERROR)
        
        # Initialize model storage
        self._models = {}
        self._initialized = True
        logger.info("ModelManager initialized")
    
    def get_scispacy_model(self, model_name: str = "en_core_sci_lg") -> Any:
        """
        Get or load a spaCy model (cached)
        
        Args:
            model_name: Name of the spaCy model to load
            
        Returns:
            Loaded spaCy model
        """
        if model_name in self._models:
            logger.debug(f"Using cached model: {model_name}")
            return self._models[model_name]
        
        logger.info(f"Loading spaCy model: {model_name} (this may take a moment...)")
        
        try:
            import spacy
            
            # Load with reduced verbosity
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nlp = spacy.load(model_name)
            
            self._models[model_name] = nlp
            logger.info(f"✅ Model {model_name} loaded and cached")
            return nlp
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def get_entity_linker(self, linker_name: str = "umls") -> Any:
        """
        Get or load an entity linker (cached)
        
        Args:
            linker_name: Name of the entity linker to load
            
        Returns:
            Loaded entity linker
        """
        cache_key = f"linker_{linker_name}"
        
        if cache_key in self._models:
            logger.debug(f"Using cached linker: {linker_name}")
            return self._models[cache_key]
        
        logger.info(f"Loading entity linker: {linker_name} (this may take a moment...)")
        
        try:
            from scispacy.linking import EntityLinker
            
            # Load with reduced verbosity
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                linker = EntityLinker(resolve_abbreviations=True, name=linker_name)
            
            self._models[cache_key] = linker
            logger.info(f"✅ Linker {linker_name} loaded and cached")
            return linker
            
        except Exception as e:
            logger.error(f"Failed to load linker {linker_name}: {e}")
            raise
    
    def get_ner_model(self, model_name: str = "en_ner_bc5cdr_md") -> Any:
        """
        Get or load a NER model (cached)
        
        Args:
            model_name: Name of the NER model to load
            
        Returns:
            Loaded NER model
        """
        cache_key = f"ner_{model_name}"
        
        if cache_key in self._models:
            logger.debug(f"Using cached NER model: {model_name}")
            return self._models[cache_key]
        
        logger.info(f"Loading NER model: {model_name} (this may take a moment...)")
        
        try:
            import spacy
            
            # Load with reduced verbosity
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nlp = spacy.load(model_name)
            
            self._models[cache_key] = nlp
            logger.info(f"✅ NER model {model_name} loaded and cached")
            return nlp
            
        except Exception as e:
            logger.error(f"Failed to load NER model {model_name}: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        return {
            "loaded_models": list(self._models.keys()),
            "model_count": len(self._models)
        }
    
    def clear_cache(self):
        """Clear all cached models (use with caution)"""
        logger.warning("Clearing all cached models")
        self._models.clear()


# Global instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get the singleton model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# Convenience functions
def get_scispacy_model(model_name: str = "en_core_sci_lg") -> Any:
    """Get or load a spaCy model (convenience function)"""
    return get_model_manager().get_scispacy_model(model_name)

def get_entity_linker(linker_name: str = "umls") -> Any:
    """Get or load an entity linker (convenience function)"""
    return get_model_manager().get_entity_linker(linker_name)

def get_ner_model(model_name: str = "en_ner_bc5cdr_md") -> Any:
    """Get or load a NER model (convenience function)"""
    return get_model_manager().get_ner_model(model_name)