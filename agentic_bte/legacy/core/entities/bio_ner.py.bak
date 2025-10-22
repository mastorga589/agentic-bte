"""
Complete BioNER Tool - Integrated Biomedical Named Entity Recognition and Linking

This module provides the complete BioNER functionality by integrating:
- Entity recognition using multiple strategies
- Entity linking via UMLS and SRI Name Resolver  
- Entity type classification
- ID to name resolution

Migrated and enhanced from original BTE-LLM MCP Server implementation.
"""

import json
import logging
from typing import Dict, List, Any, Optional

from .recognition import BiomedicalEntityRecognizer
from .linking import EntityLinker, EntityResolver
from ...config.settings import get_settings
from ...exceptions.entity_errors import EntityRecognitionError

logger = logging.getLogger(__name__)


class BioNERTool:
    """
    Complete Biological Named Entity Recognition and Linking Tool
    
    This class provides the complete BioNER functionality including:
    - Entity extraction using spaCy/SciSpaCy + LLM
    - Entity type classification
    - Entity linking to biomedical knowledge bases
    - Entity name resolution
    
    Migrated from original BTE-LLM implementation with enhancements.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the BioNER tool
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
        """
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for BioNER tool")
        
        # Initialize components
        self.recognizer = BiomedicalEntityRecognizer(openai_api_key)
        self.linker = EntityLinker(openai_api_key)
        self.resolver = EntityResolver()
    
    def extract_entities(self, query: str) -> List[str]:
        """
        Extract biomedical entities from query text
        
        Args:
            query: Query text to extract entities from
            
        Returns:
            List of extracted entity texts
        """
        return self.recognizer.extract_entities(query)
    
    def classify_entity(self, query: str, entity: str) -> str:
        """
        Classify entity type
        
        Args:
            query: Original query for context
            entity: Entity text to classify
            
        Returns:
            Entity type classification
        """
        return self.recognizer.classify_entity(query, entity)
    
    def link_entities(self, entity_list: List[str], query: str) -> Dict[str, Dict[str, str]]:
        """
        Link entities to their IDs and return entity types
        
        Args:
            entity_list: List of entity texts to link
            query: Original query for context
            
        Returns:
            Dictionary mapping entity text to {id, type}
        """
        return self.linker.link_entities(entity_list, query)
    
    def extract_entity_ids(self, entity_data: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        """Extract entity IDs for backward compatibility"""
        return {entity: data["id"] for entity, data in entity_data.items()}
    
    def resolve_entity_names(self, entity_ids: List[str]) -> Dict[str, str]:
        """
        Resolve entity IDs to human-readable names
        
        Args:
            entity_ids: List of entity IDs to resolve
            
        Returns:
            Dictionary mapping entity IDs to names
        """
        return self.resolver.resolve_multiple(entity_ids)
    
    def extract_and_link(self, query: str, include_types: bool = True) -> Dict[str, Any]:
        """
        Main method to extract and link biological entities
        
        Args:
            query: Query text to process
            include_types: Whether to include entity types in output
            
        Returns:
            Dictionary with entities, IDs, and optionally types
        """
        try:
            # Step 1: Extract entities
            entity_list = self.extract_entities(query)
            
            if not entity_list:
                return {"message": "No entities found"}
            
            # Step 2: Link entities to IDs and get types
            entity_data = self.link_entities(entity_list, query)
            
            if not entity_data:
                return {"message": "No entities could be linked"}
            
            # Step 3: Format results
            if include_types:
                # Return enhanced format with entity types
                result = {
                    "entities": entity_data,
                    "entity_ids": self.extract_entity_ids(entity_data)  # For backward compatibility
                }
            else:
                # Return simple format for backward compatibility
                result = self.extract_entity_ids(entity_data)
            
            # Add entity name resolution if configured
            if self.settings.enable_entity_name_resolution:
                entity_ids = list(self.extract_entity_ids(entity_data).values())
                entity_names = self.resolve_entity_names(entity_ids)
                if entity_names:
                    result["entity_names"] = entity_names
            
            logger.info(f"Successfully processed query with {len(entity_data)} entities")
            logger.debug(f"Final result: {result}")
            
            return result
        
        except Exception as e:
            error_msg = f"Error in biological entity extraction: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get information about available models"""
        return self.recognizer.get_available_models()


# Convenience functions for backward compatibility and easy usage
def extract_and_link_entities(query: str, openai_api_key: Optional[str] = None, include_types: bool = True) -> Dict[str, Any]:
    """
    Convenience function to extract and link entities
    
    Args:
        query: Query text to process
        openai_api_key: Optional OpenAI API key
        include_types: Whether to include entity types
        
    Returns:
        Dictionary with entities and their linked IDs
    """
    tool = BioNERTool(openai_api_key)
    return tool.extract_and_link(query, include_types)


def extract_entities(query: str, openai_api_key: Optional[str] = None) -> List[str]:
    """
    Simple convenience function to extract entities only
    
    Args:
        query: Query text
        openai_api_key: Optional OpenAI API key
        
    Returns:
        List of entity text strings
    """
    tool = BioNERTool(openai_api_key)
    return tool.extract_entities(query)


# Alias for backward compatibility
BiomedicalNER = BioNERTool
