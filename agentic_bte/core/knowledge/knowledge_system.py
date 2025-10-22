"""
Knowledge System Integration - Main interface for biomedical knowledge graph operations

This module provides a unified interface for biomedical knowledge graph operations
including entity recognition, TRAPI query building, and BTE API interactions.

Migrated and enhanced from the original BTE-LLM implementation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

from .bte_client import BTEClient
from .trapi import TRAPIQueryBuilder
from ..entities.bio_ner import BioNERTool
from ..queries.classification import SemanticQueryClassifier
from ...config.settings import get_settings
from ...exceptions.base import ExternalServiceError

logger = logging.getLogger(__name__)


class BiomedicalKnowledgeSystem:
    """
    Unified interface for biomedical knowledge graph operations
    
    This class integrates all the components needed for biomedical
    question answering using knowledge graphs:
    - Entity recognition and linking
    - Query classification  
    - TRAPI query building
    - BTE API interactions
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the biomedical knowledge system
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
        """
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        # Initialize core components
        self.bio_ner = BioNERTool(openai_api_key)
        self.query_classifier = SemanticQueryClassifier(openai_api_key)
        self.trapi_builder = TRAPIQueryBuilder(openai_api_key)
        self.bte_client = BTEClient()
    
    def process_biomedical_query(self, query: str, max_results: int = 50, 
                               k: int = 5) -> Dict[str, Any]:
        """
        Process a complete biomedical query from natural language to results
        
        Args:
            query: Natural language biomedical query
            max_results: Maximum results to return
            k: Maximum results per entity
            
        Returns:
            Complete results with entities, relationships, and metadata
        """
        logger.info(f"Processing biomedical query: {query}")
        
        try:
            # Step 1: Extract and link biomedical entities
            logger.info("Step 1: Extracting biomedical entities...")
            entity_result = self.bio_ner.extract_and_link(query)
            
            if "error" in entity_result:
                return {
                    "error": f"Entity extraction failed: {entity_result['error']}",
                    "step_failed": "entity_extraction"
                }
            
            raw_entities = entity_result.get("entities", [])
            # Normalize entities to a list of dicts [{name, id, ...}]
            if isinstance(raw_entities, dict):
                entities = [{"name": k, "id": v, "type": "general"} for k, v in raw_entities.items()]
            elif isinstance(raw_entities, list):
                entities = raw_entities
            else:
                entities = []

            # Build a simple name->id mapping expected by downstream components
            entity_ids = entity_result.get("entity_ids", {})
            if not isinstance(entity_ids, dict) or not entity_ids:
                try:
                    entity_ids = {e.get("name"): e.get("id") for e in entities if isinstance(e, dict) and e.get("name") and e.get("id")}
                except Exception:
                    entity_ids = {}
            
            if not entities:
                return {
                    "message": "No biomedical entities found in the query",
                    "query": query,
                    "entities": [],
                    "results": []
                }
            
            try:
                example_names = [e.get("name", "") for e in entities][:5]
                logger.info(f"Found {len(entities)} entities: {example_names}")
            except Exception:
                logger.info(f"Found {len(entities)} entities")
            
            # Step 2: Classify query type
            logger.info("Step 2: Classifying query type...")
            classification_result = self.query_classifier.get_classification_confidence(query, entity_ids)
            query_type = classification_result["query_type"]
            
            logger.info(f"Query classified as: {query_type.value} "
                       f"(confidence: {classification_result['confidence']:.2f})")
            
            # Step 3: Build TRAPI query
            logger.info("Step 3: Building TRAPI query...")
            trapi_query = self.trapi_builder.build_trapi_query(query, entity_ids)
            
            if "error" in trapi_query:
                return {
                    "error": f"TRAPI query building failed: {trapi_query['error']}",
                    "step_failed": "trapi_building",
                    "entities": entities,
                    "query_type": query_type.value,
                    "classification": classification_result
                }
            
            logger.info("TRAPI query built successfully")
            
            # Step 4: Execute query against BTE
            logger.info("Step 4: Executing query against BTE...")
            bte_results, entity_mappings, metadata = self.bte_client.execute_trapi_with_batching(
                trapi_query, max_results, k
            )
            
            if not bte_results and "error" in metadata:
                return {
                    "error": f"BTE query execution failed: {metadata['error']}",
                    "step_failed": "bte_execution",
                    "entities": entities,
                    "query_type": query_type.value,
                    "classification": classification_result,
                    "trapi_query": trapi_query
                }
            
            # Step 5: Compile final results
            logger.info(f"Step 5: Compiling results - found {len(bte_results)} relationships")
            
            final_result = {
                "query": query,
                "query_type": query_type.value,
                "classification": classification_result,
                "entities": entities,
                "entity_ids": entity_ids,
                "results": bte_results,
                "entity_mappings": entity_mappings,
                "metadata": {
                    "total_results": len(bte_results),
                    "execution_metadata": metadata,
                    "trapi_query": trapi_query
                }
            }
            
            # Add entity name resolution if enabled
            if self.settings.enable_entity_name_resolution and entity_result.get("entity_names"):
                final_result["entity_names"] = entity_result["entity_names"]
            
            logger.info("Successfully processed biomedical query")
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing biomedical query: {e}")
            return {
                "error": f"Query processing failed: {str(e)}",
                "query": query,
                "step_failed": "general_error"
            }
    
    def extract_entities_only(self, query: str) -> Dict[str, Any]:
        """
        Extract and link entities without full query processing
        
        Args:
            query: Natural language query
            
        Returns:
            Entity extraction results
        """
        return self.bio_ner.extract_and_link(query)
    
    def classify_query_only(self, query: str, entities: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Classify query type without full processing
        
        Args:
            query: Natural language query
            entities: Optional pre-extracted entities
            
        Returns:
            Query classification results
        """
        return self.query_classifier.get_classification_confidence(query, entities)
    
    def build_trapi_only(self, query: str, entity_data: Dict[str, str] = None, 
                        failed_trapis: List[Dict] = None) -> Dict[str, Any]:
        """
        Build TRAPI query without full processing
        
        Args:
            query: Natural language query
            entity_data: Optional entity name to ID mappings
            failed_trapis: Optional failed TRAPI queries to avoid
            
        Returns:
            TRAPI query dictionary
        """
        return self.trapi_builder.build_trapi_query(query, entity_data, failed_trapis)
    
    def execute_trapi_only(self, trapi_query: Dict[str, Any], 
                          max_results: int = 50, k: int = 5) -> Tuple[List[Dict], Dict[str, str], Dict[str, Any]]:
        """
        Execute TRAPI query without full processing
        
        Args:
            trapi_query: TRAPI query to execute
            max_results: Maximum results to return
            k: Maximum results per entity
            
        Returns:
            Tuple of (results, entity_mappings, metadata)
        """
        return self.bte_client.execute_trapi_with_batching(trapi_query, max_results, k)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get status of all system components
        
        Returns:
            System status information
        """
        status = {
            "bio_ner": {
                "available_models": self.bio_ner.get_available_models(),
                "openai_configured": bool(self.openai_api_key)
            },
            "query_classifier": {
                "semantic_classification_enabled": self.settings.enable_semantic_classification,
                "openai_configured": bool(self.openai_api_key)
            },
            "bte_client": {
                "base_url": self.bte_client.base_url,
                "healthy": self.bte_client.health_check()
            },
            "settings": {
                "max_subqueries": self.settings.max_subqueries,
                "confidence_threshold": self.settings.confidence_threshold,
                "max_results_per_query": self.settings.max_results_per_query,
                "entity_name_resolution_enabled": self.settings.enable_entity_name_resolution,
                "query_optimization_enabled": self.settings.enable_query_optimization
            }
        }
        
        return status


# Convenience functions for easy usage
def process_query(query: str, max_results: int = 50, k: int = 5, 
                 openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to process a biomedical query
    
    Args:
        query: Natural language biomedical query
        max_results: Maximum results to return
        k: Maximum results per entity
        openai_api_key: Optional OpenAI API key
        
    Returns:
        Complete processing results
    """
    system = BiomedicalKnowledgeSystem(openai_api_key)
    return system.process_biomedical_query(query, max_results, k)


def extract_biomedical_entities(query: str, openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to extract biomedical entities
    
    Args:
        query: Natural language query
        openai_api_key: Optional OpenAI API key
        
    Returns:
        Entity extraction results
    """
    system = BiomedicalKnowledgeSystem(openai_api_key)
    return system.extract_entities_only(query)


def classify_biomedical_query(query: str, entities: Dict[str, str] = None,
                            openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to classify biomedical query
    
    Args:
        query: Natural language query
        entities: Optional pre-extracted entities
        openai_api_key: Optional OpenAI API key
        
    Returns:
        Query classification results
    """
    system = BiomedicalKnowledgeSystem(openai_api_key)
    return system.classify_query_only(query, entities)

