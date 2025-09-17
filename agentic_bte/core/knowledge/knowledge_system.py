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
            
            entities = entity_result.get("entities", {})
            entity_ids = entity_result.get("entity_ids", {})
            
            if not entities:
                return {
                    "message": "No biomedical entities found in the query",
                    "query": query,
                    "entities": {},
                    "results": []
                }
            
            logger.info(f"Found {len(entities)} entities: {list(entities.keys())}")
            
            # Step 2: Classify query type
            logger.info("Step 2: Classifying query type...")
            classification_result = self.query_classifier.get_classification_confidence(query, entity_ids)
            query_type = classification_result["query_type"]
            
            logger.info(f"Query classified as: {query_type.value} "
                       f"(confidence: {classification_result['confidence']:.2f})")
            
            # Step 3: Build TRAPI query
            logger.info("Step 3: Building TRAPI query...")
            trapi_query = self.trapi_builder.build_query(query, entity_ids)
            
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
        return self.trapi_builder.build_query(query, entity_data, failed_trapis)
    
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

"""
Integrated Knowledge System

This module provides a high-level interface for biomedical knowledge retrieval
combining entity recognition, TRAPI query building, and BTE API execution.

Migrated and enhanced from original BTE-LLM implementations.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict

from .entity_recognition import BioNERTool
from .entity_linking import EntityLinker
from .trapi import TRAPIQueryBuilder
from .bte_client import BTEClient
from ...config.settings import get_settings
from ...exceptions.base import KnowledgeRetrievalError

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeQuery:
    """
    Represents a knowledge query with all relevant information
    """
    query: str
    entities: Optional[Dict[str, Any]] = None
    entity_mappings: Optional[Dict[str, str]] = None
    trapi_query: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class KnowledgeResult:
    """
    Represents the results of a knowledge query
    """
    query: str
    success: bool
    results: List[Dict[str, Any]]
    entities: Dict[str, Any]
    entity_mappings: Dict[str, str]
    total_results: int
    confidence_score: float
    metadata: Dict[str, Any]
    error: Optional[str] = None


class IntegratedKnowledgeSystem:
    """
    Integrated system for biomedical knowledge retrieval
    
    This class orchestrates the complete pipeline:
    1. Entity extraction and recognition
    2. Entity linking and ID resolution
    3. TRAPI query construction
    4. BTE API execution
    5. Result parsing and formatting
    
    Migrated from original BTE-LLM implementations with enhancements.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, bte_base_url: Optional[str] = None):
        """
        Initialize the integrated knowledge system
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
            bte_base_url: BTE API base URL
        """
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        # Initialize components
        self.bio_ner = BioNERTool()
        self.entity_linker = EntityLinker()
        self.trapi_builder = TRAPIQueryBuilder(self.openai_api_key)
        self.bte_client = BTEClient(bte_base_url)
        
        logger.info("Initialized Integrated Knowledge System")
    
    def extract_and_link_entities(self, query: str, link_entities: bool = True) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Extract entities from query and optionally link them to knowledge bases
        
        Args:
            query: Natural language query
            link_entities: Whether to perform entity linking
            
        Returns:
            Tuple of (extracted_entities, entity_mappings)
        """
        try:
            # Extract entities using BioNER
            logger.info(f"Extracting entities from query: {query}")
            entities = self.bio_ner.extract_and_link_entities(query) if link_entities else self.bio_ner.extract_entities(query)
            
            # Build entity mappings (name -> ID)
            entity_mappings = {}
            if isinstance(entities, dict) and "entities" in entities:
                for entity in entities["entities"]:
                    if isinstance(entity, dict):
                        name = entity.get("text", "")
                        entity_id = entity.get("entity_id") or entity.get("cui")
                        if name and entity_id:
                            entity_mappings[name] = entity_id
            
            logger.info(f"Extracted {len(entity_mappings)} linked entities")
            return entities, entity_mappings
            
        except Exception as e:
            logger.error(f"Error extracting/linking entities: {e}")
            return {}, {}
    
    def build_trapi_query(self, query: str, entity_mappings: Optional[Dict[str, str]] = None,
                         failed_queries: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Build TRAPI query from natural language query
        
        Args:
            query: Natural language query
            entity_mappings: Entity name to ID mappings
            failed_queries: Previously failed TRAPI queries
            
        Returns:
            TRAPI query dictionary
        """
        try:
            logger.info("Building TRAPI query")
            trapi_query = self.trapi_builder.build_trapi_query(
                query, 
                entity_mappings or {}, 
                failed_queries or []
            )
            
            if "error" in trapi_query:
                logger.error(f"TRAPI query building failed: {trapi_query['error']}")
                
            return trapi_query
            
        except Exception as e:
            logger.error(f"Error building TRAPI query: {e}")
            return {"error": str(e)}
    
    def execute_query(self, trapi_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute TRAPI query against BTE API
        
        Args:
            trapi_query: TRAPI query dictionary
            
        Returns:
            Parsed BTE results
        """
        try:
            logger.info("Executing TRAPI query against BTE")
            results = self.bte_client.query_and_parse(trapi_query)
            return results
            
        except Exception as e:
            logger.error(f"Error executing TRAPI query: {e}")
            return {
                "results": [],
                "entities": {},
                "total_results": 0,
                "query_metadata": {},
                "entity_mappings": {},
                "error": str(e)
            }
    
    def query_knowledge_graph(self, query: str, max_retries: int = 2, 
                            link_entities: bool = True) -> KnowledgeResult:
        """
        Complete knowledge graph querying pipeline
        
        Args:
            query: Natural language biomedical query
            max_retries: Maximum number of retries on failure
            link_entities: Whether to perform entity linking
            
        Returns:
            KnowledgeResult with complete results
        """
        failed_queries = []
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Knowledge graph query attempt {attempt + 1}/{max_retries + 1}")
                
                # Step 1: Extract and link entities
                entities, entity_mappings = self.extract_and_link_entities(query, link_entities)
                
                # Step 2: Build TRAPI query
                trapi_query = self.build_trapi_query(query, entity_mappings, failed_queries)
                
                if "error" in trapi_query:
                    if attempt == max_retries:
                        return KnowledgeResult(
                            query=query,
                            success=False,
                            results=[],
                            entities={},
                            entity_mappings=entity_mappings,
                            total_results=0,
                            confidence_score=0.0,
                            metadata={"attempt": attempt + 1},
                            error=f"TRAPI query building failed: {trapi_query['error']}"
                        )
                    continue
                
                # Step 3: Execute query
                bte_results = self.execute_query(trapi_query)
                
                if "error" in bte_results:
                    logger.warning(f"BTE query failed on attempt {attempt + 1}: {bte_results['error']}")
                    failed_queries.append(trapi_query)
                    if attempt == max_retries:
                        return KnowledgeResult(
                            query=query,
                            success=False,
                            results=[],
                            entities=entities,
                            entity_mappings=entity_mappings,
                            total_results=0,
                            confidence_score=0.0,
                            metadata={"attempt": attempt + 1, "trapi_query": trapi_query},
                            error=f"BTE query execution failed: {bte_results['error']}"
                        )
                    continue
                
                # Step 4: Process and return results
                confidence_score = self._calculate_confidence_score(bte_results, entities)
                
                # Merge entity mappings from BTE results
                combined_entity_mappings = {**entity_mappings, **bte_results.get("entity_mappings", {})}
                
                return KnowledgeResult(
                    query=query,
                    success=True,
                    results=bte_results.get("results", []),
                    entities=bte_results.get("entities", {}),
                    entity_mappings=combined_entity_mappings,
                    total_results=bte_results.get("total_results", 0),
                    confidence_score=confidence_score,
                    metadata={
                        "attempt": attempt + 1,
                        "trapi_query": trapi_query,
                        "query_metadata": bte_results.get("query_metadata", {}),
                        "extracted_entities": entities
                    }
                )
                
            except Exception as e:
                logger.error(f"Knowledge graph query attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    return KnowledgeResult(
                        query=query,
                        success=False,
                        results=[],
                        entities={},
                        entity_mappings={},
                        total_results=0,
                        confidence_score=0.0,
                        metadata={"attempt": attempt + 1},
                        error=str(e)
                    )
        
        # This should never be reached, but just in case
        return KnowledgeResult(
            query=query,
            success=False,
            results=[],
            entities={},
            entity_mappings={},
            total_results=0,
            confidence_score=0.0,
            metadata={"max_retries_exceeded": True},
            error="Maximum retries exceeded"
        )
    
    def _calculate_confidence_score(self, bte_results: Dict[str, Any], 
                                  extracted_entities: Dict[str, Any]) -> float:
        """
        Calculate confidence score for results
        
        Args:
            bte_results: BTE API results
            extracted_entities: Extracted entities from query
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            # Base score from number of results
            total_results = bte_results.get("total_results", 0)
            if total_results == 0:
                return 0.0
            
            # Score based on result quality
            results = bte_results.get("results", [])
            if not results:
                return 0.1
            
            # Average score from individual results
            scores = [result.get("score", 0) for result in results if isinstance(result, dict)]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            # Normalize and combine factors
            result_factor = min(1.0, total_results / 10.0)  # More results = higher confidence
            quality_factor = min(1.0, avg_score)  # Higher individual scores = higher confidence
            
            # Entity matching factor
            entity_factor = 1.0
            if isinstance(extracted_entities, dict) and "entities" in extracted_entities:
                extracted_count = len(extracted_entities["entities"])
                if extracted_count > 0:
                    mapped_count = len(bte_results.get("entity_mappings", {}))
                    entity_factor = min(1.0, mapped_count / extracted_count)
            
            # Combine factors
            confidence = (result_factor * 0.4 + quality_factor * 0.4 + entity_factor * 0.2)
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating confidence score: {e}")
            return 0.0
    
    def batch_query(self, queries: List[str], **kwargs) -> List[KnowledgeResult]:
        """
        Execute multiple knowledge graph queries
        
        Args:
            queries: List of natural language queries
            **kwargs: Additional arguments for query_knowledge_graph
            
        Returns:
            List of KnowledgeResult objects
        """
        results = []
        for i, query in enumerate(queries):
            logger.info(f"Processing batch query {i + 1}/{len(queries)}: {query}")
            result = self.query_knowledge_graph(query, **kwargs)
            results.append(result)
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Check health status of all system components
        
        Returns:
            Health status dictionary
        """
        status = {
            "system": "healthy",
            "components": {},
            "timestamp": None
        }
        
        try:
            # Check BTE API health
            bte_healthy = self.bte_client.health_check()
            status["components"]["bte_api"] = "healthy" if bte_healthy else "unhealthy"
            
            # Check entity recognition (basic test)
            try:
                test_entities = self.bio_ner.extract_entities("diabetes")
                status["components"]["entity_recognition"] = "healthy"
            except Exception as e:
                logger.warning(f"Entity recognition health check failed: {e}")
                status["components"]["entity_recognition"] = "unhealthy"
            
            # Check TRAPI builder (basic test)
            try:
                test_nodes = self.trapi_builder.identify_nodes("What drugs treat diabetes?")
                status["components"]["trapi_builder"] = "healthy" if test_nodes else "unhealthy"
            except Exception as e:
                logger.warning(f"TRAPI builder health check failed: {e}")
                status["components"]["trapi_builder"] = "unhealthy"
            
            # Overall system health
            unhealthy_components = [k for k, v in status["components"].items() if v == "unhealthy"]
            if unhealthy_components:
                status["system"] = "degraded" if len(unhealthy_components) < len(status["components"]) else "unhealthy"
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            status["system"] = "unhealthy"
            status["error"] = str(e)
        
        return status


# Convenience functions
def query_biomedical_knowledge(query: str, openai_api_key: Optional[str] = None,
                             bte_base_url: Optional[str] = None, **kwargs) -> KnowledgeResult:
    """
    Convenience function for single knowledge graph query
    
    Args:
        query: Natural language biomedical query
        openai_api_key: Optional OpenAI API key
        bte_base_url: Optional BTE API base URL
        **kwargs: Additional arguments for query_knowledge_graph
        
    Returns:
        KnowledgeResult
    """
    system = IntegratedKnowledgeSystem(openai_api_key, bte_base_url)
    return system.query_knowledge_graph(query, **kwargs)


def batch_query_biomedical_knowledge(queries: List[str], 
                                   openai_api_key: Optional[str] = None,
                                   bte_base_url: Optional[str] = None,
                                   **kwargs) -> List[KnowledgeResult]:
    """
    Convenience function for batch knowledge graph queries
    
    Args:
        queries: List of natural language queries
        openai_api_key: Optional OpenAI API key
        bte_base_url: Optional BTE API base URL
        **kwargs: Additional arguments for query_knowledge_graph
        
    Returns:
        List of KnowledgeResult objects
    """
    system = IntegratedKnowledgeSystem(openai_api_key, bte_base_url)
    return system.batch_query(queries, **kwargs)