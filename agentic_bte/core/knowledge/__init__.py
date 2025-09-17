"""Knowledge system components.

This module provides comprehensive biomedical knowledge retrieval capabilities
including entity recognition, linking, TRAPI query building, and BTE API integration.

Migrated and enhanced from original BTE-LLM implementations.
"""

# Core classes
from .entity_recognition import (
    BioNERTool,
    BiomedicalEntityRecognizer,
    ExtractedEntity,
    LinkedEntity,
    extract_biomedical_entities,
    classify_entity_types
)

from .entity_linking import (
    EntityLinker,
    EntityResolver,
    LinkingCandidate,
    link_entities_to_kb,
    resolve_entity_names
)

from .trapi import (
    TRAPIQueryBuilder,
    build_trapi_query
)

from .bte_client import (
    BTEClient,
    execute_trapi_query,
    get_meta_knowledge_graph
)

from .knowledge_system import (
    IntegratedKnowledgeSystem,
    KnowledgeQuery,
    KnowledgeResult,
    query_biomedical_knowledge,
    batch_query_biomedical_knowledge
)

# Version info
__version__ = "1.0.0"
__author__ = "Agentic BTE Development Team"

# Main exports
__all__ = [
    # Entity Recognition
    "BioNERTool",
    "BiomedicalEntityRecognizer", 
    "ExtractedEntity",
    "LinkedEntity",
    "extract_biomedical_entities",
    "classify_entity_types",
    
    # Entity Linking
    "EntityLinker",
    "EntityResolver",
    "LinkingCandidate",
    "link_entities_to_kb",
    "resolve_entity_names",
    
    # TRAPI Query Building
    "TRAPIQueryBuilder",
    "build_trapi_query",
    
    # BTE Client
    "BTEClient",
    "execute_trapi_query",
    "get_meta_knowledge_graph",
    
    # Integrated System
    "IntegratedKnowledgeSystem",
    "KnowledgeQuery",
    "KnowledgeResult", 
    "query_biomedical_knowledge",
    "batch_query_biomedical_knowledge",
]