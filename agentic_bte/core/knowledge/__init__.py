"""Knowledge system components.

This module provides comprehensive biomedical knowledge retrieval capabilities
including entity recognition, linking, TRAPI query building, and BTE API integration.

Migrated and enhanced from original BTE-LLM implementations.
"""

# Core classes - using updated architecture
from ..entities.bio_ner import BioNERTool
from ..entities.linking import EntityLinker

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
    BiomedicalKnowledgeSystem,
    process_query,
    extract_biomedical_entities,
    classify_biomedical_query
)

# Version info
__version__ = "1.0.0"
__author__ = "Agentic BTE Development Team"

# Main exports
__all__ = [
    # Entity Recognition & Linking
    "BioNERTool",
    "EntityLinker",
    
    # TRAPI Query Building
    "TRAPIQueryBuilder",
    "build_trapi_query",
    
    # BTE Client
    "BTEClient",
    "execute_trapi_query",
    "get_meta_knowledge_graph",
    
    # Knowledge System
    "BiomedicalKnowledgeSystem",
    "process_query",
    "extract_biomedical_entities",
    "classify_biomedical_query",
]
