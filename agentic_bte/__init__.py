"""Agentic BTE - AI-powered biomedical knowledge graph queries"""

from .core.knowledge.knowledge_system import BiomedicalKnowledgeSystem
from .core.entities.bio_ner import BioNERTool
from .servers.mcp.server import AgenticBTEMCPServer

__version__ = "0.1.0"
__author__ = "BTE Research Team"
__description__ = "Agentic BioThings Explorer - AI-powered biomedical knowledge graph queries"

__all__ = [
    "BiomedicalKnowledgeSystem",
    "BioNERTool", 
    "AgenticBTEMCPServer",
    "__version__",
    "__author__",
    "__description__"
]