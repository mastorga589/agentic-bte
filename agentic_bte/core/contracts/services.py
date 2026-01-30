"""
Service protocol contracts to decouple core from MCP wrappers.

These protocols define the minimal interfaces used by the production optimizer
so the core can run with or without MCP by swapping concrete adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple


@dataclass
class Entity:
    name: str
    id: str
    type: str
    confidence: float = 0.0


class EntityExtractionService(Protocol):
    async def extract(self, query: str) -> Tuple[List[Dict[str, Any]], float, str]:
        """
        Extract biomedical entities from the query.
        Returns (entities, confidence, source)
        - entities: list of dicts with at least keys: name, id, type[, confidence]
        - confidence: float in [0,1]
        - source: string identifier of the backend
        """
        ...


class TrapiBuilderService(Protocol):
    async def build(
        self,
        query: str,
        entity_data: Dict[str, str],
        failed_trapis: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float, str]:
        """
        Build a single-hop TRAPI query and provide pre-split batches.
        Returns (trapi_query, trapi_batches, confidence, source)
        """
        ...


class BteExecutionService(Protocol):
    async def execute(
        self,
        trapi_query: Dict[str, Any],
        k: int,
        max_results: int,
        predicate: Optional[str] = None,
        query_intent: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
        """
        Execute a TRAPI query against BTE (optionally with predicate/intention for scoring).
        Returns (results, entity_mappings, metadata)
        """
        ...
