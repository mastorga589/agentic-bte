"""
MCP-backed service adapters that satisfy the service Protocols.

These adapters keep MCP concerns at the boundary, allowing the core to
depend on Protocols instead of MCP directly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..contracts.services import (
    EntityExtractionService,
    TrapiBuilderService,
    BteExecutionService,
)
from ..queries.mcp_integration import call_mcp_tool


class MCPBioNERAdapter(EntityExtractionService):
    async def extract(self, query: str) -> Tuple[List[Dict[str, Any]], float, str]:
        resp = await call_mcp_tool("bio_ner", query=query)
        entities = resp.get("entities", []) or []
        confidence = float(resp.get("confidence", 0.0))
        source = str(resp.get("source", "mcp:bio_ner"))
        return entities, confidence, source


class MCPTrapiAdapter(TrapiBuilderService):
    async def build(
        self,
        query: str,
        entity_data: Dict[str, str],
        failed_trapis: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float, str]:
        resp = await call_mcp_tool(
            "build_trapi_query",
            query=query,
            entity_data=entity_data or {},
            failed_trapis=failed_trapis or [],
        )
        trapi_query = resp.get("query", {}) or {}
        trapi_batches = resp.get("queries", [trapi_query] if trapi_query else []) or []
        confidence = float(resp.get("confidence", 0.0))
        source = str(resp.get("source", "mcp:build_trapi_query"))
        return trapi_query, trapi_batches, confidence, source


class MCPBteAdapter(BteExecutionService):
    async def execute(
        self,
        trapi_query: Dict[str, Any],
        k: int,
        max_results: int,
        predicate: Optional[str] = None,
        query_intent: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
        args: Dict[str, Any] = {
            "json_query": trapi_query,
            "k": k,
            "maxresults": max_results,
        }
        if predicate is not None:
            args["predicate"] = predicate
        if query_intent is not None:
            args["query_intent"] = query_intent
        resp = await call_mcp_tool("call_bte_api", **args)
        results = resp.get("results", []) or []
        entity_mappings = resp.get("entity_mappings", {}) or {}
        metadata = resp.get("metadata", {}) or {}
        return results, entity_mappings, metadata
