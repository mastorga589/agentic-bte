"""
MCP TRAPI Tool - TRAPI Query Building

This module provides the MCP tool interface for building TRAPI queries
from natural language biomedical questions.

Migrated and enhanced from the original BTE-LLM implementation.
"""

import json
import logging
from typing import Dict, Any, List

from pydantic import BaseModel, Field

from ....core.knowledge.trapi import TRAPIQueryBuilder
from ....config.settings import get_settings

logger = logging.getLogger(__name__)


class TRAPIQueryInput(BaseModel):
    """Input schema for TRAPI query building tool"""
    query: str = Field(
        description="Natural language biomedical query to convert to TRAPI format"
    )
    entity_data: Dict[str, str] = Field(
        default={},
        description="Dictionary of entity names to their IDs"
    )
    failed_trapis: List[Dict[str, Any]] = Field(
        default=[],
        description="List of previously failed TRAPI queries to avoid"
    )


def get_trapi_query_tool_definition() -> Dict[str, Any]:
    """Get the MCP tool definition for TRAPI query building"""
    return {
        "name": "build_trapi_query",
        "description": "Build a TRAPI (Translator Reasoner API) query from a natural language biomedical question",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language biomedical query to convert to TRAPI format"
                },
                "entity_data": {
                    "type": "object",
                    "description": "Dictionary of entity names to their IDs",
                    "additionalProperties": {"type": "string"},
                    "default": {}
                },
                "failed_trapis": {
                    "type": "array",
                    "description": "List of previously failed TRAPI queries to avoid",
                    "items": {"type": "object"},
                    "default": []
                }
            },
            "required": ["query"]
        }
    }


async def handle_trapi_query(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle TRAPI query building tool calls
    
    Args:
        arguments: Tool call arguments
        
    Returns:
        MCP-formatted response with TRAPI query
    """
    try:
        query = arguments.get("query")
        entity_data = arguments.get("entity_data", {})
        failed_trapis = arguments.get("failed_trapis", [])
        
        if not query:
            return {
                "error": "Query parameter is required",
                "content": [
                    {
                        "type": "text",
                        "text": "Error: Query parameter is required"
                    }
                ]
            }
        
        logger.info(f"Building TRAPI query for: {query[:100]}...")
        if entity_data:
            logger.info(f"Using entities: {list(entity_data.keys())}")
        
        # Initialize TRAPI query builder
        trapi_builder = TRAPIQueryBuilder()
        
        # Build TRAPI query
        trapi_query = trapi_builder.build_query(query, entity_data, failed_trapis)
        
        if "error" in trapi_query:
            return {
                "error": trapi_query["error"],
                "content": [
                    {
                        "type": "text",
                        "text": f"TRAPI building error: {trapi_query['error']}"
                    }
                ]
            }
        
        # Validate the TRAPI query
        is_valid, validation_message = trapi_builder.validate_trapi_query(trapi_query)
        
        if not is_valid:
            return {
                "error": f"Invalid TRAPI query: {validation_message}",
                "content": [
                    {
                        "type": "text",
                        "text": f"TRAPI validation error: {validation_message}"
                    }
                ]
            }
        
        # Format successful response
        response_text = f"Generated TRAPI query for: {query}\n\n"
        response_text += f"Query Structure:\n"
        response_text += f"```json\n{json.dumps(trapi_query, indent=2)}\n```\n\n"
        response_text += f"Validation: {validation_message}"
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": response_text
                }
            ],
            "trapi_query": trapi_query  # Include raw TRAPI for programmatic use
        }
        
    except Exception as e:
        error_msg = f"Error building TRAPI query: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "content": [
                {
                    "type": "text",
                    "text": error_msg
                }
            ]
        }