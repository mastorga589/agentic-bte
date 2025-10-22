"""
MCP BioNER Tool - Biomedical Entity Recognition and Linking

This module provides the MCP tool interface for biomedical named entity
recognition and linking functionality.

Migrated and enhanced from the original BTE-LLM implementation.
"""

import json
import logging
from typing import Dict, Any, List

from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field

from ....core.entities.bio_ner import BioNERTool
from ....config.settings import get_settings

logger = logging.getLogger(__name__)

# Global cached BioNER tool instance to avoid redundant model loading
_cached_bio_ner_tool = None


def get_cached_bio_ner_tool() -> BioNERTool:
    """
    Get a cached BioNER tool instance to avoid redundant model loading
    
    Returns:
        Cached BioNER tool instance
    """
    global _cached_bio_ner_tool
    
    if _cached_bio_ner_tool is None:
        logger.info("Initializing cached BioNER tool (first time)")
        _cached_bio_ner_tool = BioNERTool()
        logger.info("BioNER tool cached and ready for reuse")
    else:
        logger.debug("Using cached BioNER tool instance")
    
    return _cached_bio_ner_tool


class BioNERInput(BaseModel):
    """Input schema for BioNER tool"""
    query: str = Field(
        description="Query text to extract biological entities from"
    )


def get_bio_ner_tool_definition() -> Dict[str, Any]:
    """Get the MCP tool definition for BioNER"""
    return {
        "name": "bio_ner",
        "description": "Extract biological entities from a query and return them along with their IDs using biomedical NER and entity linking",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query text to extract biological entities from"
                }
            },
            "required": ["query"]
        }
    }


async def handle_bio_ner(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle BioNER tool calls
    
    Args:
        arguments: Tool call arguments containing query
        
    Returns:
        MCP-formatted response with biomedical entities
    """
    try:
        query = arguments.get("query")
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
        
        logger.info(f"Processing BioNER request for query: {query[:100]}...")
        
        # Get cached BioNER tool instance to avoid redundant model loading
        bio_ner_tool = get_cached_bio_ner_tool()
        
        # Extract and link entities
        result = bio_ner_tool.extract_and_link(query, include_types=True)
        
        if "error" in result:
            return {
                "error": result["error"],
                "content": [
                    {
                        "type": "text",
                        "text": f"BioNER Error: {result['error']}"
                    }
                ]
            }
        
        # Format successful response
        response_text = f"Extracted biological entities from query: {query}\n\n"
        
        if "entities" in result and result["entities"]:
            response_text += "Entities with types and IDs:\n"
            # Handle both list and dict formats
            entities = result["entities"]
            if isinstance(entities, list):
                # New format: list of entity objects
                for entity_obj in entities:
                    entity_name = entity_obj.get("name", "Unknown")
                    entity_id = entity_obj.get("id", "Unknown")
                    entity_type = entity_obj.get("type", "Unknown")
                    response_text += f"  - {entity_name}: {entity_id} ({entity_type})\n"
            elif isinstance(entities, dict):
                # Old format: dict with entity names as keys
                for entity, data in entities.items():
                    entity_id = data.get("id", "Unknown")
                    entity_type = data.get("type", "Unknown")
                    response_text += f"  - {entity}: {entity_id} ({entity_type})\n"
        
        if "entity_names" in result and result["entity_names"]:
            response_text += "\nResolved entity names:\n"
            for entity_id, name in result["entity_names"].items():
                response_text += f"  - {entity_id}: {name}\n"
        
        if not result.get("entities"):
            response_text = f"No biomedical entities found in query: {query}"
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": response_text
                }
            ],
            "result_data": result  # Include raw data for programmatic use
        }
        
    except Exception as e:
        error_msg = f"Error in BioNER tool: {str(e)}"
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