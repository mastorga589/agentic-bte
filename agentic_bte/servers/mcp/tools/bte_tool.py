"""
MCP BTE Tool - BioThings Explorer API Execution

This module provides the MCP tool interface for executing TRAPI queries
against the BioThings Explorer API.

Migrated and enhanced from the original BTE-LLM implementation.
"""

import json
import logging
from typing import Dict, Any

from pydantic import BaseModel, Field

from ....core.knowledge.bte_client import BTEClient
from ....config.settings import get_settings

logger = logging.getLogger(__name__)


class BTECallInput(BaseModel):
    """Input schema for BTE API call tool"""
    json_query: Dict[str, Any] = Field(
        description="TRAPI query object to send to BTE API"
    )
    maxresults: int = Field(
        default=50,
        description="Maximum number of results to return"
    )
    k: int = Field(
        default=5,
        description="Maximum results per entity"
    )


def get_bte_call_tool_definition() -> Dict[str, Any]:
    """Get the MCP tool definition for BTE API calls"""
    return {
        "name": "call_bte_api",
        "description": "Make an API request to BioThings Explorer using a TRAPI query and return structured results",
        "inputSchema": {
            "type": "object",
            "properties": {
                "json_query": {
                    "type": "object",
                    "description": "TRAPI query object to send to BTE API"
                },
                "maxresults": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 1000
                },
                "k": {
                    "type": "integer",
                    "description": "Maximum results per entity",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 50
                }
            },
            "required": ["json_query"]
        }
    }


async def handle_bte_call(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle BTE API call tool calls
    
    Args:
        arguments: Tool call arguments
        
    Returns:
        MCP-formatted response with BTE results
    """
    try:
        json_query = arguments.get("json_query")
        max_results = arguments.get("maxresults", 50)
        k = arguments.get("k", 5)
        
        if not json_query:
            return {
                "error": "json_query parameter is required",
                "content": [
                    {
                        "type": "text",
                        "text": "Error: json_query parameter is required"
                    }
                ]
            }
        
        if not isinstance(json_query, dict):
            return {
                "error": "json_query must be a valid JSON object",
                "content": [
                    {
                        "type": "text",
                        "text": "Error: json_query must be a valid JSON object"
                    }
                ]
            }
        
        logger.info(f"Executing BTE API call with max_results={max_results}, k={k}")
        
        # Initialize BTE client
        bte_client = BTEClient()
        
        # Execute TRAPI query with batching
        results, entity_mappings, metadata = bte_client.execute_trapi_with_batching(
            json_query, max_results, k
        )
        
        if "error" in metadata:
            return {
                "error": metadata["error"],
                "content": [
                    {
                        "type": "text",
                        "text": f"BTE API Error: {metadata['error']}"
                    }
                ]
            }
        
        # Format successful response
        if not results:
            response_text = "No results found from BTE API.\n\n"
            if "message" in metadata:
                response_text += f"API Message: {metadata['message']}"
        else:
            response_text = f"BTE API Results ({len(results)} relationships found):\n\n"
            
            # Show execution metadata
            if metadata:
                response_text += f"Execution Summary:\n"
                response_text += f"- Total batches: {metadata.get('total_batches', 1)}\n"
                response_text += f"- Successful batches: {metadata.get('successful_batches', 1)}\n"
                response_text += f"- Total results: {metadata.get('total_results', len(results))}\n\n"
            
            # Show sample results
            sample_size = min(10, len(results))
            response_text += f"Sample Results (showing {sample_size}/{len(results)}):\n"
            
            for i, result in enumerate(results[:sample_size]):
                subject = result.get("subject", "Unknown")
                predicate = result.get("predicate", "unknown_relation")
                obj = result.get("object", "Unknown")
                
                # Clean up predicate for display
                clean_predicate = predicate.replace("biolink:", "").replace("_", " ")
                
                response_text += f"  {i+1}. {subject} --{clean_predicate}--> {obj}\n"
            
            if len(results) > sample_size:
                response_text += f"\n... and {len(results) - sample_size} more results.\n"
            
            # Show entity mappings if available
            if entity_mappings:
                response_text += f"\nEntity ID Mappings ({len(entity_mappings)} entities):\n"
                for entity_name, entity_id in list(entity_mappings.items())[:5]:
                    response_text += f"  - {entity_name}: {entity_id}\n"
                if len(entity_mappings) > 5:
                    response_text += f"  ... and {len(entity_mappings) - 5} more entities.\n"
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": response_text
                }
            ],
            "results": results,
            "entity_mappings": entity_mappings,
            "metadata": metadata
        }
        
    except Exception as e:
        error_msg = f"Error calling BTE API: {str(e)}"
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