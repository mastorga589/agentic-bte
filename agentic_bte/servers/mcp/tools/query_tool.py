"""
MCP Comprehensive Query Tool - End-to-End Biomedical Query Processing

This module provides the MCP tool interface for complete biomedical
query processing from natural language to final results.

Migrated and enhanced from the original BTE-LLM implementation.
"""

import json
import logging
from typing import Dict, Any

from pydantic import BaseModel, Field

from ....core.knowledge.knowledge_system import BiomedicalKnowledgeSystem
from ....config.settings import get_settings

logger = logging.getLogger(__name__)


class PlanAndExecuteInput(BaseModel):
    """Input schema for comprehensive query processing"""
    query: str = Field(
        description="Complex biomedical query to process"
    )
    entities: Dict[str, str] = Field(
        default={},
        description="Pre-extracted biomedical entities (optional)"
    )
    execute_after_plan: bool = Field(
        default=True,
        description="Whether to execute after planning (default: true)"
    )
    max_results: int = Field(
        default=50,
        description="Maximum results per API call"
    )
    k: int = Field(
        default=5,
        description="Maximum results per entity"
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence threshold for results"
    )
    show_plan_details: bool = Field(
        default=True,
        description="Whether to include detailed plan information in output"
    )


def get_plan_and_execute_tool_definition() -> Dict[str, Any]:
    """Get the MCP tool definition for comprehensive query processing"""
    return {
        "name": "plan_and_execute_query",
        "description": "Plan and execute complex biomedical queries using advanced optimization strategies including query decomposition, parallel execution, and dynamic replanning based on intermediate results",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Complex biomedical query to execute with optimization"
                },
                "entities": {
                    "type": "object",
                    "description": "Pre-extracted biomedical entities (optional)",
                    "additionalProperties": {"type": "string"},
                    "default": {}
                },
                "execute_after_plan": {
                    "type": "boolean",
                    "description": "Whether to execute after planning (default: true)",
                    "default": True
                },
                "k": {
                    "type": "integer",
                    "description": "Maximum results per entity",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 50
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results per API call",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 1000
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Minimum confidence threshold for results",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "show_plan_details": {
                    "type": "boolean",
                    "description": "Whether to include detailed plan information in output",
                    "default": True
                }
            },
            "required": ["query"]
        }
    }


async def handle_plan_and_execute(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle comprehensive query processing tool calls
    
    Args:
        arguments: Tool call arguments
        
    Returns:
        MCP-formatted response with complete processing results
    """
    try:
        query = arguments.get("query")
        entities = arguments.get("entities", {})
        execute_after_plan = arguments.get("execute_after_plan", True)
        max_results = arguments.get("max_results", 50)
        k = arguments.get("k", 5)
        confidence_threshold = arguments.get("confidence_threshold", 0.7)
        show_plan_details = arguments.get("show_plan_details", True)
        
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
        
        logger.info(f"Processing comprehensive biomedical query: {query[:100]}...")
        
        # Initialize knowledge system
        knowledge_system = BiomedicalKnowledgeSystem()
        
        # Process the complete query
        result = knowledge_system.process_biomedical_query(query, max_results, k)
        
        if "error" in result:
            return {
                "error": result["error"],
                "content": [
                    {
                        "type": "text",
                        "text": f"Query processing error: {result['error']}\n\nFailed at step: {result.get('step_failed', 'unknown')}"
                    }
                ]
            }
        
        # Format successful response
        response_text = f"Biomedical Query Results for: {query}\n\n"
        
        # Show query classification
        if "query_type" in result:
            classification = result.get("classification", {})
            response_text += f"Query Classification:\n"
            response_text += f"  - Type: {result['query_type']}\n"
            response_text += f"  - Confidence: {classification.get('confidence', 0):.2f}\n"
            response_text += f"  - Method: {classification.get('method', 'unknown')}\n\n"
        
        # Show extracted entities
        if "entities" in result and result["entities"]:
            response_text += f"Extracted Entities ({len(result['entities'])}):\n"
            for entity, data in result["entities"].items():
                entity_id = data.get("id", "Unknown")
                entity_type = data.get("type", "Unknown")
                response_text += f"  - {entity}: {entity_id} ({entity_type})\n"
            response_text += "\n"
        
        # Show results
        if "results" in result and result["results"]:
            results = result["results"]
            response_text += f"Biomedical Relationships Found ({len(results)}):\n\n"
            
            # Show sample results
            sample_size = min(10, len(results))
            for i, relationship in enumerate(results[:sample_size]):
                subject = relationship.get("subject", "Unknown")
                predicate = relationship.get("predicate", "unknown_relation")
                obj = relationship.get("object", "Unknown")
                
                # Clean up predicate for display
                clean_predicate = predicate.replace("biolink:", "").replace("_", " ")
                
                response_text += f"  {i+1}. {subject} --{clean_predicate}--> {obj}\n"
            
            if len(results) > sample_size:
                response_text += f"\n... and {len(results) - sample_size} more relationships.\n"
            
        else:
            response_text += "No biomedical relationships found.\n"
            if "message" in result:
                response_text += f"\nMessage: {result['message']}\n"
        
        # Show execution metadata if available
        if "metadata" in result and show_plan_details:
            metadata = result["metadata"]
            response_text += f"\nExecution Summary:\n"
            response_text += f"  - Total results: {metadata.get('total_results', 0)}\n"
            if "execution_metadata" in metadata:
                exec_meta = metadata["execution_metadata"]
                response_text += f"  - API batches: {exec_meta.get('total_batches', 1)}\n"
                response_text += f"  - Successful batches: {exec_meta.get('successful_batches', 1)}\n"
        
        # Show entity name mappings if available
        if "entity_mappings" in result and result["entity_mappings"]:
            mappings = result["entity_mappings"]
            response_text += f"\nEntity Name Mappings ({len(mappings)} entities):\n"
            for entity_name, entity_id in list(mappings.items())[:5]:
                response_text += f"  - {entity_name}: {entity_id}\n"
            if len(mappings) > 5:
                response_text += f"  ... and {len(mappings) - 5} more entities.\n"
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": response_text
                }
            ],
            "results": result.get("results", []),
            "entities": result.get("entities", {}),
            "entity_mappings": result.get("entity_mappings", {}),
            "metadata": result.get("metadata", {}),
            "query_type": result.get("query_type"),
            "classification": result.get("classification", {})
        }
        
    except Exception as e:
        error_msg = f"Error in comprehensive query processing: {str(e)}"
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