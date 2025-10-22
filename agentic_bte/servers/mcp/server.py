#!/usr/bin/env python3
"""
Agentic BTE MCP Server - Main Server Application

This module implements the main MCP server that provides biomedical
NER, TRAPI query building, BTE API calls, and comprehensive query processing.

Migrated and enhanced from the original BTE-LLM implementation.
"""

import asyncio
import logging
import sys
from typing import Any, Sequence

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Import tool modules
from .tools.bio_ner_tool import (
    get_bio_ner_tool_definition,
    handle_bio_ner
)
from .tools.trapi_tool import (
    get_trapi_query_tool_definition,
    handle_trapi_query
)
from .tools.bte_tool import (
    get_bte_call_tool_definition,
    handle_bte_call
)
from .tools.query_tool import (
    get_basic_plan_and_execute_tool_definition,
    handle_basic_plan_and_execute
)
from .tools.got_tool import (
    get_got_tool_definition,
    handle_got_query
)
# GoT tool provides advanced Graph of Thoughts optimization
# This represents the latest research in biomedical query optimization
# using graph-based reasoning with parallel execution
from ...config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentic-bte-mcp-server")


class AgenticBTEMCPServer:
    """Agentic BTE MCP Server implementation"""
    
    def __init__(self):
        self.server = Server("agentic-bte-mcp-server")
        self.settings = get_settings()
        self._register_handlers()
        
        # Log server initialization
        logger.info("Agentic BTE MCP Server initialized")
        logger.info(f"Settings: semantic_classification={self.settings.enable_semantic_classification}")
        logger.info(f"Settings: entity_name_resolution={self.settings.enable_entity_name_resolution}")
    
    def _register_handlers(self):
        """Register all MCP handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available biomedical tools"""
            tools = [
                Tool(**get_bio_ner_tool_definition()),
                Tool(**get_trapi_query_tool_definition()),
                Tool(**get_bte_call_tool_definition()),
                Tool(**get_basic_plan_and_execute_tool_definition()),
                Tool(**get_got_tool_definition()),
            ]
            
            logger.info(f"Listed {len(tools)} available tools")
            return tools
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            """Handle tool calls with comprehensive error handling"""
            logger.info(f"Tool call: {name} with {len(arguments)} arguments")
            logger.debug(f"Arguments: {arguments}")
            
            try:
                # Route to appropriate handler
                if name == "bio_ner":
                    result = await handle_bio_ner(arguments)
                elif name == "build_trapi_query":
                    result = await handle_trapi_query(arguments)
                elif name == "call_bte_api":
                    result = await handle_bte_call(arguments)
                elif name == "basic_plan_and_execute_query":
                    result = await handle_basic_plan_and_execute(arguments)
                elif name == "got_biomedical_query":
                    result = await handle_got_query(arguments)
                else:
                    error_msg = f"Unknown tool: {name}"
                    logger.error(error_msg)
                    return [TextContent(type="text", text=error_msg)]
                
                # Handle error responses
                if "error" in result:
                    logger.error(f"Tool {name} returned error: {result['error']}")
                    return [TextContent(type="text", text=f"Tool Error: {result['error']}")]
                
                # Extract and format content from result
                if "content" in result:
                    contents = []
                    for item in result["content"]:
                        if item["type"] == "text":
                            contents.append(TextContent(type="text", text=item["text"]))
                        # Add support for other content types as needed
                    
                    if not contents:
                        logger.warning(f"Tool {name} returned empty content")
                        return [TextContent(type="text", text=f"Tool {name} completed but returned no content")]
                    
                    logger.info(f"Tool {name} completed successfully")
                    return contents
                else:
                    # Fallback for tools that don't return structured content
                    result_text = str(result)
                    if len(result_text) > 10000:  # Truncate very long responses
                        result_text = result_text[:10000] + "\n\n[Response truncated - too long]" 
                    return [TextContent(type="text", text=result_text)]
                    
            except Exception as e:
                error_msg = f"Error executing tool {name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return [TextContent(type="text", text=error_msg)]
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """List available resources (none for this server currently)"""
            return []
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read resource content (not implemented for this server)"""
            raise ValueError(f"Resource not found: {uri}")
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting Agentic BTE MCP Server...")
        
        # Check system status
        try:
            from ...core.knowledge.knowledge_system import BiomedicalKnowledgeSystem
            knowledge_system = BiomedicalKnowledgeSystem()
            status = knowledge_system.get_system_status()
            
            logger.info(f"System status check:")
            logger.info(f"  - BTE API healthy: {status['bte_client']['healthy']}")
            logger.info(f"  - OpenAI configured: {status['bio_ner']['openai_configured']}")
            logger.info(f"  - SpaCy models: {status['bio_ner']['available_models']}")
            
        except Exception as e:
            logger.warning(f"System status check failed: {e}")
        
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP Server running on stdio")
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="agentic-bte-mcp-server",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


async def main():
    """Main entry point"""
    try:
        server = AgenticBTEMCPServer()
        await server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


def run_server():
    """Synchronous entry point for console script"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


# Alias for backward compatibility
MCPServer = AgenticBTEMCPServer


if __name__ == "__main__":
    run_server()
