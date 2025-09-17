"""
MCP Server Module - Model Context Protocol server implementation

This module provides the MCP server implementation for Agentic BTE,
allowing integration with MCP-compatible clients like Claude Desktop.
"""

from .server import AgenticBTEMCPServer, main, run_server

__all__ = [
    "AgenticBTEMCPServer",
    "main",
    "run_server",
]