"""
MCP Server Module - Model Context Protocol server implementation

This module provides the MCP server implementation for Agentic BTE,
allowing integration with MCP-compatible clients like Claude Desktop.

Note: Avoid importing the server at package import time to prevent optional
tool dependencies (e.g., experimental GoT tool) from breaking core tool usage.
"""

# Lazy/optional re-exports; do not fail import of this package if server extras are missing
try:  # pragma: no cover
    from .server import AgenticBTEMCPServer, main, run_server  # type: ignore
    __all__ = [
        "AgenticBTEMCPServer",
        "main",
        "run_server",
    ]
except Exception:  # Defer any import errors until server is explicitly used
    __all__ = []
