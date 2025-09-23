"""
MCP Tools Module - Collection of MCP tool implementations

This module contains all the MCP tool implementations for the
Agentic BTE MCP server.
"""

from .bio_ner_tool import (
    get_bio_ner_tool_definition,
    handle_bio_ner
)
from .trapi_tool import (
    get_trapi_query_tool_definition,
    handle_trapi_query
)
from .bte_tool import (
    get_bte_call_tool_definition,
    handle_bte_call
)
from .query_tool import (
    get_plan_and_execute_tool_definition,
    handle_plan_and_execute
)
# Optimization tools are now integrated into plan_and_execute_query
# from .optimization_tools import (...)

__all__ = [
    "get_bio_ner_tool_definition",
    "handle_bio_ner",
    "get_trapi_query_tool_definition", 
    "handle_trapi_query",
    "get_bte_call_tool_definition",
    "handle_bte_call",
    "get_plan_and_execute_tool_definition",
    "handle_plan_and_execute",
]
