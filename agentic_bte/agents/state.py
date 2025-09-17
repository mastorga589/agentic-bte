"""
Agent State Management - LangGraph State Definitions

This module defines the state structures used by the multi-agent system
for maintaining conversation context, query decomposition, entity tracking,
and RDF graph accumulation across agent interactions.

Migrated and enhanced from the original BTE-LLM implementation.
"""

from typing import Literal, List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from operator import add


# Available agent members in the workflow
AGENT_MEMBERS = ["annotator", "planner", "BTE_search"]
ROUTE_OPTIONS = AGENT_MEMBERS + ["FINISH"]


class RouterOutput(TypedDict):
    """Router output for agent selection"""
    next: Literal[*ROUTE_OPTIONS]


class AgentState(MessagesState):
    """
    Enhanced state for the multi-agent biomedical research system
    
    Maintains conversation state, query tracking, entity data, and execution context
    across all agent interactions in the LangGraph workflow.
    """
    
    # Core workflow state
    next: str  # Next agent to execute
    query: str  # Original user query
    subQuery: Annotated[List[str], add]  # Decomposed subqueries
    
    # Entity and knowledge tracking
    entity_data: Annotated[List[Dict[str, Any]], add]  # Extracted entities across iterations
    
    # API execution parameters
    maxresults: int  # Maximum results per BTE API call
    k: int  # Maximum results per entity
    
    # Results and completion
    final_answer: str  # Generated final response
    
    # Enhanced tracking (not in original)
    execution_metadata: Dict[str, Any]  # Track execution stats
    failed_trapis: List[Dict[str, Any]]  # Track failed TRAPI queries for retry logic
    confidence_threshold: float  # Minimum confidence for results
    
    
class SubQueryResult(TypedDict):
    """Structure for individual subquery execution results"""
    subquery: str
    success: bool
    results_count: int
    execution_time: float
    confidence: float
    error_message: Optional[str]


class ExecutionSummary(TypedDict):
    """Summary of complete agent execution"""
    total_subqueries: int
    successful_subqueries: int
    failed_subqueries: int
    total_results: int
    total_execution_time: float
    average_confidence: float
    final_answer: str
    rdf_triples_count: int


def create_initial_state(
    query: str,
    maxresults: int = 50,
    k: int = 5,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Create initial state for agent workflow
    
    Args:
        query: User's biomedical query
        maxresults: Maximum results per BTE API call
        k: Maximum results per entity
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Initial state dictionary for LangGraph workflow
    """
    return {
        "messages": [("human", query)],
        "query": query,
        "subQuery": [],
        "entity_data": [],
        "maxresults": maxresults,
        "k": k,
        "final_answer": "",
        "next": "orchestrator",
        "execution_metadata": {
            "start_time": None,
            "subquery_results": [],
            "total_api_calls": 0,
            "total_results": 0
        },
        "failed_trapis": [],
        "confidence_threshold": confidence_threshold
    }