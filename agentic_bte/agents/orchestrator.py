"""
Biomedical Research Orchestrator - LangGraph Multi-Agent System

This module implements the main orchestrator for the multi-agent biomedical
research system using LangGraph for sophisticated query decomposition, 
iterative knowledge accumulation, and research synthesis.

Migrated and enhanced from the original BTE-LLM implementation.
"""

import time
import logging
from typing import Literal, Dict, Any, Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from .state import (
    AgentState, 
    RouterOutput, 
    AGENT_MEMBERS, 
    create_initial_state,
    ExecutionSummary
)
from .rdf_manager import RDFGraphManager
from .nodes import AnnotatorNode, PlannerNode, BTESearchNode, SummaryNode
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class BiomedicalOrchestrator:
    """
    LangGraph-based multi-agent orchestrator for biomedical research
    
    Coordinates the execution of specialized agents (Annotator, Planner, BTE Search, Summary)
    to conduct sophisticated biomedical research through iterative query decomposition
    and knowledge graph accumulation.
    """
    
    def __init__(self):
        """Initialize the orchestrator with agents and settings"""
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key
        )
        
        # Initialize RDF graph manager
        self.rdf_manager = RDFGraphManager()
        
        # Initialize agent nodes
        self.annotator = AnnotatorNode()
        self.planner = PlannerNode()
        self.bte_search = BTESearchNode()
        self.summary = SummaryNode()
        
        # Build LangGraph workflow
        self.graph = self._build_workflow()
        
        logger.info("Biomedical orchestrator initialized")
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow with agent nodes and routing
        
        Returns:
            Compiled StateGraph for execution
        """
        builder = StateGraph(AgentState)
        
        # Add agent nodes
        builder.add_node("orchestrator", self._orchestrator_node)
        builder.add_node("annotator", self._annotator_wrapper)
        builder.add_node("planner", self._planner_wrapper)
        builder.add_node("BTE_search", self._bte_search_wrapper)
        builder.add_node("summary", self._summary_wrapper)
        
        # Set entry point
        builder.add_edge(START, "orchestrator")
        
        # Compile and return
        return builder.compile()
    
    def _orchestrator_node(self, state: AgentState) -> Command[Literal[*AGENT_MEMBERS, "__end__"]]:
        """
        Central orchestrator node for agent routing decisions
        
        Args:
            state: Current agent state
            
        Returns:
            Command with next agent to execute
        """
        logger.info("Orchestrator making routing decision...")
        
        try:
            # Build system prompt for routing
            system_prompt = """
            You are a biomedical research supervisor with access to a biomedical knowledge graph. 
            You have access to an annotator ("annotator") which can annotate biomedical entities with their IDs, 
            a planner ("planner") which can tell you which single-hop subquery would help you answer the user query, and 
            a knowledge graph tool ("BTE_search") which can only answer single-hop queries.

            In no particular order, your job is to:
            1. If necessary, annotate the biomedical entities within the user query with their IDs using the annotator. ONLY use the annotator for this.
            2. Construct a plan to answer the user query by deconstructing it to subqueries. These subquestions MUST be single-hop questions, and you MUST use the planner to help you with this. Use a mechanistic approach.
            3. You MUST answer the subquery prescribed by the planner before asking the planner for the next one. This will result in a step-wise approach.
            4. Make sure that each subquestion is answered before you provide your final answer.
            5. Given the subquestions and the user query, respond with the team to act next. Each team will perform a task and respond with their results and status.
            6. You are responsible for making sure that the user query is fully answered by the subqueries and their answers before giving your final answer.
            7. Provide the user with a summary of your findings.
            8. When finished, respond with FINISH. Only respond with FINISH if all subqueries have been answered.

            CORE PRINCIPLES:
            - Analyze the user's request thoroughly
            - Select teams strategically and efficiently
            - Provide clear reasoning for worker selection
            - Maintain complete transparency about your problem-solving process
            - Prioritize accuracy and user understanding
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
            ] + state["messages"]
            
            # Get structured routing decision
            next_agent = self.llm.with_structured_output(RouterOutput).invoke(messages)
            goto = next_agent["next"]
            
            # Handle finish condition
            if goto == "FINISH":
                goto = "summary"
            
            logger.info(f"Orchestrator routing to: {goto}")
            
            return Command(goto=goto, update={"next": goto})
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {str(e)}")
            # Default fallback routing
            if not state.get("entity_data"):
                return Command(goto="annotator", update={"next": "annotator"})
            elif not state.get("subQuery"):
                return Command(goto="planner", update={"next": "planner"})
            else:
                return Command(goto="BTE_search", update={"next": "BTE_search"})
    
    def _annotator_wrapper(self, state: AgentState) -> Command[Literal["orchestrator"]]:
        """Wrapper for annotator node with RDF manager access"""
        return self.annotator(state)
    
    def _planner_wrapper(self, state: AgentState) -> Command[Literal["orchestrator"]]:
        """Wrapper for planner node with RDF manager access"""
        return self.planner(state, self.rdf_manager)
    
    def _bte_search_wrapper(self, state: AgentState) -> Command[Literal["orchestrator"]]:
        """Wrapper for BTE search node with RDF manager access"""
        return self.bte_search(state, self.rdf_manager)
    
    def _summary_wrapper(self, state: AgentState) -> Command[Literal["__end__"]]:
        """Wrapper for summary node with RDF manager access"""
        return self.summary(state, self.rdf_manager)
    
    def execute_research(
        self,
        query: str,
        maxresults: int = 50,
        k: int = 5,
        confidence_threshold: float = 0.7,
        recursion_limit: int = 50
    ) -> Dict[str, Any]:
        """
        Execute complete biomedical research workflow
        
        Args:
            query: User's biomedical research query
            maxresults: Maximum results per BTE API call
            k: Maximum results per entity
            confidence_threshold: Minimum confidence threshold
            recursion_limit: Maximum recursion depth for LangGraph
            
        Returns:
            Complete research results with final answer and metadata
        """
        start_time = time.time()
        
        # Clear previous RDF graph state
        self.rdf_manager.clear_graph()
        
        # Create initial state
        initial_state = create_initial_state(
            query=query,
            maxresults=maxresults,
            k=k,
            confidence_threshold=confidence_threshold
        )
        
        logger.info(f"Starting biomedical research for query: {query}")
        
        try:
            # Execute workflow
            final_state = None
            execution_steps = []
            
            for step in self.graph.stream(
                initial_state, 
                {"recursion_limit": recursion_limit}, 
                subgraphs=True
            ):
                execution_steps.append(step)
                final_state = step
                
                # Log progress
                if isinstance(step, dict) and len(step) > 0:
                    step_info = list(step.keys())[0] if step else "unknown"
                    logger.info(f"Executed step: {step_info}")
            
            # Extract final results
            if final_state and isinstance(final_state, dict):
                # Find the final state with summary results
                summary_state = None
                for node_name, node_state in final_state.items():
                    if "final_answer" in node_state:
                        summary_state = node_state
                        break
                
                if summary_state:
                    final_answer = summary_state.get("final_answer", "")
                else:
                    final_answer = "Research completed but no final answer generated."
            else:
                final_answer = "Research workflow did not complete successfully."
            
            # Generate execution summary
            execution_time = time.time() - start_time
            rdf_stats = self.rdf_manager.get_summary_stats()
            
            # Calculate subquery statistics
            subquery_results = []
            if final_state and isinstance(final_state, dict):
                for node_state in final_state.values():
                    if isinstance(node_state, dict) and "execution_metadata" in node_state:
                        metadata = node_state["execution_metadata"]
                        if "subquery_results" in metadata:
                            subquery_results.extend(metadata["subquery_results"])
            
            successful_subqueries = sum(1 for r in subquery_results if r.get("success", False))
            failed_subqueries = len(subquery_results) - successful_subqueries
            avg_confidence = sum(r.get("confidence", 0) for r in subquery_results) / max(len(subquery_results), 1)
            
            summary = ExecutionSummary(
                total_subqueries=len(subquery_results),
                successful_subqueries=successful_subqueries,
                failed_subqueries=failed_subqueries,
                total_results=rdf_stats["total_triples"],
                total_execution_time=execution_time,
                average_confidence=avg_confidence,
                final_answer=final_answer,
                rdf_triples_count=rdf_stats["total_triples"]
            )
            
            logger.info(f"Research completed in {execution_time:.2f}s with {rdf_stats['total_triples']} knowledge triples")
            
            return {
                "success": True,
                "final_answer": final_answer,
                "execution_summary": summary,
                "rdf_graph": self.rdf_manager.get_turtle_representation(),
                "graph_statistics": rdf_stats,
                "subquery_results": subquery_results,
                "execution_steps": len(execution_steps)
            }
            
        except Exception as e:
            logger.error(f"Research execution failed: {str(e)}")
            execution_time = time.time() - start_time
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "final_answer": f"Research failed due to error: {str(e)}",
                "execution_summary": None,
                "rdf_graph": "",
                "graph_statistics": {},
                "subquery_results": [],
                "execution_steps": 0
            }
    
    def get_rdf_graph(self) -> str:
        """
        Get current RDF graph representation
        
        Returns:
            Turtle-formatted RDF graph
        """
        return self.rdf_manager.get_turtle_representation()
    
    def clear_research_state(self):
        """Clear all accumulated research state"""
        self.rdf_manager.clear_graph()
        logger.info("Cleared research state")


# Convenience function for direct execution
def execute_biomedical_research(
    query: str,
    maxresults: int = 50,
    k: int = 5,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Convenience function to execute biomedical research
    
    Args:
        query: Biomedical research query
        maxresults: Maximum results per API call
        k: Maximum results per entity
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Complete research results
    """
    orchestrator = BiomedicalOrchestrator()
    return orchestrator.execute_research(
        query=query,
        maxresults=maxresults,
        k=k,
        confidence_threshold=confidence_threshold
    )