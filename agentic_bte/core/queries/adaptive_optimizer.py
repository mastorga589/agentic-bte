"""
Adaptive Query Optimizer - LangGraph-inspired dynamic query planning

This module implements an adaptive query optimization approach inspired by the
LangGraph planner node, featuring contextual planning, mechanistic decomposition,
and dynamic replanning based on intermediate results.

Key features:
- Context-aware subquery generation
- Mechanistic reasoning chains
- Adaptive replanning based on results
- Single-hop query focus
- Result-driven iteration
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from copy import deepcopy

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..entities.bio_ner import BioNERTool
from ..knowledge.trapi import TRAPIQueryBuilder
from ..knowledge.bte_client import BTEClient
from ...config.settings import get_settings
from ...exceptions.base import ExternalServiceError

logger = logging.getLogger(__name__)


class AdaptivePlanningStrategy(Enum):
    """Adaptive planning strategies for different query types"""
    MECHANISTIC_CHAIN = "mechanistic_chain"     # Drug -> Target -> Pathway -> Disease
    BIDIRECTIONAL_SEARCH = "bidirectional"     # Forward + backward convergence  
    ITERATIVE_REFINEMENT = "iterative"        # Progressively refined queries
    EXPLORATORY_BRANCHING = "exploratory"     # Multiple exploration paths
    HYPOTHESIS_TESTING = "hypothesis"         # Test specific hypotheses


class ExecutionContext(Enum):
    """Current execution context for adaptive planning"""
    INITIAL = "initial"                        # First subquery
    INTERMEDIATE = "intermediate"              # Middle of execution
    REFINEMENT = "refinement"                 # Refining based on results
    CONVERGENCE = "convergence"               # Bringing results together
    FINALIZATION = "finalization"            # Final synthesis


@dataclass
class AdaptiveSubquery:
    """Adaptive subquery with context and reasoning"""
    id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    reasoning: str = ""
    context_dependencies: List[str] = field(default_factory=list)
    expected_relationships: List[str] = field(default_factory=list)
    execution_context: ExecutionContext = ExecutionContext.INITIAL
    priority: int = 1
    results: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class AdaptivePlan:
    """Adaptive execution plan with dynamic replanning capabilities"""
    id: str = field(default_factory=lambda: str(uuid4()))
    original_query: str = ""
    strategy: AdaptivePlanningStrategy = AdaptivePlanningStrategy.MECHANISTIC_CHAIN
    subqueries: List[AdaptiveSubquery] = field(default_factory=list)
    executed_subqueries: List[AdaptiveSubquery] = field(default_factory=list)
    current_context: Dict[str, Any] = field(default_factory=dict)
    accumulated_results: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    replanning_count: int = 0
    total_execution_time: float = 0.0
    final_answer: str = ""


class AdaptiveQueryOptimizer:
    """
    Adaptive Query Optimizer inspired by LangGraph planner node
    
    This optimizer creates dynamic execution plans that adapt based on 
    intermediate results, using contextual reasoning and mechanistic 
    decomposition approaches.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the adaptive query optimizer
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
        """
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for adaptive query optimization")
        
        # Initialize components
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
        self.bio_ner = BioNERTool(openai_api_key)
        self.trapi_builder = TRAPIQueryBuilder(openai_api_key)
        self.bte_client = BTEClient()
        
        # Adaptive planning cache
        self._active_plans: Dict[str, AdaptivePlan] = {}
        
    def determine_planning_strategy(self, query: str, entities: Dict[str, str]) -> AdaptivePlanningStrategy:
        """
        Determine the best adaptive planning strategy for the query
        
        Args:
            query: Original biomedical query
            entities: Extracted biomedical entities
            
        Returns:
            Recommended planning strategy
        """
        query_lower = query.lower()
        
        # Check for mechanistic indicators
        mechanistic_terms = ["mechanism", "how does", "mode of action", "pathway", "target", "interact"]
        if any(term in query_lower for term in mechanistic_terms):
            return AdaptivePlanningStrategy.MECHANISTIC_CHAIN
        
        # Check for treatment/therapeutic queries
        treatment_terms = ["treat", "therapy", "therapeutic", "drug", "medication"]
        if any(term in query_lower for term in treatment_terms):
            return AdaptivePlanningStrategy.BIDIRECTIONAL_SEARCH
        
        # Check for hypothesis testing
        hypothesis_terms = ["whether", "if", "test", "validate", "confirm"]
        if any(term in query_lower for term in hypothesis_terms):
            return AdaptivePlanningStrategy.HYPOTHESIS_TESTING
        
        # Check for exploratory queries
        exploratory_terms = ["what", "which", "find", "identify", "discover"]
        if any(term in query_lower for term in exploratory_terms):
            return AdaptivePlanningStrategy.EXPLORATORY_BRANCHING
        
        # Default to iterative refinement
        return AdaptivePlanningStrategy.ITERATIVE_REFINEMENT
    
    def create_adaptive_plan(self, 
                           query: str, 
                           entities: Dict[str, str] = None,
                           max_iterations: int = 10) -> AdaptivePlan:
        """
        Create initial adaptive execution plan
        
        Args:
            query: Original biomedical query
            entities: Optional pre-extracted entities
            max_iterations: Maximum number of adaptive iterations
            
        Returns:
            Initial adaptive plan
        """
        logger.info(f"Creating adaptive plan for query: {query[:100]}...")
        
        # Extract entities if needed
        if entities is None:
            entity_result = self.bio_ner.extract_and_link(query)
            entities = entity_result.get("entity_ids", {})
        
        # Determine strategy
        strategy = self.determine_planning_strategy(query, entities)
        logger.info(f"Selected adaptive strategy: {strategy.value}")
        
        # Create initial plan
        plan = AdaptivePlan(
            original_query=query,
            strategy=strategy,
            current_context={
                "entities": entities,
                "max_iterations": max_iterations,
                "iteration_count": 0,
                "knowledge_graph_types": [
                    "Disease", "PhysiologicalProcess", "BiologicalEntity", 
                    "Gene", "PathologicalProcess", "Polypeptide", 
                    "SmallMolecule", "PhenotypicFeature"
                ],
                "predicate_categories": [
                    "causality", "treatment", "genetics & biomarkers",
                    "interactions", "phenotypes & diagnostics", 
                    "responses & effects", "associations",
                    "structural/hierarchical relationships"
                ]
            }
        )
        
        # Generate initial subquery
        initial_subquery = self._generate_contextual_subquery(
            plan, ExecutionContext.INITIAL, ""
        )
        
        if initial_subquery:
            plan.subqueries.append(initial_subquery)
        
        # Cache the plan
        self._active_plans[plan.id] = plan
        
        logger.info(f"Created adaptive plan with initial subquery: {initial_subquery.query if initial_subquery else 'None'}")
        return plan
    
    def _generate_contextual_subquery(self, 
                                    plan: AdaptivePlan,
                                    context: ExecutionContext,
                                    current_results: str) -> Optional[AdaptiveSubquery]:
        """
        Generate next subquery based on current context and results
        
        Args:
            plan: Current adaptive plan
            context: Current execution context
            current_results: Current accumulated results (RDF/structured format)
            
        Returns:
            Next adaptive subquery or None if planning complete
        """
        try:
            # Build contextual prompt based on LangGraph planner approach
            prompt = self._build_contextual_prompt(plan, context, current_results)
            
            # Get LLM response
            messages = [{"role": "system", "content": prompt}]
            response = self.llm.invoke(messages)
            subquery_text = response.content.strip()
            
            # Parse potential reasoning if included
            reasoning = ""
            if "REASONING:" in subquery_text:
                parts = subquery_text.split("REASONING:")
                subquery_text = parts[0].strip()
                reasoning = parts[1].strip() if len(parts) > 1 else ""
            
            # Check for completion signal
            if any(term in subquery_text.lower() for term in ["complete", "finished", "done", "no more"]):
                logger.info("Planning completed - no more subqueries needed")
                return None
            
            # Create adaptive subquery
            subquery = AdaptiveSubquery(
                query=subquery_text,
                reasoning=reasoning,
                execution_context=context,
                context_dependencies=list(plan.current_context.get("entities", {}).keys()),
                priority=len(plan.subqueries) + 1
            )
            
            logger.info(f"Generated contextual subquery: {subquery.query}")
            if reasoning:
                logger.debug(f"Reasoning: {reasoning}")
            
            return subquery
            
        except Exception as e:
            logger.error(f"Error generating contextual subquery: {e}")
            return None
    
    def _build_contextual_prompt(self, 
                               plan: AdaptivePlan,
                               context: ExecutionContext,
                               current_results: str) -> str:
        """
        Build contextual prompt based on LangGraph planner approach
        
        Args:
            plan: Current adaptive plan
            context: Current execution context
            current_results: Accumulated results so far
            
        Returns:
            Contextual planning prompt
        """
        base_prompt = f"""Your role is to create a plan to answer the main query by dividing it into several simpler single hop subqueries. 
The system has access to a knowledge graph where necessary data will be retrieved from to answer the query.

This will be an iterative process where answers to previous subqueries might be relevant in constructing future queries. 
Initial subqueries should encompass the overall query path; each subsequent subquery should be more refined. A mechanistic approach should be used in developing the query plan.

For example, for "Which drugs can treat Crohn's disease by targeting the inflammatory response?", 
the first subquery might be "Which drugs can treat Crohn's disease?" followed by "Which genes do these drugs target?", 
then "Which of these genes are related to the inflammatory response?".

Here are the node types present in the knowledge graph: {', '.join(plan.current_context['knowledge_graph_types'])}

The predicates in the knowledge graph can be grouped into: {', '.join(plan.current_context['predicate_categories'])}.
Restrict your queries to within these relationships.

ORIGINAL QUERY: {plan.original_query}
STRATEGY: {plan.strategy.value}
CONTEXT: {context.value}
ITERATION: {plan.current_context.get('iteration_count', 0)}/{plan.current_context.get('max_iterations', 10)}

Here are the current results so far:
{current_results if current_results else "No results yet"}

Make sure that each subquery interrogates discrete relationships between node types. 
For example, instead of directly asking "What are the mechanisms of action of these drugs?", 
you must create subqueries that can help form a reasoning chain to answer the question ("What genes do these drugs interact with?" and "Which physiological processes are these genes involved in?")

If a subquery results in "No results found", rephrase/reframe the question or explore the question from a different angle.

Do NOT prescribe which nodes to use in your subquery, the nodes and predicates are only for your reference.
You need to determine what the next single-hop subquery is and formulate it into a natural language subquestion.

Your response MUST ONLY CONTAIN the single-hop natural language subquestion (for example, "What genes does doxorubicin target?"). 
Please DO NOT include your thoughts or anything else as it will interfere with downstream processes.

If you believe all necessary subqueries have been generated and the original query can be answered, respond with "COMPLETE".
"""
        
        # Add context-specific guidance
        if context == ExecutionContext.INITIAL:
            base_prompt += "\n\nThis is the FIRST subquery. Start with the most fundamental relationship in the query."
        elif context == ExecutionContext.INTERMEDIATE:
            base_prompt += "\n\nThis is an INTERMEDIATE step. Build upon previous results to explore the next logical relationship."
        elif context == ExecutionContext.REFINEMENT:
            base_prompt += "\n\nThis is a REFINEMENT step. The previous query may not have yielded results - try a different angle or broader/narrower scope."
        elif context == ExecutionContext.CONVERGENCE:
            base_prompt += "\n\nThis is a CONVERGENCE step. Focus on connecting different parts of the research together."
        
        return base_prompt
    
    def execute_adaptive_plan(self, plan: AdaptivePlan) -> Dict[str, Any]:
        """
        Execute adaptive plan with dynamic replanning
        
        Args:
            plan: Adaptive plan to execute
            
        Returns:
            Complete execution results with final answer
        """
        logger.info(f"Executing adaptive plan for query: {plan.original_query[:100]}...")
        start_time = time.time()
        
        try:
            while (plan.current_context.get("iteration_count", 0) < 
                   plan.current_context.get("max_iterations", 10)):
                
                iteration = plan.current_context.get("iteration_count", 0)
                logger.info(f"Adaptive iteration {iteration + 1}")
                
                # Get next subquery
                if not plan.subqueries:
                    logger.warning("No subqueries available")
                    break
                
                current_subquery = plan.subqueries.pop(0)
                
                # Execute subquery
                execution_result = self._execute_subquery(current_subquery, plan)
                
                # Update plan with results
                current_subquery.results = execution_result.get("results", [])
                current_subquery.success = execution_result.get("success", False)
                current_subquery.execution_time = execution_result.get("execution_time", 0.0)
                current_subquery.metadata = execution_result.get("metadata", {})
                
                plan.executed_subqueries.append(current_subquery)
                plan.accumulated_results.extend(current_subquery.results)
                plan.reasoning_chain.append(f"Q{iteration+1}: {current_subquery.query} -> {len(current_subquery.results)} results")
                
                # Check if we should continue or replan
                context = self._determine_next_context(plan, current_subquery)
                
                if context == ExecutionContext.FINALIZATION:
                    logger.info("Adaptive planning reached finalization context")
                    break
                
                # Generate next subquery if needed
                current_results_summary = self._summarize_current_results(plan)
                next_subquery = self._generate_contextual_subquery(
                    plan, context, current_results_summary
                )
                
                if next_subquery:
                    plan.subqueries.append(next_subquery)
                else:
                    logger.info("No more subqueries generated - planning complete")
                    break
                
                # Update iteration count
                plan.current_context["iteration_count"] = iteration + 1
                
                # Check for replanning if needed
                if not current_subquery.success and iteration < 3:
                    logger.info("Replanning due to failed subquery")
                    self._replan_strategy(plan, current_subquery)
                    plan.replanning_count += 1
            
            # Generate final answer
            plan.final_answer = self._synthesize_final_answer(plan)
            plan.total_execution_time = time.time() - start_time
            
            logger.info(f"Adaptive execution completed in {plan.total_execution_time:.2f}s with {len(plan.accumulated_results)} total results")
            
            return {
                "success": True,
                "plan_id": plan.id,
                "original_query": plan.original_query,
                "strategy": plan.strategy.value,
                "executed_subqueries": len(plan.executed_subqueries),
                "total_results": len(plan.accumulated_results),
                "execution_time": plan.total_execution_time,
                "replanning_count": plan.replanning_count,
                "reasoning_chain": plan.reasoning_chain,
                "final_answer": plan.final_answer,
                "results": plan.accumulated_results[:50]  # Limit for response size
            }
            
        except Exception as e:
            logger.error(f"Error executing adaptive plan: {e}")
            plan.total_execution_time = time.time() - start_time
            
            return {
                "success": False,
                "error": str(e),
                "plan_id": plan.id,
                "executed_subqueries": len(plan.executed_subqueries),
                "execution_time": plan.total_execution_time,
                "partial_results": plan.accumulated_results
            }
    
    def _execute_subquery(self, subquery: AdaptiveSubquery, plan: AdaptivePlan) -> Dict[str, Any]:
        """
        Execute individual subquery using BTE
        
        Args:
            subquery: Subquery to execute
            plan: Current plan context
            
        Returns:
            Execution results
        """
        logger.info(f"Executing subquery: {subquery.query}")
        start_time = time.time()
        
        try:
            # Build TRAPI query
            entities = plan.current_context.get("entities", {})
            trapi_query = self.trapi_builder.build_trapi_query(
                subquery.query, entities
            )
            
            if "error" in trapi_query:
                logger.error(f"TRAPI building failed: {trapi_query['error']}")
                return {
                    "success": False,
                    "results": [],
                    "execution_time": time.time() - start_time,
                    "error": trapi_query["error"]
                }
            
            # Execute via BTE
            bte_results, entity_mappings, metadata = self.bte_client.execute_trapi_with_batching(
                trapi_query, max_results=50, k=5
            )
            
            execution_time = time.time() - start_time
            
            return {
                "success": len(bte_results) > 0,
                "results": bte_results,
                "execution_time": execution_time,
                "metadata": {
                    **metadata,
                    "entity_mappings": entity_mappings,
                    "trapi_query": trapi_query
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing subquery: {e}")
            return {
                "success": False,
                "results": [],
                "execution_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _determine_next_context(self, plan: AdaptivePlan, last_subquery: AdaptiveSubquery) -> ExecutionContext:
        """
        Determine next execution context based on current state
        
        Args:
            plan: Current plan
            last_subquery: Last executed subquery
            
        Returns:
            Next execution context
        """
        iteration = plan.current_context.get("iteration_count", 0)
        
        # Check if we should finalize
        if iteration >= 5 or len(plan.accumulated_results) >= 100:
            return ExecutionContext.FINALIZATION
        
        # Check if last query failed - need refinement
        if not last_subquery.success:
            return ExecutionContext.REFINEMENT
        
        # Check if we have enough diverse results for convergence
        if len(plan.executed_subqueries) >= 3 and len(plan.accumulated_results) > 20:
            return ExecutionContext.CONVERGENCE
        
        # Default to intermediate
        return ExecutionContext.INTERMEDIATE
    
    def _summarize_current_results(self, plan: AdaptivePlan) -> str:
        """
        Summarize current results for contextual planning
        
        Args:
            plan: Current plan with accumulated results
            
        Returns:
            Summary of current results
        """
        if not plan.accumulated_results:
            return "No results yet"
        
        # Create summary format similar to RDF turtle used in LangGraph
        summary_lines = []
        for i, result in enumerate(plan.accumulated_results[:10]):  # Limit to first 10
            subject = result.get("subject_name", result.get("subject_id", "unknown"))
            predicate = result.get("predicate", "unknown")
            obj = result.get("object_name", result.get("object_id", "unknown"))
            summary_lines.append(f"{subject} {predicate} {obj}")
        
        if len(plan.accumulated_results) > 10:
            summary_lines.append(f"... and {len(plan.accumulated_results) - 10} more results")
        
        return "\n".join(summary_lines)
    
    def _replan_strategy(self, plan: AdaptivePlan, failed_subquery: AdaptiveSubquery):
        """
        Replan strategy based on failed subquery
        
        Args:
            plan: Current plan to modify
            failed_subquery: Subquery that failed
        """
        logger.info(f"Replanning after failed subquery: {failed_subquery.query}")
        
        # Clear remaining subqueries and generate alternative approach
        plan.subqueries.clear()
        
        # Add replanning context
        plan.current_context["last_failure"] = failed_subquery.query
        plan.current_context["replanning"] = True
    
    def _synthesize_final_answer(self, plan: AdaptivePlan) -> str:
        """
        Synthesize final answer from accumulated results
        
        Args:
            plan: Completed plan with all results
            
        Returns:
            Final synthesized answer
        """
        try:
            synthesis_prompt = f"""Based on the following research findings, provide a comprehensive answer to the original query.

Original Query: {plan.original_query}

Research Process:
{chr(10).join(plan.reasoning_chain)}

Total Results Found: {len(plan.accumulated_results)}

Key Findings Summary:
{self._summarize_current_results(plan)}

Please provide a clear, comprehensive answer that:
1. Directly addresses the original query
2. Synthesizes the key findings from the research
3. Acknowledges any limitations or gaps in the results
4. Uses biomedical terminology appropriately

Answer:"""

            messages = [{"role": "system", "content": synthesis_prompt}]
            response = self.llm.invoke(messages)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error synthesizing final answer: {e}")
            return f"Research completed with {len(plan.accumulated_results)} findings. Synthesis error: {str(e)}"
    
    def compare_with_static_approach(self, query: str, entities: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Compare adaptive approach with static query type classification
        
        Args:
            query: Test query
            entities: Optional entities
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing adaptive vs static approaches for: {query[:100]}...")
        
        # Execute adaptive approach
        adaptive_start = time.time()
        adaptive_plan = self.create_adaptive_plan(query, entities)
        adaptive_results = self.execute_adaptive_plan(adaptive_plan)
        adaptive_time = time.time() - adaptive_start
        
        # Simulate static approach (using current query classification)
        static_start = time.time()
        from ..queries.classification import SemanticQueryClassifier
        from ..knowledge.knowledge_system import BiomedicalKnowledgeSystem
        
        try:
            static_system = BiomedicalKnowledgeSystem()
            static_results = static_system.process_biomedical_query(query, max_results=50, k=5)
            static_time = time.time() - static_start
        except Exception as e:
            static_results = {"error": str(e), "results": []}
            static_time = time.time() - static_start
        
        # Compare results
        comparison = {
            "query": query,
            "adaptive_approach": {
                "execution_time": adaptive_time,
                "subqueries_executed": adaptive_results.get("executed_subqueries", 0),
                "total_results": adaptive_results.get("total_results", 0),
                "replanning_count": adaptive_results.get("replanning_count", 0),
                "strategy": adaptive_results.get("strategy", "unknown"),
                "success": adaptive_results.get("success", False),
                "final_answer_length": len(adaptive_results.get("final_answer", ""))
            },
            "static_approach": {
                "execution_time": static_time,
                "total_results": len(static_results.get("results", [])),
                "query_type": static_results.get("query_type", "unknown"),
                "success": "error" not in static_results,
                "classification_confidence": static_results.get("classification", {}).get("confidence", 0.0)
            },
            "comparison_metrics": {
                "adaptive_more_results": (adaptive_results.get("total_results", 0) > 
                                        len(static_results.get("results", []))),
                "adaptive_faster": adaptive_time < static_time,
                "adaptive_more_comprehensive": adaptive_results.get("executed_subqueries", 0) > 1,
                "result_difference": (adaptive_results.get("total_results", 0) - 
                                    len(static_results.get("results", []))),
                "time_difference": adaptive_time - static_time
            }
        }
        
        logger.info(f"Comparison completed - Adaptive: {adaptive_results.get('total_results', 0)} results in {adaptive_time:.2f}s, Static: {len(static_results.get('results', []))} results in {static_time:.2f}s")
        
        return comparison


# Convenience function for direct usage
def optimize_biomedical_query_adaptive(query: str, 
                                      entities: Dict[str, str] = None,
                                      openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for adaptive biomedical query optimization
    
    Args:
        query: Biomedical query to optimize
        entities: Optional pre-extracted entities
        openai_api_key: Optional OpenAI API key
        
    Returns:
        Adaptive optimization results
    """
    optimizer = AdaptiveQueryOptimizer(openai_api_key)
    plan = optimizer.create_adaptive_plan(query, entities)
    return optimizer.execute_adaptive_plan(plan)