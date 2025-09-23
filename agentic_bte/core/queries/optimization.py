"""
Query Optimization - Advanced biomedical query decomposition and optimization

This module provides sophisticated query planning and optimization strategies
for complex biomedical research queries, including:
- Query decomposition into manageable subqueries
- Dependency analysis and parallel execution opportunities
- Dynamic replanning based on intermediate results
- Result aggregation and synthesis

Migrated and enhanced from the original BTE-LLM implementation.
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from langchain_openai import ChatOpenAI

from ..entities.bio_ner import BioNERTool
from ..knowledge.trapi import TRAPIQueryBuilder
from ..knowledge.bte_client import BTEClient
from ...config.settings import get_settings
from ...exceptions.base import ExternalServiceError

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels for optimization strategy selection"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class SubqueryStatus(Enum):
    """Status of individual subqueries in the execution plan"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubqueryNode:
    """Represents a subquery in the optimization plan"""
    id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    status: SubqueryStatus = SubqueryStatus.PENDING
    dependencies: Set[str] = field(default_factory=set)
    priority: int = 1
    estimated_cost: float = 1.0
    actual_cost: float = 0.0
    results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class OptimizationPlan:
    """Complete optimization plan for a complex query"""
    id: str = field(default_factory=lambda: str(uuid4()))
    original_query: str = ""
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    subqueries: List[SubqueryNode] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    total_estimated_cost: float = 0.0
    created_at: float = field(default_factory=time.time)
    optimization_metadata: Dict[str, Any] = field(default_factory=dict)


class QueryOptimizer:
    """
    Advanced biomedical query optimizer
    
    Analyzes complex queries and creates optimized execution plans with
    parallel processing opportunities and dependency management.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the query optimizer
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
        """
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for query optimization")
        
        # Initialize components
        self.llm = ChatOpenAI(
            temperature=0,
            model=self.settings.openai_model,
            api_key=self.openai_api_key
        )
        self.bio_ner = BioNERTool(openai_api_key)
        self.trapi_builder = TRAPIQueryBuilder(openai_api_key)
        self.bte_client = BTEClient()
        
        # Optimization cache
        self._plan_cache: Dict[str, OptimizationPlan] = {}
        
    def analyze_query_complexity(self, query: str, entities: Dict[str, str] = None) -> QueryComplexity:
        """
        Analyze the complexity of a biomedical query
        
        Args:
            query: Natural language query to analyze
            entities: Optional pre-extracted entities
            
        Returns:
            Query complexity level
        """
        try:
            # Extract entities if not provided
            if entities is None:
                entity_result = self.bio_ner.extract_and_link(query)
                entities = entity_result.get("entity_ids", {})
            
            # Complexity factors
            entity_count = len(entities)
            query_length = len(query.split())
            
            # Check for complexity indicators
            complexity_indicators = [
                "mechanism", "pathway", "interaction", "cascade",
                "upstream", "downstream", "regulate", "modulate",
                "cause", "effect", "target", "biomarker"
            ]
            
            indicator_matches = sum(1 for indicator in complexity_indicators 
                                  if indicator.lower() in query.lower())
            
            # Determine complexity
            if entity_count <= 2 and query_length <= 20 and indicator_matches <= 1:
                return QueryComplexity.SIMPLE
            elif entity_count <= 4 and query_length <= 40 and indicator_matches <= 3:
                return QueryComplexity.MODERATE
            elif entity_count <= 8 and query_length <= 80 and indicator_matches <= 6:
                return QueryComplexity.COMPLEX
            else:
                return QueryComplexity.VERY_COMPLEX
                
        except Exception as e:
            logger.warning(f"Failed to analyze query complexity: {e}")
            return QueryComplexity.MODERATE  # Default fallback
    
    def create_optimization_plan(self, 
                               query: str, 
                               entities: Dict[str, str] = None,
                               max_subqueries: int = 10,
                               enable_dynamic_optimization: bool = True) -> OptimizationPlan:
        """
        Create an optimized execution plan for a complex query
        
        Args:
            query: Natural language biomedical query
            entities: Optional pre-extracted entities
            max_subqueries: Maximum number of subqueries to generate
            enable_dynamic_optimization: Enable dynamic replanning
            
        Returns:
            Optimization plan with decomposed subqueries
        """
        logger.info(f"Creating optimization plan for query: {query[:100]}...")
        
        try:
            # Analyze complexity
            complexity = self.analyze_query_complexity(query, entities)
            logger.info(f"Query complexity assessed as: {complexity.value}")
            
            # Create base plan
            plan = OptimizationPlan(
                original_query=query,
                complexity=complexity,
                optimization_metadata={
                    "enable_dynamic_optimization": enable_dynamic_optimization,
                    "max_subqueries": max_subqueries,
                    "creation_strategy": "llm_decomposition"
                }
            )
            
            # Extract entities if needed
            if entities is None:
                entity_result = self.bio_ner.extract_and_link(query)
                entities = entity_result.get("entity_ids", {})
            
            # Generate subqueries based on complexity
            if complexity == QueryComplexity.SIMPLE:
                subqueries = [query]  # No decomposition needed
            else:
                subqueries = self._decompose_query_with_llm(query, entities, max_subqueries)
            
            # Create subquery nodes
            for i, subquery in enumerate(subqueries):
                node = SubqueryNode(
                    query=subquery.strip(),
                    priority=len(subqueries) - i,  # Earlier queries have higher priority
                    estimated_cost=self._estimate_subquery_cost(subquery, entities)
                )
                plan.subqueries.append(node)
            
            # Analyze dependencies and create execution order
            self._analyze_dependencies(plan)
            self._create_execution_order(plan)
            
            # Calculate total estimated cost
            plan.total_estimated_cost = sum(node.estimated_cost for node in plan.subqueries)
            
            # Cache the plan
            self._plan_cache[plan.id] = plan
            
            logger.info(f"Created optimization plan with {len(plan.subqueries)} subqueries")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create optimization plan: {e}")
            # Return simple fallback plan
            return OptimizationPlan(
                original_query=query,
                complexity=QueryComplexity.SIMPLE,
                subqueries=[SubqueryNode(query=query)],
                execution_order=[str(uuid4())],
                optimization_metadata={"error": str(e), "fallback": True}
            )
    
    def _decompose_query_with_llm(self, 
                                query: str, 
                                entities: Dict[str, str],
                                max_subqueries: int) -> List[str]:
        """
        Use LLM to decompose complex query into simpler subqueries
        
        Args:
            query: Original complex query
            entities: Extracted entities
            max_subqueries: Maximum subqueries to generate
            
        Returns:
            List of subquery strings
        """
        try:
            entity_list = list(entities.keys())[:10]  # Limit for prompt size
            
            decomposition_prompt = f"""
            You are an expert biomedical researcher tasked with breaking down complex research questions 
            into simpler, manageable subqueries that can be answered using a biomedical knowledge graph.
            
            Original complex query: "{query}"
            
            Key entities identified: {entity_list}
            
            Please decompose this query into {min(max_subqueries, 8)} or fewer simpler subqueries that:
            1. Can be answered individually using biomedical knowledge graphs
            2. Build upon each other logically (early subqueries inform later ones)
            3. Use a mechanistic approach (e.g., drug -> target -> pathway -> disease)
            4. Focus on single-hop relationships between entities
            
            Guidelines:
            - Start with broad relationships, then get more specific
            - Each subquery should be a complete, standalone question
            - Avoid complex multi-part questions in individual subqueries
            - Use natural language (not technical query syntax)
            
            Return ONLY a JSON list of subquery strings, like:
            ["subquery 1", "subquery 2", "subquery 3"]
            
            Do not include any other text or explanation.
            """
            
            response = self.llm.invoke(decomposition_prompt)
            result = response.content.strip()
            
            # Parse the JSON response
            import json
            try:
                subqueries = json.loads(result)
                if isinstance(subqueries, list) and all(isinstance(sq, str) for sq in subqueries):
                    # Filter out empty or very short subqueries
                    filtered_subqueries = [sq for sq in subqueries if len(sq.strip()) > 10]
                    return filtered_subqueries[:max_subqueries]
                else:
                    raise ValueError("Invalid subquery format")
            except json.JSONDecodeError:
                # Fallback: try to extract from text
                logger.warning("Failed to parse JSON, attempting text extraction")
                return self._extract_subqueries_from_text(result, max_subqueries)
                
        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}")
            # Fallback to simple decomposition
            return [query]
    
    def _extract_subqueries_from_text(self, text: str, max_subqueries: int) -> List[str]:
        """
        Extract subqueries from text when JSON parsing fails
        
        Args:
            text: Response text from LLM
            max_subqueries: Maximum subqueries to extract
            
        Returns:
            List of subquery strings
        """
        import re
        
        # Look for numbered lists or quoted strings
        patterns = [
            r'\d+\.\s*"([^"]+)"',  # 1. "query"
            r'\d+\.\s*([^?\n]+\?)',  # 1. query?
            r'"([^"]{20,})"',  # "longer quoted strings"
            r'([A-Z][^?\n]{20,}\?)',  # Sentence-like queries
        ]
        
        subqueries = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            subqueries.extend(matches)
            if len(subqueries) >= max_subqueries:
                break
        
        return subqueries[:max_subqueries] if subqueries else [text]
    
    def _estimate_subquery_cost(self, subquery: str, entities: Dict[str, str]) -> float:
        """
        Estimate the computational cost of a subquery
        
        Args:
            subquery: Subquery string
            entities: Available entities
            
        Returns:
            Estimated cost (higher = more expensive)
        """
        # Simple heuristic-based cost estimation
        base_cost = 1.0
        
        # Length factor
        length_factor = len(subquery.split()) / 10.0
        
        # Entity factor (more entities = potentially more expensive)
        entity_factor = min(len(entities) / 5.0, 2.0)
        
        # Complexity keywords
        complex_keywords = ["mechanism", "pathway", "interaction", "regulate", "modulate"]
        complexity_factor = sum(1 for kw in complex_keywords if kw.lower() in subquery.lower()) * 0.5
        
        return base_cost + length_factor + entity_factor + complexity_factor
    
    def _analyze_dependencies(self, plan: OptimizationPlan):
        """
        Analyze dependencies between subqueries
        
        Args:
            plan: Optimization plan to analyze
        """
        # Simple dependency analysis based on query content and order
        # In a more sophisticated implementation, this would use semantic analysis
        
        for i, current_node in enumerate(plan.subqueries):
            # Later queries may depend on earlier ones
            for j in range(i):
                prev_node = plan.subqueries[j]
                if self._queries_have_dependency(prev_node.query, current_node.query):
                    current_node.dependencies.add(prev_node.id)
    
    def _queries_have_dependency(self, query1: str, query2: str) -> bool:
        """
        Simple heuristic to check if query2 depends on query1
        
        Args:
            query1: First query
            query2: Second query
            
        Returns:
            True if query2 likely depends on query1
        """
        # Check for pronouns and references in query2
        dependency_indicators = ["these", "those", "they", "them", "which", "that"]
        return any(indicator in query2.lower() for indicator in dependency_indicators)
    
    def _create_execution_order(self, plan: OptimizationPlan):
        """
        Create execution order considering dependencies and parallelization
        
        Args:
            plan: Optimization plan to order
        """
        # Topological sort for dependency-aware ordering
        remaining_nodes = {node.id: node for node in plan.subqueries}
        execution_order = []
        parallel_groups = []
        
        while remaining_nodes:
            # Find nodes with no unresolved dependencies
            ready_nodes = []
            for node_id, node in remaining_nodes.items():
                unresolved_deps = node.dependencies - set(execution_order)
                if not unresolved_deps:
                    ready_nodes.append((node_id, node))
            
            if not ready_nodes:
                # Handle circular dependencies by picking highest priority
                ready_nodes = [max(remaining_nodes.items(), key=lambda x: x[1].priority)]
                logger.warning("Circular dependency detected, breaking with highest priority node")
            
            # Sort by priority
            ready_nodes.sort(key=lambda x: x[1].priority, reverse=True)
            
            # Create parallel group from ready nodes
            parallel_group = [node_id for node_id, _ in ready_nodes]
            parallel_groups.append(parallel_group)
            
            # Add to execution order and remove from remaining
            for node_id, _ in ready_nodes:
                execution_order.append(node_id)
                del remaining_nodes[node_id]
        
        plan.execution_order = execution_order
        plan.parallel_groups = parallel_groups
    
    def get_plan_by_id(self, plan_id: str) -> Optional[OptimizationPlan]:
        """
        Retrieve a cached optimization plan by ID
        
        Args:
            plan_id: Plan identifier
            
        Returns:
            Optimization plan if found, None otherwise
        """
        return self._plan_cache.get(plan_id)
    
    def clear_plan_cache(self):
        """Clear the optimization plan cache"""
        self._plan_cache.clear()
    
    def get_plan_summary(self, plan: OptimizationPlan) -> Dict[str, Any]:
        """
        Get a summary of an optimization plan
        
        Args:
            plan: Optimization plan
            
        Returns:
            Plan summary information
        """
        return {
            "plan_id": plan.id,
            "original_query": plan.original_query,
            "complexity": plan.complexity.value,
            "total_subqueries": len(plan.subqueries),
            "parallel_groups": len(plan.parallel_groups),
            "estimated_total_cost": plan.total_estimated_cost,
            "subqueries": [
                {
                    "id": node.id,
                    "query": node.query,
                    "status": node.status.value,
                    "priority": node.priority,
                    "dependencies": list(node.dependencies),
                    "estimated_cost": node.estimated_cost
                }
                for node in plan.subqueries
            ],
            "execution_metadata": plan.optimization_metadata
        }


# Convenience functions
def create_optimized_plan(query: str, entities: Dict[str, str] = None, 
                         openai_api_key: Optional[str] = None) -> OptimizationPlan:
    """
    Convenience function to create an optimization plan
    
    Args:
        query: Natural language query
        entities: Optional pre-extracted entities
        openai_api_key: Optional OpenAI API key
        
    Returns:
        Optimization plan
    """
    optimizer = QueryOptimizer(openai_api_key)
    return optimizer.create_optimization_plan(query, entities)


def analyze_complexity(query: str, entities: Dict[str, str] = None,
                      openai_api_key: Optional[str] = None) -> QueryComplexity:
    """
    Convenience function to analyze query complexity
    
    Args:
        query: Natural language query
        entities: Optional pre-extracted entities
        openai_api_key: Optional OpenAI API key
        
    Returns:
        Query complexity level
    """
    optimizer = QueryOptimizer(openai_api_key)
    return optimizer.analyze_query_complexity(query, entities)