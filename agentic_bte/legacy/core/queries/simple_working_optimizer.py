"""
Simple Working Optimizer

A lightweight, functional optimizer that actually works with real MCP integration
and shared model loading. This is designed to be fast and reliable.
"""

import logging
import time
import sys
import os
from typing import Dict, List, Any, Optional

# Add the project root to Python path
sys.path.append('/Users/mastorga/Documents/agentic-bte')

from .interfaces import (
    BaseOptimizer, OptimizationStrategy, OptimizationResult, 
    OptimizationMetrics, OptimizerConfig
)

# Import the real MCP tool function
from call_mcp_tool import call_mcp_tool

logger = logging.getLogger(__name__)

class SimpleWorkingOptimizer(BaseOptimizer):
    """
    A simple, working optimizer that focuses on functionality over complexity
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None, openai_api_key: Optional[str] = None):
        """Initialize the simple optimizer"""
        super().__init__(config, openai_api_key)
        
        # Set up basic configuration
        self.openai_api_key = openai_api_key or "test_key"
        
        # Finalize configuration
        self._finalize_config()
        
        logger.info("Simple working optimizer initialized")
    
    def get_strategy(self) -> OptimizationStrategy:
        """Return the optimization strategy"""
        return OptimizationStrategy.BASIC_ADAPTIVE
    
    def optimize_query(self, query: str, entities: Optional[Dict[str, str]] = None) -> OptimizationResult:
        """
        Execute a biomedical query using the working pipeline
        
        Args:
            query: The biomedical query to execute
            entities: Optional pre-extracted entities
            
        Returns:
            OptimizationResult with execution details
        """
        start_time = time.time()
        
        result = OptimizationResult(
            query=query,
            strategy=self.get_strategy(),
            start_time=start_time
        )
        
        try:
            logger.info(f"Executing query: {query}")
            result.reasoning_chain.append(f"Starting execution of: {query}")
            
            # Step 1: Entity extraction using MCP
            if entities is None:
                logger.info("Extracting entities using MCP bio_ner tool")
                result.reasoning_chain.append("Extracting biomedical entities")
                
                try:
                    ner_response = call_mcp_tool("bio_ner", query=query)
                    entities = ner_response.get("entities", {})
                    
                    result.entities = entities
                    result.metrics.entities_found = len(entities)
                    
                    entity_list = list(entities.keys())
                    result.reasoning_chain.append(f"Found {len(entities)} entities: {entity_list}")
                    logger.info(f"Extracted entities: {entity_list}")
                    
                except Exception as e:
                    error_msg = f"Entity extraction failed: {str(e)}"
                    result.warnings.append(error_msg)
                    result.reasoning_chain.append(error_msg)
                    logger.warning(error_msg)
                    entities = {}
            
            # Step 2: Build TRAPI query using MCP
            logger.info("Building TRAPI query")
            result.reasoning_chain.append("Building TRAPI query from entities")
            
            try:
                trapi_response = call_mcp_tool(
                    "build_trapi_query",
                    query=query,
                    entity_data=entities
                )
                
                trapi_query = trapi_response.get("query", {})
                result.execution_plan.append("TRAPI query built successfully")
                result.reasoning_chain.append(f"TRAPI query with {len(trapi_query.get('nodes', {}))} nodes")
                
            except Exception as e:
                error_msg = f"TRAPI query building failed: {str(e)}"
                result.errors.append(error_msg)
                result.reasoning_chain.append(error_msg)
                logger.error(error_msg)
                result.finalize()
                return result
            
            # Step 3: Execute BTE API call using MCP
            logger.info("Calling BTE API")
            result.reasoning_chain.append("Querying BioThings Explorer API")
            
            try:
                k = self.config.k if self.config else 5
                max_results = self.config.max_results if self.config else 50
                
                bte_response = call_mcp_tool(
                    "call_bte_api",
                    json_query=trapi_query,
                    k=k,
                    maxresults=max_results
                )
                
                # Extract results
                message = bte_response.get("message", {})
                api_results = message.get("results", [])
                
                result.results = api_results
                result.metrics.api_calls_made = 1
                result.metrics.total_results = len(api_results)
                
                result.reasoning_chain.append(f"BTE API returned {len(api_results)} results")
                logger.info(f"BTE API returned {len(api_results)} results")
                
                # Generate final answer
                if api_results:
                    result.final_answer = f"Found {len(api_results)} results for query '{query}'. "
                    if entities:
                        entity_names = list(entities.keys())
                        result.final_answer += f"Key entities identified: {', '.join(entity_names)}. "
                    
                    result.final_answer += f"Top result scored {api_results[0].get('analyses', [{}])[0].get('score', 0):.2f}."
                    
                    result.success = True
                    result.reasoning_chain.append("Query executed successfully")
                else:
                    result.final_answer = f"No results found for query '{query}'."
                    result.success = False
                    result.reasoning_chain.append("No results returned from BTE API")
                
            except Exception as e:
                error_msg = f"BTE API call failed: {str(e)}"
                result.errors.append(error_msg)
                result.reasoning_chain.append(error_msg)
                logger.error(error_msg)
                result.success = False
        
        except Exception as e:
            error_msg = f"Unexpected error during query execution: {str(e)}"
            result.errors.append(error_msg)
            result.reasoning_chain.append(error_msg)
            logger.error(error_msg)
            result.success = False
        
        finally:
            result.finalize()
            self._update_stats(result)
            
            execution_time = time.time() - start_time
            logger.info(f"Query execution completed in {execution_time:.2f}s, success: {result.success}")
        
        return result
    
    def _update_stats(self, result: OptimizationResult):
        """Update optimizer statistics"""
        self._total_queries += 1
        if result.success:
            self._successful_queries += 1
        self._total_execution_time += result.metrics.execution_time
        
        # Update performance monitor if available
        if hasattr(self, '_performance_monitor') and self._performance_monitor:
            try:
                self._performance_monitor.record_optimization_result(
                    optimizer_type=self.get_strategy().value,
                    success=result.success,
                    execution_time=result.metrics.execution_time,
                    result_count=len(result.results),
                    entity_count=len(result.entities)
                )
            except Exception as e:
                logger.debug(f"Performance monitoring update failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        success_rate = (
            self._successful_queries / self._total_queries 
            if self._total_queries > 0 else 0.0
        )
        avg_time = (
            self._total_execution_time / self._total_queries 
            if self._total_queries > 0 else 0.0
        )
        
        return {
            "total_queries": self._total_queries,
            "successful_queries": self._successful_queries,
            "success_rate": success_rate,
            "average_execution_time": avg_time
        }