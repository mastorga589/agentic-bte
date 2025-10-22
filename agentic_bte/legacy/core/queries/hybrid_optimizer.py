"""
Hybrid Intelligent Optimizer

This module provides an intelligent optimizer that can automatically select
and combine the strengths of multiple optimization strategies based on query
analysis and dynamic adaptation.
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple

from .interfaces import (
    BaseOptimizer, OptimizationStrategy, OptimizationResult, 
    OptimizationMetrics, OptimizerConfig, OptimizerFactory
)
# Removed heavy complexity analyzer to improve performance
from .caching import cache_get, cache_put
from .simple_working_optimizer import SimpleWorkingOptimizer
from call_mcp_tool import call_mcp_tool
from ...config.settings import get_settings

logger = logging.getLogger(__name__)


class HybridIntelligentOptimizer(BaseOptimizer):
    """
    Hybrid intelligent optimizer that combines multiple strategies
    
    This optimizer analyzes queries to automatically select the best strategy
    or combination of strategies, with fallback mechanisms and adaptive learning.
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None, openai_api_key: Optional[str] = None):
        """Initialize the hybrid intelligent optimizer"""
        super().__init__(config, openai_api_key)
        
        # Initialize settings and API key
        self.settings = get_settings()
        self.openai_api_key = openai_api_key or self.settings.openai_api_key
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Note: Heavy model loading removed to prevent performance issues
        # Query analysis now uses lightweight MCP integration
        
        # Strategy performance tracking
        self.strategy_performance: Dict[OptimizationStrategy, Dict[str, float]] = {}
        for strategy in OptimizationStrategy:
            self.strategy_performance[strategy] = {
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "avg_quality_score": 0.0,
                "total_uses": 0
            }
        
        # Cached optimizer instances
        self.optimizers: Dict[OptimizationStrategy, BaseOptimizer] = {}
        
        # Finalize configuration after initialization
        self._finalize_config()
        
        logger.info("Hybrid intelligent optimizer initialized")
    
    def get_strategy(self) -> OptimizationStrategy:
        """Return the optimization strategy"""
        return OptimizationStrategy.HYBRID_INTELLIGENT
    
    def optimize_query(self, query: str, entities: Optional[Dict[str, str]] = None) -> OptimizationResult:
        """
        Intelligently optimize and execute the biomedical query
        
        Args:
            query: The biomedical query to optimize and execute
            entities: Optional pre-extracted entities
            
        Returns:
            OptimizationResult with standardized format
        """
        result = OptimizationResult(
            query=query,
            strategy=self.get_strategy(),
            start_time=time.time()
        )
        
        try:
            # Check cache first if enabled
            if self.config and self.config.enable_caching:
                cache_params = {
                    "max_results": self.config.max_results,
                    "k": self.config.k,
                    "confidence_threshold": self.config.confidence_threshold
                }
                cached_result = cache_get(query, self.get_strategy().value, cache_params)
                
                if cached_result:
                    result.metrics.cache_hits += 1
                    logger.info(f"Cache hit for hybrid query: {query[:50]}...")
                    
                    # Convert cached result to OptimizationResult
                    self._populate_result_from_cache(result, cached_result)
                    result.finalize()
                    self._update_stats(result)
                    return result
                else:
                    result.metrics.cache_misses += 1
            
            # Use lightweight query analysis via MCP
            logger.info("Analyzing query for strategy selection")
            
            # Extract entities if not provided
            if entities is None:
                try:
                    ner_response = call_mcp_tool("bio_ner", query=query)
                    entities = ner_response.get("entities", {})
                    result.entities = entities
                    result.metrics.entities_found = len(entities)
                except Exception as e:
                    logger.warning(f"Entity extraction failed: {e}")
                    entities = {}
            
            # Simple complexity analysis based on query characteristics
            complexity = self._analyze_query_complexity(query, entities)
            result.reasoning_chain.append(f"Query complexity: {complexity}")
            
            # Select strategy based on complexity and performance history
            selected_strategy, fallback_strategies = self._select_strategy_based_on_complexity(complexity)
            
            # Execute with selected strategy and fallbacks
            success = False
            last_error = None
            
            strategies_to_try = [selected_strategy] + fallback_strategies
            
            for strategy in strategies_to_try:
                try:
                    logger.info(f"Attempting execution with {strategy.value} strategy")
                    result.reasoning_chain.append(f"Trying {strategy.value} strategy")
                    
                    # Execute with selected strategy
                    strategy_result = self._execute_with_strategy(query, strategy, entities)
                    
                    # Update performance tracking
                    self._update_strategy_performance(strategy, strategy_result)
                    
                    if strategy_result.success:
                        # Copy results from strategy execution
                        self._merge_results(result, strategy_result)
                        result.reasoning_chain.append(f"Successfully executed with {strategy.value}")
                        success = True
                        break
                    else:
                        result.warnings.extend(strategy_result.errors)
                        result.reasoning_chain.append(f"{strategy.value} failed: {strategy_result.errors}")
                        
                except Exception as e:
                    last_error = e
                    error_msg = f"{strategy.value} strategy failed: {str(e)}"
                    result.warnings.append(error_msg)
                    result.reasoning_chain.append(error_msg)
                    logger.warning(error_msg)
                    
                    # Update strategy performance for failures
                    self._update_strategy_performance(strategy, None, failed=True)
            
            if not success:
                # All strategies failed - try emergency fallback
                logger.warning("All primary strategies failed, attempting emergency fallback")
                result.reasoning_chain.append("All primary strategies failed, using emergency fallback")
                
                try:
                    fallback_result = self._emergency_fallback(query, entities)
                    if fallback_result.success:
                        self._merge_results(result, fallback_result)
                        result.reasoning_chain.append("Emergency fallback successful")
                        success = True
                    else:
                        result.errors.extend(fallback_result.errors)
                        result.reasoning_chain.append("Emergency fallback also failed")
                except Exception as e:
                    result.errors.append(f"Emergency fallback failed: {str(e)}")
                    result.reasoning_chain.append(f"Emergency fallback error: {str(e)}")
            
            result.success = success
            
            if not success and last_error:
                result.errors.append(f"All optimization strategies failed. Last error: {str(last_error)}")
                result.metrics.error_count += 1
            
            # Cache successful results
            if result.success and self.config and self.config.enable_caching:
                cache_params = {
                    "max_results": self.config.max_results,
                    "k": self.config.k,
                    "confidence_threshold": self.config.confidence_threshold
                }
                cache_put(query, self.get_strategy().value, result.to_dict(), cache_params, self.config.cache_ttl)
                logger.debug("Hybrid result cached successfully")
        
        except Exception as e:
            result.errors.append(f"Unexpected error in hybrid optimizer: {str(e)}")
            result.metrics.error_count += 1
            logger.error(f"Unexpected error in hybrid optimizer: {e}")
        
        finally:
            result.finalize()
            self._update_stats(result)
        
        return result
    
    def _analyze_query_complexity(self, query: str, entities: Dict[str, str]) -> str:
        """Simple query complexity analysis"""
        word_count = len(query.split())
        entity_count = len(entities)
        question_words = len([w for w in query.lower().split() if w in ['what', 'how', 'which', 'where', 'when', 'why']])
        conjunctions = len([w for w in query.lower().split() if w in ['and', 'or', 'but', 'by', 'through']])
        
        if word_count <= 8 and entity_count <= 2 and conjunctions == 0:
            return "simple"
        elif word_count <= 15 and entity_count <= 4 and conjunctions <= 1:
            return "moderate"
        elif word_count <= 25 and conjunctions <= 3:
            return "complex"
        else:
            return "research"
    
    def _select_strategy_based_on_complexity(self, complexity: str) -> Tuple[OptimizationStrategy, List[OptimizationStrategy]]:
        """Select strategy based on complexity analysis"""
        if complexity == "simple":
            primary = OptimizationStrategy.BASIC_ADAPTIVE
            fallbacks = []
        elif complexity == "moderate":
            primary = OptimizationStrategy.META_KG_AWARE  
            fallbacks = [OptimizationStrategy.BASIC_ADAPTIVE]
        elif complexity == "complex":
            primary = OptimizationStrategy.PLACEHOLDER_ENHANCED
            fallbacks = [OptimizationStrategy.META_KG_AWARE, OptimizationStrategy.BASIC_ADAPTIVE]
        else:  # research
            primary = OptimizationStrategy.PARALLEL_EXECUTION
            fallbacks = [OptimizationStrategy.PLACEHOLDER_ENHANCED, OptimizationStrategy.BASIC_ADAPTIVE]
        
        logger.info(f"Selected {primary.value} for {complexity} complexity, fallbacks: {[s.value for s in fallbacks]}")
        return primary, fallbacks
    
    
    def _execute_with_strategy(self, query: str, strategy: OptimizationStrategy, 
                             entities: Optional[Dict[str, str]]) -> OptimizationResult:
        """Execute query with a specific optimization strategy"""
        # Get or create optimizer instance
        if strategy not in self.optimizers:
            # All strategies use the working optimizer for now
            self.optimizers[strategy] = SimpleWorkingOptimizer(
                config=self.config, 
                openai_api_key=self.openai_api_key
            )
            
            if strategy != OptimizationStrategy.BASIC_ADAPTIVE:
                logger.info(f"Using SimpleWorkingOptimizer for {strategy.value} strategy")
        
        optimizer = self.optimizers[strategy]
        return optimizer.optimize_query(query, entities)
    
    def _emergency_fallback(self, query: str, entities: Optional[Dict[str, str]]) -> OptimizationResult:
        """Emergency fallback when all other strategies fail"""
        logger.info("Executing emergency fallback with minimal basic optimizer")
        
        try:
            # Create a minimal basic optimizer with relaxed settings
            fallback_config = OptimizerConfig(
                max_results=10,  # Reduced for faster execution
                k=3,
                confidence_threshold=0.5,  # Lowered threshold
                max_retries=1,  # Reduced retries
                enable_caching=False  # Disable caching for fallback
            )
            
            fallback_optimizer = SimpleWorkingOptimizer(
                config=fallback_config,
                openai_api_key=self.openai_api_key
            )
            
            result = fallback_optimizer.optimize_query(query, entities)
            result.reasoning_chain.append("Emergency fallback executed with relaxed parameters")
            
            return result
            
        except Exception as e:
            # Create minimal failed result
            failed_result = OptimizationResult(
                query=query,
                strategy=OptimizationStrategy.BASIC_ADAPTIVE,
                start_time=time.time()
            )
            failed_result.errors.append(f"Emergency fallback failed: {str(e)}")
            failed_result.finalize()
            
            return failed_result
    
    def _update_strategy_performance(self, strategy: OptimizationStrategy, 
                                   result: Optional[OptimizationResult], failed: bool = False):
        """Update performance tracking for a strategy"""
        perf = self.strategy_performance[strategy]
        
        if failed or (result and not result.success):
            # Record failure
            if perf["total_uses"] > 0:
                # Update success rate
                old_successes = perf["success_rate"] * perf["total_uses"]
                perf["total_uses"] += 1
                perf["success_rate"] = old_successes / perf["total_uses"]
            else:
                perf["total_uses"] = 1
                perf["success_rate"] = 0.0
        
        elif result and result.success:
            # Record success
            old_total = perf["total_uses"]
            old_successes = perf["success_rate"] * old_total if old_total > 0 else 0
            old_time_total = perf["avg_execution_time"] * old_total if old_total > 0 else 0
            old_quality_total = perf["avg_quality_score"] * old_total if old_total > 0 else 0
            
            perf["total_uses"] += 1
            perf["success_rate"] = (old_successes + 1) / perf["total_uses"]
            perf["avg_execution_time"] = (old_time_total + result.metrics.execution_time) / perf["total_uses"]
            perf["avg_quality_score"] = (old_quality_total + result.metrics.quality_score) / perf["total_uses"]
        
        logger.debug(f"Updated {strategy.value} performance: {perf}")
    
    def _populate_result_from_cache(self, result: OptimizationResult, cached_result: Dict[str, Any]):
        """Populate result from cached data"""
        result.success = cached_result.get("success", False)
        result.results = cached_result.get("results", [])
        result.final_answer = cached_result.get("final_answer", "")
        result.entities = cached_result.get("entities", {})
        result.entity_types = cached_result.get("entity_types", {})
        result.relationships = cached_result.get("relationships", [])
        result.execution_plan = cached_result.get("execution_plan", [])
        result.reasoning_chain = cached_result.get("reasoning_chain", [])
        
        # Update metrics from cache
        cached_metrics = cached_result.get("metrics", {})
        result.metrics.subqueries_executed = cached_metrics.get("subqueries_executed", 0)
        result.metrics.api_calls_made = cached_metrics.get("api_calls_made", 0)
    
    def _merge_results(self, target: OptimizationResult, source: OptimizationResult):
        """Merge results from strategy execution into target result"""
        target.success = source.success
        target.results = source.results
        target.final_answer = source.final_answer
        target.relationships = source.relationships
        target.execution_plan.extend(source.execution_plan)
        
        # Merge entities (target takes precedence for conflicts)
        for entity, entity_id in source.entities.items():
            if entity not in target.entities:
                target.entities[entity] = entity_id
        
        for entity, entity_type in source.entity_types.items():
            if entity not in target.entity_types:
                target.entity_types[entity] = entity_type
        
        # Merge metrics
        target.metrics.subqueries_executed += source.metrics.subqueries_executed
        target.metrics.api_calls_made += source.metrics.api_calls_made
        target.metrics.api_calls_saved += source.metrics.api_calls_saved
        target.metrics.placeholders_created += source.metrics.placeholders_created
        target.metrics.concurrent_batches += source.metrics.concurrent_batches
        target.metrics.cache_hits += source.metrics.cache_hits
        target.metrics.cache_misses += source.metrics.cache_misses
        target.metrics.retry_count += source.metrics.retry_count
        
        # Update entity count
        target.metrics.entities_found = len(target.entities)
    
    def get_strategy_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all strategies"""
        return {
            strategy.value: perf.copy() 
            for strategy, perf in self.strategy_performance.items()
        }
    
    def reset_strategy_performance(self):
        """Reset strategy performance tracking"""
        for strategy in self.strategy_performance:
            self.strategy_performance[strategy] = {
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "avg_quality_score": 0.0,
                "total_uses": 0
            }
        logger.info("Strategy performance tracking reset")


# Register with factory
OptimizerFactory.register(OptimizationStrategy.HYBRID_INTELLIGENT, HybridIntelligentOptimizer)