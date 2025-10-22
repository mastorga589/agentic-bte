"""
Unified Optimizer Interface

This module defines the common interfaces and data structures that all
query optimizers must follow to ensure consistency and interoperability.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from .config_manager import OptimizerConfig, get_config_manager
from .performance_monitor import get_performance_monitor


class OptimizationStrategy(Enum):
    """Supported optimization strategies"""
    BASIC_ADAPTIVE = "basic_adaptive"
    META_KG_AWARE = "meta_kg_aware"
    PARALLEL_EXECUTION = "parallel_execution"
    PLACEHOLDER_ENHANCED = "placeholder_enhanced"
    HYBRID_INTELLIGENT = "hybrid_intelligent"


class QueryComplexity(Enum):
    """Query complexity levels for optimizer selection"""
    SIMPLE = "simple"           # 1-2 entities, straightforward relationships
    MODERATE = "moderate"       # 3-4 entities, some complexity
    COMPLEX = "complex"         # 5+ entities, multiple relationships
    RESEARCH = "research"       # Comprehensive exploration needed


@dataclass
class OptimizationMetrics:
    """Standardized performance and quality metrics"""
    execution_time: float = 0.0
    total_results: int = 0
    entities_found: int = 0
    subqueries_executed: int = 0
    api_calls_made: int = 0
    api_calls_saved: int = 0
    placeholders_created: int = 0
    concurrent_batches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    retry_count: int = 0
    quality_score: float = 0.0
    efficiency_factor: float = 1.0
    speedup_factor: float = 1.0


@dataclass
class OptimizationResult:
    """Standardized result format for all optimizers"""
    # Core results
    query: str = ""
    strategy: OptimizationStrategy = OptimizationStrategy.BASIC_ADAPTIVE
    success: bool = False
    results: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    
    # Entity and relationship data
    entities: Dict[str, str] = field(default_factory=dict)  # entity_text -> entity_id
    entity_types: Dict[str, str] = field(default_factory=dict)  # entity_text -> type
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution metadata
    metrics: OptimizationMetrics = field(default_factory=OptimizationMetrics)
    execution_plan: List[str] = field(default_factory=list)  # Subqueries executed
    reasoning_chain: List[str] = field(default_factory=list)  # Planning reasoning
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Timestamps
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def finalize(self):
        """Finalize the result and calculate derived metrics"""
        self.end_time = time.time()
        self.metrics.execution_time = self.end_time - self.start_time
        self.metrics.total_results = len(self.results)
        self.metrics.entities_found = len(self.entities)
        
        # Calculate quality score based on multiple factors
        self._calculate_quality_score()
    
    def _calculate_quality_score(self):
        """Calculate overall quality score (0-1)"""
        factors = []
        
        # Result completeness (0-0.4)
        if self.metrics.total_results > 0:
            result_score = min(0.4, self.metrics.total_results / 100.0 * 0.4)
            factors.append(result_score)
        
        # Entity coverage (0-0.3)
        if self.metrics.entities_found > 0:
            entity_score = min(0.3, self.metrics.entities_found / 20.0 * 0.3)
            factors.append(entity_score)
        
        # Answer quality (0-0.2)
        if len(self.final_answer) > 100:  # Reasonable answer length
            answer_score = 0.2
            factors.append(answer_score)
        
        # Error penalty (0-0.1)
        error_penalty = max(0, 0.1 - (len(self.errors) * 0.02))
        factors.append(error_penalty)
        
        self.metrics.quality_score = sum(factors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API compatibility"""
        return {
            "query": self.query,
            "strategy": self.strategy.value,
            "success": self.success,
            "results": self.results,
            "final_answer": self.final_answer,
            "entities": self.entities,
            "entity_types": self.entity_types,
            "relationships": self.relationships,
            "metrics": {
                "execution_time": self.metrics.execution_time,
                "total_results": self.metrics.total_results,
                "entities_found": self.metrics.entities_found,
                "subqueries_executed": self.metrics.subqueries_executed,
                "api_calls_made": self.metrics.api_calls_made,
                "api_calls_saved": self.metrics.api_calls_saved,
                "placeholders_created": self.metrics.placeholders_created,
                "concurrent_batches": self.metrics.concurrent_batches,
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "error_count": self.metrics.error_count,
                "retry_count": self.metrics.retry_count,
                "quality_score": self.metrics.quality_score,
                "efficiency_factor": self.metrics.efficiency_factor,
                "speedup_factor": self.metrics.speedup_factor
            },
            "execution_plan": self.execution_plan,
            "reasoning_chain": self.reasoning_chain,
            "errors": self.errors,
            "warnings": self.warnings,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


# OptimizerConfig has been moved to config_manager.py


class BaseOptimizer(ABC):
    """Abstract base class that all optimizers must inherit from"""
    
    def __init__(self, config: Optional[OptimizerConfig] = None, openai_api_key: Optional[str] = None, session_id: Optional[str] = None):
        """Initialize the optimizer with configuration"""
        self.config = config  # Will be set after subclass initialization
        self.openai_api_key = openai_api_key
        self.session_id = session_id
        
        # These will be set after subclass initialization
        self._config_manager = get_config_manager()
        self._performance_monitor = get_performance_monitor()
        
        # Initialize metrics tracking
        self._total_queries = 0
        self._successful_queries = 0
        self._total_execution_time = 0.0
    
    def _finalize_config(self):
        """Finalize configuration after subclass initialization"""
        if self.config is None:
            self.config = self._config_manager.get_config(self.get_strategy().value, session_id=self.session_id)
    
    @abstractmethod
    def get_strategy(self) -> OptimizationStrategy:
        """Return the optimization strategy implemented by this optimizer"""
        pass
    
    @abstractmethod
    def optimize_query(self, query: str, entities: Optional[Dict[str, str]] = None) -> OptimizationResult:
        """
        Optimize and execute the given biomedical query
        
        Args:
            query: The biomedical query to optimize and execute
            entities: Optional pre-extracted entities
            
        Returns:
            OptimizationResult with standardized format
        """
        pass
    
    def get_complexity(self, query: str) -> QueryComplexity:
        """
        Analyze query complexity to help with optimizer selection
        
        Args:
            query: Query to analyze
            
        Returns:
            QueryComplexity level
        """
        # Basic heuristics for complexity analysis
        word_count = len(query.split())
        question_words = len([w for w in query.lower().split() if w in ['what', 'how', 'which', 'where', 'when', 'why']])
        conjunctions = len([w for w in query.lower().split() if w in ['and', 'or', 'but', 'by', 'through']])
        
        if word_count <= 8 and question_words <= 1 and conjunctions == 0:
            return QueryComplexity.SIMPLE
        elif word_count <= 15 and question_words <= 2 and conjunctions <= 1:
            return QueryComplexity.MODERATE
        elif word_count <= 25 and conjunctions <= 3:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.RESEARCH
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get aggregated performance statistics"""
        if self._total_queries == 0:
            return {"success_rate": 0.0, "avg_execution_time": 0.0, "total_queries": 0}
        
        return {
            "success_rate": self._successful_queries / self._total_queries,
            "avg_execution_time": self._total_execution_time / self._total_queries,
            "total_queries": self._total_queries
        }
    
    def _update_stats(self, result: OptimizationResult):
        """Update performance statistics"""
        self._total_queries += 1
        if result.success:
            self._successful_queries += 1
        self._total_execution_time += result.metrics.execution_time
        
        # Record in performance monitor
        self._performance_monitor.record_optimization_result(result, session_id=self.session_id)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(strategy={self.get_strategy().value})"


class OptimizerFactory:
    """Factory for creating optimizer instances"""
    
    _optimizers = {}
    
    @classmethod
    def register(cls, strategy: OptimizationStrategy, optimizer_class: type):
        """Register an optimizer class for a strategy"""
        cls._optimizers[strategy] = optimizer_class
    
    @classmethod
    def create(cls, strategy: OptimizationStrategy, config: Optional[OptimizerConfig] = None, 
               openai_api_key: Optional[str] = None, session_id: Optional[str] = None) -> BaseOptimizer:
        """Create an optimizer instance for the given strategy"""
        if strategy not in cls._optimizers:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
        
        # Get config from manager if not provided
        if config is None:
            config_manager = get_config_manager()
            config = config_manager.get_config(strategy.value, session_id=session_id)
        
        optimizer_class = cls._optimizers[strategy]
        return optimizer_class(config=config, openai_api_key=openai_api_key, session_id=session_id)
    
    @classmethod
    def get_available_strategies(cls) -> List[OptimizationStrategy]:
        """Get list of available optimization strategies"""
        return list(cls._optimizers.keys())


# Convenience functions for backward compatibility
def create_optimizer(strategy: Union[str, OptimizationStrategy], 
                    config: Optional[OptimizerConfig] = None,
                    openai_api_key: Optional[str] = None,
                    session_id: Optional[str] = None) -> BaseOptimizer:
    """Create an optimizer instance"""
    if isinstance(strategy, str):
        strategy = OptimizationStrategy(strategy)
    
    return OptimizerFactory.create(strategy, config, openai_api_key, session_id)


def optimize_biomedical_query(query: str, 
                             strategy: Union[str, OptimizationStrategy] = OptimizationStrategy.BASIC_ADAPTIVE,
                             config: Optional[OptimizerConfig] = None,
                             entities: Optional[Dict[str, str]] = None,
                             openai_api_key: Optional[str] = None,
                             session_id: Optional[str] = None,
                             query_id: Optional[str] = None) -> OptimizationResult:
    """
    Convenience function to optimize a biomedical query
    
    Args:
        query: Biomedical query to optimize
        strategy: Optimization strategy to use
        config: Optimizer configuration
        entities: Pre-extracted entities
        openai_api_key: OpenAI API key
        session_id: Optional session identifier
        query_id: Optional query identifier for config tracking
        
    Returns:
        OptimizationResult with standardized format
    """
    # If query_id is provided but no config, get query-specific config
    if query_id and config is None and session_id:
        config_manager = get_config_manager()
        config = config_manager.get_config(
            optimizer_type=strategy.value if isinstance(strategy, OptimizationStrategy) else strategy,
            session_id=session_id,
            query_id=query_id
        )
    
    optimizer = create_optimizer(strategy, config, openai_api_key, session_id)
    return optimizer.optimize_query(query, entities)