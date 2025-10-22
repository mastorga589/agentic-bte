"""
Canonical interfaces and types for all optimization strategies
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time

class OptimizationStrategy(Enum):
    BASIC_ADAPTIVE = "basic_adaptive"
    META_KG_AWARE = "meta_kg_aware"
    PARALLEL_EXECUTION = "parallel_execution"
    PLACEHOLDER_ENHANCED = "placeholder_enhanced"
    HYBRID_INTELLIGENT = "hybrid_intelligent"

@dataclass
class OptimizationMetrics:
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
    query: str = ""
    strategy: OptimizationStrategy = OptimizationStrategy.BASIC_ADAPTIVE
    success: bool = False
    results: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    entities: Dict[str, str] = field(default_factory=dict)
    entity_types: Dict[str, str] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    metrics: OptimizationMetrics = field(default_factory=OptimizationMetrics)
    execution_plan: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def finalize(self):
        self.end_time = time.time()
        self.metrics.execution_time = self.end_time - self.start_time
        self.metrics.total_results = len(self.results)
        self.metrics.entities_found = len(self.entities)
        self._calculate_quality_score()

    def _calculate_quality_score(self):
        factors = []
        if self.metrics.total_results > 0:
            factors.append(min(0.4, self.metrics.total_results / 100.0 * 0.4))
        if self.metrics.entities_found > 0:
            factors.append(min(0.3, self.metrics.entities_found / 20.0 * 0.3))
        if len(self.final_answer) > 100:
            factors.append(0.2)
        error_penalty = max(0, 0.1 - (len(self.errors) * 0.02))
        factors.append(error_penalty)
        self.metrics.quality_score = sum(factors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy.value,
            "success": self.success,
            "results": self.results,
            "final_answer": self.final_answer,
            "entities": self.entities,
            "entity_types": self.entity_types,
            "relationships": self.relationships,
            "metrics": {k: getattr(self.metrics, k) for k in self.metrics.__dataclass_fields__},
            "execution_plan": self.execution_plan,
            "reasoning_chain": self.reasoning_chain,
            "errors": self.errors,
            "warnings": self.warnings,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
