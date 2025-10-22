"""
Unified System Module

This module contains the unified system implementation that consolidates
all features from different optimizers and strategies into a cohesive system.
"""

from .config import (
    UnifiedConfig,
    CacheBackend,
    LogLevel,
    PerformanceConfig,
    QualityConfig,
    CachingConfig,
    DomainConfig,
    IntegrationConfig,
    DebugConfig,
    create_development_config,
    create_production_config,
    create_testing_config
)

from .types import (
    BiomedicalEntity,
    EntityContext,
    BiomedicalRelationship,
    KnowledgeGraph,
    ExecutionStep,
    ExecutionContext,
    PerformanceMetrics,
    BiomedicalResult,
    EntityType,
    ConfidenceLevel,
    ExecutionStatus,
    create_error_result,
    merge_results
)

from .entity_processor import (
    UnifiedEntityProcessor,
    GenericEntityResolver,
    PlaceholderSystem
)

from .performance import (
    UnifiedPerformanceMonitor,
    ResourceMonitor,
    PerformanceProfiler,
    CacheMetricsCollector,
    StrategyPerformanceTracker
)


from .execution_engine import (
    UnifiedExecutionEngine,
    UnifiedCache,
    ErrorHandler,
    ExecutionTimeout,
    ExecutionPlan
)

from .knowledge_manager import (
    UnifiedKnowledgeManager,
    TRAPIQuery,
    KnowledgeAssertion,
    KnowledgeEvidence,
    KnowledgeSource,
    PredicateRanking
)

from .parallel_executor import (
    UnifiedParallelExecutor,
    ExecutionTask,
    ExecutionBatch,
    ExecutionMode,
    TaskPriority,
    TaskResult,
    BatchResult
)

from .agent import (
    UnifiedBiomedicalAgent,
    QueryRequest,
    QueryResponse,
    BatchQueryRequest,
    BatchQueryResponse,
    QueryMode,
    ProcessingStage
)

__all__ = [
    # Configuration
    "UnifiedConfig",
    "CacheBackend",
    "LogLevel",
    "PerformanceConfig",
    "QualityConfig", 
    "CachingConfig",
    "DomainConfig",
    "IntegrationConfig",
    "DebugConfig",
    "create_development_config",
    "create_production_config",
    "create_testing_config",
    
    # Data Types
    "BiomedicalEntity",
    "EntityContext",
    "BiomedicalRelationship", 
    "KnowledgeGraph",
    "ExecutionStep",
    "ExecutionContext",
    "PerformanceMetrics",
    "BiomedicalResult",
    "EntityType",
    "ConfidenceLevel",
    "ExecutionStatus",
    "create_error_result",
    "merge_results",
    
    # Entity Processing
    "UnifiedEntityProcessor",
    "GenericEntityResolver",
    "PlaceholderSystem",
    
    # Performance Monitoring
    "UnifiedPerformanceMonitor",
    "ResourceMonitor",
    "PerformanceProfiler",
    "CacheMetricsCollector",
    "StrategyPerformanceTracker",
    
    # Strategy Router
    "UnifiedStrategyRouter",
    "QueryComplexityAnalyzer",
    "ResourceConstraintAnalyzer",
    "StrategyRecommendation",
    
    # Execution Engine
    "UnifiedExecutionEngine",
    "UnifiedCache",
    "ErrorHandler",
    "ExecutionTimeout",
    "ExecutionPlan",
    
    # Knowledge Manager
    "UnifiedKnowledgeManager",
    "TRAPIQuery",
    "KnowledgeAssertion",
    "KnowledgeEvidence",
    "KnowledgeSource",
    "PredicateRanking",
    
    # Parallel Executor
    "UnifiedParallelExecutor",
    "ExecutionTask",
    "ExecutionBatch", 
    "ExecutionMode",
    "TaskPriority",
    "TaskResult",
    "BatchResult",
    
    # Unified Agent
    "UnifiedBiomedicalAgent",
    "QueryRequest",
    "QueryResponse",
    "BatchQueryRequest",
    "BatchQueryResponse",
    "QueryMode",
    "ProcessingStage"
]
