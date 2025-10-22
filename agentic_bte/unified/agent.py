"""
Unified Biomedical Agent

This module implements the main unified interface for biomedical query processing.
The UnifiedBiomedicalAgent provides a single entry point that automatically selects
the optimal strategy based on query characteristics, resource availability, and 
performance metrics.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .config import UnifiedConfig
from .types import (
    BiomedicalResult, EntityContext, ExecutionContext, 
    BiomedicalEntity, EntityType
)
from .entity_processor import UnifiedEntityProcessor
from .performance import UnifiedPerformanceMonitor
# [REMOVED] from .strategy_router import UnifiedStrategyRouter, QueryComplexity
from .got_planner import GoTPlanner, GPT41GoTLLM
from .execution_engine import UnifiedExecutionEngine
from .knowledge_manager import UnifiedKnowledgeManager, TRAPIQuery
from .parallel_executor import UnifiedParallelExecutor
from ..core.knowledge.predicate_strategy import QueryIntent
from ..core.queries.types import QueryComplexity


logger = logging.getLogger(__name__)


class QueryMode(Enum):
    """Query execution modes"""
    STANDARD = "standard"          # Normal processing
    FAST = "fast"                 # Prioritize speed over completeness
    COMPREHENSIVE = "comprehensive" # Prioritize completeness over speed
    BALANCED = "balanced"         # Balance speed and completeness
    EXPERIMENTAL = "experimental" # Use experimental features


class ProcessingStage(Enum):
    """Processing stages for tracking"""
    INITIALIZED = "initialized"
    ENTITY_EXTRACTION = "entity_extraction" 
    STRATEGY_SELECTION = "strategy_selection"
    QUERY_BUILDING = "query_building"
    EXECUTION = "execution"
    RESULT_PROCESSING = "result_processing"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueryRequest:
    """Unified query request structure"""
    query_id: str
    text: str
    query_mode: QueryMode = QueryMode.BALANCED
    max_results: int = 100
    timeout_seconds: float = 120.0
    context: Optional[Dict[str, Any]] = None
    preferred_strategies: Optional[List[str]] = None
    exclude_strategies: Optional[List[str]] = None
    enable_parallel: bool = True
    enable_caching: bool = True
    confidence_threshold: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResponse:
    """Unified query response structure"""
    query_id: str
    request: QueryRequest
    results: List[BiomedicalResult]
    total_results: int
    processing_time: float
    strategy_used: str
    stages_completed: List[ProcessingStage]
    performance_metrics: Dict[str, Any]
    knowledge_graph: Optional[Any] = None
    cached: bool = False
    parallel_execution: bool = False
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchQueryRequest:
    """Batch query processing request"""
    batch_id: str
    queries: List[QueryRequest]
    max_concurrency: int = 5
    fail_fast: bool = False
    consolidate_results: bool = True
    timeout_seconds: float = 300.0


@dataclass  
class BatchQueryResponse:
    """Batch query processing response"""
    batch_id: str
    request: BatchQueryRequest
    responses: List[QueryResponse]
    total_processing_time: float
    successful_queries: int
    failed_queries: int
    average_response_time: float
    consolidated_knowledge: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedBiomedicalAgent:
    """
    Main unified interface for biomedical query processing.
    
    This class provides a single entry point that automatically:
    - Extracts and processes entities from natural language queries
    - Selects optimal processing strategies based on query characteristics
    - Executes queries using the best available resources
    - Integrates results into unified knowledge representations
    - Provides comprehensive performance tracking and caching
    """
    
    def __init__(
        self,
        config: Optional[UnifiedConfig] = None,
        enable_caching: bool = True,
        enable_parallel: bool = True
    ):
        """
        Initialize the unified biomedical agent.
        
        Args:
            config: Unified configuration object
            enable_caching: Whether to enable result caching
            enable_parallel: Whether to enable parallel execution
        """
        self.config = config or UnifiedConfig()
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        
        # Initialize core components
        self.entity_processor = UnifiedEntityProcessor(self.config)
        self.performance_tracker = UnifiedPerformanceMonitor(self.config)
# [REMOVED] self.strategy_router = UnifiedStrategyRouter(self.config)
        self.got_planner = GoTPlanner(GPT41GoTLLM())
        # Will set entity_context later after extraction
        self.execution_engine = UnifiedExecutionEngine(self.config)
        self.knowledge_manager = UnifiedKnowledgeManager(self.config)
        
        if self.enable_parallel:
            self.parallel_executor = UnifiedParallelExecutor(self.config)
        else:
            self.parallel_executor = None
        
        # Internal state
        self._active_queries: Dict[str, QueryResponse] = {}
        self._query_history: List[QueryResponse] = []
        self._cache: Dict[str, QueryResponse] = {}
        self._initialized = False
        
        logger.info("UnifiedBiomedicalAgent initialized")
    
    async def initialize(self) -> None:
        """Initialize all components"""
        if self._initialized:
            return
        
        logger.info("Initializing UnifiedBiomedicalAgent components...")
        
        try:
            # Initialize all components
            await self.entity_processor.initialize()
# [REMOVED] await self.strategy_router.initialize()
            await self.execution_engine.initialize()
            await self.knowledge_manager.initialize()
            
            if self.parallel_executor:
                # Parallel executor doesn't need explicit initialization
                pass
            
            self._initialized = True
            logger.info("UnifiedBiomedicalAgent initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize UnifiedBiomedicalAgent: {str(e)}")
            raise
    
    async def process_query(
        self,
        text: str,
        query_mode: QueryMode = QueryMode.BALANCED,
        max_results: int = 100,
        timeout_seconds: float = 120.0,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> QueryResponse:
        """
        Process a single biomedical query.
        
        Args:
            text: Natural language query text
            query_mode: Processing mode (fast, comprehensive, balanced, etc.)
            max_results: Maximum number of results to return
            timeout_seconds: Query timeout in seconds
            context: Additional query context
            **kwargs: Additional parameters
            
        Returns:
            QueryResponse with processed results
        """
        # Ensure initialization
        if not self._initialized:
            await self.initialize()
        
        # Create query request
        request = QueryRequest(
            query_id=str(uuid.uuid4()),
            text=text,
            query_mode=query_mode,
            max_results=max_results,
            timeout_seconds=timeout_seconds,
            context=context or {},
            **{k: v for k, v in kwargs.items() if hasattr(QueryRequest, k)}
        )
        
        return await self._process_single_query(request)
    
    async def process_batch(
        self,
        queries: List[Union[str, QueryRequest]],
        max_concurrency: int = 5,
        fail_fast: bool = False,
        consolidate_results: bool = True,
        timeout_seconds: float = 300.0
    ) -> BatchQueryResponse:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query texts or QueryRequest objects
            max_concurrency: Maximum concurrent query processing
            fail_fast: Whether to stop on first failure
            consolidate_results: Whether to consolidate results into unified knowledge
            timeout_seconds: Total batch timeout
            
        Returns:
            BatchQueryResponse with all results
        """
        if not self._initialized:
            await self.initialize()
        
        # Convert to QueryRequest objects
        query_requests = []
        for query in queries:
            if isinstance(query, str):
                query_requests.append(QueryRequest(
                    query_id=str(uuid.uuid4()),
                    text=query
                ))
            else:
                query_requests.append(query)
        
        # Create batch request
        batch_request = BatchQueryRequest(
            batch_id=str(uuid.uuid4()),
            queries=query_requests,
            max_concurrency=max_concurrency,
            fail_fast=fail_fast,
            consolidate_results=consolidate_results,
            timeout_seconds=timeout_seconds
        )
        
        return await self._process_batch_queries(batch_request)
    
    async def _process_single_query(self, request: QueryRequest) -> QueryResponse:
        """Process a single query request"""
        start_time = time.time()
        stages_completed = [ProcessingStage.INITIALIZED]
        
        # Initialize response
        response = QueryResponse(
            query_id=request.query_id,
            request=request,
            results=[],
            total_results=0,
            processing_time=0.0,
strategy_used="got_framework",  # Was ExecutionStrategy.ENHANCED_GOT
            stages_completed=stages_completed,
            performance_metrics={},
            parallel_execution=self.enable_parallel and request.enable_parallel
        )
        
        # Add to active queries
        self._active_queries[request.query_id] = response
        
        try:
            # Check cache first
            if self.enable_caching and request.enable_caching:
                cache_key = self._get_cache_key(request)
                if cache_key in self._cache:
                    cached_response = self._cache[cache_key]
                    cached_response.cached = True
                    logger.info(f"Returning cached result for query: {request.query_id}")
                    # Guarantee final_answer for cached flows
                    if not hasattr(cached_response, 'final_answer'):
                        print('[DEBUG] No final_answer in cached_response')
                        setattr(cached_response, 'final_answer', '[No answer available]')
                    return cached_response
            
            # Stage 1: Entity extraction and processing
            logger.info(f"Processing query: {request.text[:100]}...")
            
            entities = await self.entity_processor.extract_entities(
                request.text,
                context=EntityContext(**request.context) if request.context else None
            )
            # Always set entity_context on planner so it uses canonical mappings
            self.got_planner.entity_context = EntityContext(entities=entities)
            stages_completed.append(ProcessingStage.ENTITY_EXTRACTION)
            
            # GoT-based decomposition and execution begins here
            print('[DEBUG] Entering GoTPlanner execution block.')
            try:
                got_final_answer = await self.got_planner.execute(request.text)
                print(f'[DEBUG] Got final answer from GoTPlanner: {got_final_answer}')
                response.final_answer = got_final_answer
                # NEW: collect GoT results and save to response
                try:
                    collected = self.got_planner.get_all_results()
                    response.results = collected
                    response.total_results = len(collected)
                except Exception as agg_ex:
                    print(f'[DEBUG] Failed to collect GoT results: {agg_ex}')
            except Exception as got_ex:
                import logging
                print(f'[DEBUG] GoTPlanner error: {got_ex}')
                logging.getLogger(__name__).warning(f"GoTPlanner error: {got_ex}")
                response.final_answer = "[No answer available]"
            stages_completed.append(ProcessingStage.STRATEGY_SELECTION)
            
            # Existing pipeline logic for building queries (skipped under GoT planner)
            
            # Stage 4: Execution
            # Create a simple context dict for execution parameters since the ExecutionContext
            # dataclass expects different parameters
            # The GoTPlanner-only unified pipeline sets response.final_answer and handles all dependency injection. Legacy execution_context, TRAPI, and scoring steps are bypassed in this mode.
            print("[AGENT] Finished GoTPlanner pipeline; intermediate and final GoT reasoning should now be fully logged.")
            
        except Exception as ex:
            logger.error(f"Query processing failed: {str(ex)}")
            response.error = str(ex)
            if not hasattr(response, 'final_answer'):
                print('[DEBUG] No final_answer on QueryResponse in outer error handler!')
                setattr(response, 'final_answer', '[No answer available]')
            stages_completed.append(ProcessingStage.FAILED)
        
        finally:
            # Update response
            response.processing_time = time.time() - start_time
            response.stages_completed = stages_completed
            response.performance_metrics = self.performance_tracker.get_current_metrics()
            
            # Update cache
            if (self.enable_caching and request.enable_caching and 
                response.error is None and response.results):
                cache_key = self._get_cache_key(request)
                self._cache[cache_key] = response
            
            # Update history
            self._query_history.append(response)
            if request.query_id in self._active_queries:
                del self._active_queries[request.query_id]
            
            logger.info(f"Query {request.query_id} completed in {response.processing_time:.3f}s "
                       f"with {len(response.results)} results")
        
        return response
    
    async def _process_batch_queries(self, batch_request: BatchQueryRequest) -> BatchQueryResponse:
        """Process batch queries"""
        start_time = time.time()
        
        logger.info(f"Processing batch of {len(batch_request.queries)} queries")
        
        if self.enable_parallel and self.parallel_executor:
            # Use parallel executor for batch processing
            responses = await self._process_batch_parallel(batch_request)
        else:
            # Sequential processing with concurrency limit
            responses = await self._process_batch_sequential(batch_request)
        
        # Calculate batch metrics
        successful_queries = len([r for r in responses if r.error is None])
        failed_queries = len(responses) - successful_queries
        total_processing_time = time.time() - start_time
        
        response_times = [r.processing_time for r in responses if r.error is None]
        average_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Consolidate knowledge if requested
        consolidated_knowledge = None
        if batch_request.consolidate_results and successful_queries > 0:
            try:
                all_results = []
                all_entities = []
                for response in responses:
                    if response.error is None:
                        all_results.extend(response.results)
                        # Extract entities from original request context if available
                        if response.request.context:
                            # This would need to be implemented based on your specific needs
                            pass
                
                if all_results:
                    knowledge_assertions = await self.knowledge_manager.integrate_results(
                        all_results, all_entities
                    )
                    consolidated_knowledge = await self.knowledge_manager.build_knowledge_graph(
                        knowledge_assertions
                    )
            except Exception as e:
                logger.warning(f"Knowledge consolidation failed: {str(e)}")
        
        return BatchQueryResponse(
            batch_id=batch_request.batch_id,
            request=batch_request,
            responses=responses,
            total_processing_time=total_processing_time,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            average_response_time=average_response_time,
            consolidated_knowledge=consolidated_knowledge
        )
    
    async def _process_batch_parallel(self, batch_request: BatchQueryRequest) -> List[QueryResponse]:
        """Process batch queries using parallel executor"""
        from .parallel_executor import ExecutionTask, ExecutionBatch, ExecutionMode, TaskPriority
        
        # Create execution tasks
        tasks = []
        for i, query_request in enumerate(batch_request.queries):
            task = ExecutionTask(
                task_id=query_request.query_id,
                task_type="biomedical_query",
                task_func=self._process_single_query,
                args=(query_request,),
                priority=TaskPriority.MEDIUM,
                timeout=query_request.timeout_seconds
            )
            tasks.append(task)
        
        # Create execution batch
        execution_batch = ExecutionBatch(
            batch_id=batch_request.batch_id,
            tasks=tasks,
            execution_mode=ExecutionMode.CONCURRENT,
            max_concurrency=batch_request.max_concurrency,
            error_tolerance=0.0 if batch_request.fail_fast else 1.0
        )
        
        # Execute batch
        batch_result = await self.parallel_executor.execute_batch(execution_batch)
        
        # Convert results to QueryResponse objects
        responses = []
        for task_result in batch_result.task_results:
            if task_result.success:
                responses.append(task_result.result)
            else:
                # Create error response
                original_request = next(
                    (req for req in batch_request.queries if req.query_id == task_result.task_id),
                    None
                )
                if original_request:
                    error_response = QueryResponse(
                        query_id=task_result.task_id,
                        request=original_request,
                        results=[],
                        total_results=0,
                        processing_time=task_result.execution_time,
strategy_used="got_framework",
                        stages_completed=[ProcessingStage.FAILED],
                        performance_metrics={},
                        error=str(task_result.error)
                    )
                    responses.append(error_response)
        
        return responses
    
    async def _process_batch_sequential(self, batch_request: BatchQueryRequest) -> List[QueryResponse]:
        """Process batch queries sequentially with concurrency control"""
        semaphore = asyncio.Semaphore(batch_request.max_concurrency)
        
        async def process_with_semaphore(query_request: QueryRequest) -> QueryResponse:
            async with semaphore:
                try:
                    return await self._process_single_query(query_request)
                except Exception as e:
                    # Create error response
                    return QueryResponse(
                        query_id=query_request.query_id,
                        request=query_request,
                        results=[],
                        total_results=0,
                        processing_time=0.0,
strategy_used="got_framework",
                        stages_completed=[ProcessingStage.FAILED],
                        performance_metrics={},
                        error=str(e)
                    )
        
        # Execute all queries
        if batch_request.fail_fast:
            responses = []
            for query_request in batch_request.queries:
                response = await process_with_semaphore(query_request)
                responses.append(response)
                if response.error is not None:
                    break
        else:
            tasks = [process_with_semaphore(query_request) for query_request in batch_request.queries]
            responses = await asyncio.gather(*tasks)
        
        return responses
    
    async def _execute_parallel(
        self,
        trapi_queries: List[TRAPIQuery],
        execution_context: Dict[str, Any]
    ) -> List[BiomedicalResult]:
        """Execute TRAPI queries in parallel"""
        async def execute_query(query: TRAPIQuery) -> List[BiomedicalResult]:
            return await self.execution_engine.execute(
                strategy=execution_context['strategy'],
                queries=[query],
                context=execution_context
            )
        
        # Execute queries in parallel
        results = await self.parallel_executor.execute_parallel_predicates(
            trapi_queries,
            execute_query,
            max_concurrency=self.config.performance.max_concurrent_queries
        )
        
        # Flatten results
        all_results = []
        for query, query_results in results:
            if query_results:
                all_results.extend(query_results)
        
        return all_results
    
    async def _execute_sequential(
        self,
        trapi_queries: List[TRAPIQuery],
        execution_context: Dict[str, Any]
    ) -> List[BiomedicalResult]:
        """Execute TRAPI queries sequentially"""
        all_results = []
        
        for query in trapi_queries:
            try:
                results = await self.execution_engine.execute(
                    strategy=execution_context['strategy'],
                    queries=[query],
                    context=execution_context
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Query execution failed: {str(e)}")
                # Continue with other queries
        
        return all_results
    
    def _assess_query_complexity(
        self,
        request: QueryRequest,
        entities: List[BiomedicalEntity]
    ) -> QueryComplexity:
        """Assess query complexity based on various factors"""
        # Simple heuristic based on query characteristics
        complexity_score = 0
        
        # Text length
        if len(request.text) > 200:
            complexity_score += 2
        elif len(request.text) > 100:
            complexity_score += 1
        
        # Number of entities
        if len(entities) > 5:
            complexity_score += 2
        elif len(entities) > 2:
            complexity_score += 1
        
        # Entity types diversity
        entity_types = set(entity.entity_type for entity in entities)
        if len(entity_types) > 3:
            complexity_score += 2
        elif len(entity_types) > 1:
            complexity_score += 1
        
        # Query mode
        if request.query_mode == QueryMode.COMPREHENSIVE:
            complexity_score += 2
        elif request.query_mode == QueryMode.BALANCED:
            complexity_score += 1
        
        # Max results requested
        if request.max_results > 500:
            complexity_score += 2
        elif request.max_results > 100:
            complexity_score += 1
        
        # Map score to complexity
        if complexity_score >= 6:
            return QueryComplexity.HIGH
        elif complexity_score >= 3:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.LOW
    
    def _get_cache_key(self, request: QueryRequest) -> str:
        """Generate cache key for query request"""
        import hashlib
        
        # Create a string representation of cacheable request properties
        cache_data = {
            'text': request.text.lower().strip(),
'query_mode': str(request.query_mode) if request.query_mode else 'GoT',
            'max_results': request.max_results,
            'confidence_threshold': request.confidence_threshold
        }
        
        # Add context if present and relevant
        if request.context:
            cache_data['context'] = str(sorted(request.context.items()))
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    # Public methods for monitoring and management
    
    def get_active_queries(self) -> Dict[str, QueryResponse]:
        """Get currently active queries"""
        return self._active_queries.copy()
    
    def get_query_history(
        self,
        limit: Optional[int] = None,
        include_errors: bool = True
    ) -> List[QueryResponse]:
        """Get query history"""
        history = self._query_history
        
        if not include_errors:
            history = [r for r in history if r.error is None]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self._query_history:
            return {"message": "No queries processed yet"}
        
        # Basic stats
        total_queries = len(self._query_history)
        successful_queries = len([r for r in self._query_history if r.error is None])
        failed_queries = total_queries - successful_queries
        
        # Timing stats
        processing_times = [r.processing_time for r in self._query_history if r.error is None]
        if processing_times:
            avg_processing_time = sum(processing_times) / len(processing_times)
            min_processing_time = min(processing_times)
            max_processing_time = max(processing_times)
        else:
            avg_processing_time = min_processing_time = max_processing_time = 0.0
        
        # Strategy usage
        strategy_usage = {}
        for response in self._query_history:
            if response.error is None:
                strategy = str(response.strategy_used)
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        # Cache hit rate
        cached_queries = len([r for r in self._query_history if r.cached])
        cache_hit_rate = cached_queries / total_queries if total_queries > 0 else 0.0
        
        # Parallel execution rate
        parallel_queries = len([r for r in self._query_history if r.parallel_execution])
        parallel_execution_rate = parallel_queries / total_queries if total_queries > 0 else 0.0
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": successful_queries / total_queries if total_queries > 0 else 0.0,
            "average_processing_time": avg_processing_time,
            "min_processing_time": min_processing_time,
            "max_processing_time": max_processing_time,
            "cache_hit_rate": cache_hit_rate,
            "parallel_execution_rate": parallel_execution_rate,
            "strategy_usage": strategy_usage,
            "performance_tracker": self.performance_tracker.get_current_metrics()
        }
    
    def clear_cache(self) -> int:
        """Clear query cache and return number of entries cleared"""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {cache_size} cache entries")
        return cache_size
    
    def clear_history(self) -> int:
        """Clear query history and return number of entries cleared"""
        history_size = len(self._query_history)
        self._query_history.clear()
        logger.info(f"Cleared {history_size} history entries")
        return history_size
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "initialized": self._initialized,
            "active_queries": len(self._active_queries),
            "cache_size": len(self._cache),
            "history_size": len(self._query_history),
            "components": {}
        }
        
        # Check component health
        try:
            # Entity processor
            health_status["components"]["entity_processor"] = "healthy"
            
            # Strategy router
            health_status["components"]["strategy_router"] = "healthy"
            
            # Execution engine
            health_status["components"]["execution_engine"] = "healthy"
            
            # Knowledge manager
            health_status["components"]["knowledge_manager"] = "healthy"
            
            # Parallel executor
            if self.parallel_executor:
                health_status["components"]["parallel_executor"] = "healthy"
            
            # Performance tracker
            health_status["components"]["performance_tracker"] = "healthy"
            
        except Exception as e:
            health_status["error"] = str(e)
        
        return health_status
    
    async def shutdown(self) -> None:
        """Shutdown the agent and clean up resources"""
        logger.info("Shutting down UnifiedBiomedicalAgent...")
        
        try:
            # Cancel active queries
            if self._active_queries:
                logger.info(f"Canceling {len(self._active_queries)} active queries")
                self._active_queries.clear()
            
            # Shutdown parallel executor
            if self.parallel_executor:
                self.parallel_executor.shutdown()
            
            # Shutdown other components if they have cleanup methods
            # (Most components don't need explicit shutdown in our current implementation)
            
            self._initialized = False
            logger.info("UnifiedBiomedicalAgent shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            raise