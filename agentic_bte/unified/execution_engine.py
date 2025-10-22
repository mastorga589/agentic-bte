"""
Unified Execution Engine

This module provides a comprehensive execution engine that can run any strategy
with unified caching, monitoring, error handling, and parallel processing
capabilities.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager

from .config import UnifiedConfig
from .types import (
    BiomedicalResult, EntityContext, ExecutionContext, ExecutionStep,
    ExecutionStatus, PerformanceMetrics, create_error_result
)
from .entity_processor import UnifiedEntityProcessor
from .performance import UnifiedPerformanceMonitor
from ..core.queries.interfaces import OptimizationResult

# Removed UnifiedStrategyRouter

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Execution plan with strategy and configuration"""
    primary_strategy: str
    fallback_strategies: list[str]
    entity_context: EntityContext
    execution_context: ExecutionContext
    configuration: Dict[str, Any]
    estimated_resources: Dict[str, Any]




class UnifiedCache:
    """Unified caching system for all strategies"""
    
    def __init__(self, config: UnifiedConfig):
        logger.debug(f"Initializing UnifiedCache with config: {config}")
        self.config = config
        self.memory_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize backend-specific caching if configured
        self.redis_client = None
        if config.caching.backend == "redis":
            logger.debug("Redis caching backend configured - initializing Redis client")
            self._initialize_redis()
        else:
            logger.debug(f"Using caching backend: {config.caching.backend}")
        
        logger.info(f"UnifiedCache initialized with backend: {config.caching.backend}")
    
    def _initialize_redis(self):
        """Initialize Redis client if configured"""
        logger.debug("Attempting Redis initialization...")
        try:
            import redis
            logger.debug(f"Connecting to Redis at {self.config.caching.redis_host}:{self.config.caching.redis_port}")
            self.redis_client = redis.Redis(
                host=self.config.caching.redis_host,
                port=self.config.caching.redis_port,
                db=self.config.caching.redis_db,
                password=self.config.caching.redis_password,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except ImportError as e:
            logger.error(f"Redis library not available: {e}")
            self.redis_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")
            logger.debug(f"Full Redis init error: {str(e)}", exc_info=True)
            self.redis_client = None

    def _generate_cache_key(self, query: str, strategy: str, config_hash: str) -> str:
        """Generate cache key"""
        import hashlib
        key_data = f"{query}:{strategy}:{config_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def get(self, query: str, strategy: str, config_hash: str) -> Optional[BiomedicalResult]:
        """Get cached result"""
        if not self.config.caching.enable_caching:
            logger.debug("Caching disabled - returning None")
            return None
        
        cache_key = self._generate_cache_key(query, strategy, config_hash)
        logger.debug(f"Cache lookup for key: {cache_key} (strategy: {strategy})")
            
        try:
            # Check memory cache first
            if cache_key in self.memory_cache:
                timestamp = self.cache_timestamps.get(cache_key, 0)
                cache_age = time.time() - timestamp
                logger.debug(f"Found in memory cache, age: {cache_age:.2f}s, TTL: {self.config.caching.cache_ttl}s")
                
                if cache_age < self.config.caching.cache_ttl:
                    self.cache_hits += 1
                    logger.debug(f"Memory cache hit for key: {cache_key}")
                    return self.memory_cache[cache_key]
                else:
                    # Expired
                    logger.debug(f"Memory cache entry expired, removing key: {cache_key}")
                    del self.memory_cache[cache_key]
                    del self.cache_timestamps[cache_key]
            
            # Check Redis cache if available
            if self.redis_client:
                logger.debug("Checking Redis cache...")
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    logger.debug(f"Found data in Redis cache, size: {len(cached_data)} bytes")
                    import json
                    result_data = json.loads(cached_data)
                    result = BiomedicalResult.from_dict(result_data)
                    
                    # Store in memory cache for faster access
                    self.memory_cache[cache_key] = result
                    self.cache_timestamps[cache_key] = time.time()
                    
                    self.cache_hits += 1
                    logger.debug(f"Redis cache hit for key: {cache_key}")
                    return result
                else:
                    logger.debug("No data found in Redis cache")
            else:
                logger.debug("Redis client not available")
            
            self.cache_misses += 1
            logger.debug(f"Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {cache_key}: {e}")
            logger.debug(f"Full cache get error: {str(e)}", exc_info=True)
            return None
    
    async def put(self, query: str, strategy: str, config_hash: str, result: BiomedicalResult):
        """Store result in cache"""
        if not self.config.caching.enable_caching:
            logger.debug("Caching disabled - not storing result")
            return
        
        if not result.success:
            logger.debug("Result unsuccessful - not caching")
            return
        
        cache_key = self._generate_cache_key(query, strategy, config_hash)
        logger.debug(f"Storing result in cache for key: {cache_key} (strategy: {strategy})")
            
        try:
            # Store in memory cache
            self.memory_cache[cache_key] = result
            self.cache_timestamps[cache_key] = time.time()
            logger.debug(f"Stored in memory cache, total entries: {len(self.memory_cache)}")
            
            # Store in Redis if available
            if self.redis_client:
                import json
                cached_data = json.dumps(result.to_dict(), default=str)
                data_size = len(cached_data)
                logger.debug(f"Storing {data_size} bytes in Redis with TTL: {self.config.caching.cache_ttl}s")
                self.redis_client.setex(cache_key, self.config.caching.cache_ttl, cached_data)
                logger.debug("Successfully stored in Redis cache")
            else:
                logger.debug("Redis client not available - only stored in memory")
            
            logger.debug(f"Cached result for key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Cache put error for key {cache_key}: {e}")
            logger.debug(f"Full cache put error: {str(e)}", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'redis_available': self.redis_client is not None
        }


class ExecutionTimeout:
    """Manages execution timeouts"""
    
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
    
    @asynccontextmanager
    async def timeout_context(self):
        """Context manager for timeout handling"""
        try:
            async with asyncio.timeout(self.timeout_seconds):
                yield
        except asyncio.TimeoutError:
            logger.error(f"Execution timed out after {self.timeout_seconds}s")
            raise


class ErrorHandler:
    """Unified error handling for all strategies"""
    
    def __init__(self, config: UnifiedConfig):
        logger.debug(f"Initializing ErrorHandler with max retries: {config.performance.max_retries}")
        self.config = config
        self.retry_counts: Dict[str, int] = {}
        logger.info(f"ErrorHandler initialized with max_retries: {config.performance.max_retries}")
    
    async def handle_execution_error(self, 
                                   error: Exception, 
                                   strategy: str,
                                   query: str,
                                   attempt: int) -> bool:
        """
        Handle execution error and determine if retry should be attempted
        
        Returns:
            True if should retry, False otherwise
        """
        error_key = f"{strategy}:{query[:50]}"
        error_type = type(error).__name__
                
        logger.error(f"Execution error in strategy {strategy} (attempt {attempt}): {error_type}: {str(error)}")
        logger.debug(f"Full error traceback for {error_key}:", exc_info=True)
        
        # Update retry count
        self.retry_counts[error_key] = self.retry_counts.get(error_key, 0) + 1
        current_retry_count = self.retry_counts[error_key]
        
        logger.debug(f"Error key: {error_key}, current retry count: {current_retry_count}")
        
        # Check if we should retry
        is_retryable = self._is_retryable_error(error)
        within_limit = attempt < self.config.performance.max_retries
        
        should_retry = within_limit and is_retryable
        
        logger.debug(f"Retry decision: within_limit={within_limit}, is_retryable={is_retryable}, should_retry={should_retry}")
        
        if should_retry:
            delay = self.config.performance.retry_delay_seconds * (
                self.config.performance.backoff_multiplier ** (attempt - 1)
            )
            logger.warning(f"Execution failed (attempt {attempt}), retrying in {delay}s: {str(error)}")
            logger.debug(f"Backoff delay calculation: base={self.config.performance.retry_delay_seconds}, multiplier={self.config.performance.backoff_multiplier}, attempt={attempt}")
            await asyncio.sleep(delay)
        else:
            if not within_limit:
                logger.error(f"Execution failed permanently - max retries ({self.config.performance.max_retries}) exceeded after {attempt} attempts: {str(error)}")
            elif not is_retryable:
                logger.error(f"Execution failed permanently - error not retryable: {error_type}: {str(error)}")
        
        return should_retry
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        logger.debug(f"Checking if error is retryable: {error_type}: {error_msg[:100]}...")
        
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        )
        
        # Also check for specific error messages
        retryable_messages = [
            'timeout', 'connection', 'network', 'temporary', 'rate limit'
        ]
        
        is_retryable_type = isinstance(error, retryable_errors)
        has_retryable_message = any(msg in error_msg for msg in retryable_messages)
        
        is_retryable = is_retryable_type or has_retryable_message
        
        logger.debug(f"Retryable analysis: type_match={is_retryable_type}, message_match={has_retryable_message}, is_retryable={is_retryable}")
        
        if has_retryable_message:
            matched_messages = [msg for msg in retryable_messages if msg in error_msg]
            logger.debug(f"Matched retryable messages: {matched_messages}")
        
        return is_retryable


class UnifiedExecutionEngine:
    """
    Unified execution engine but without any strategies/strategy router logic.
    """
    
    def __init__(self, config: UnifiedConfig):
        logger.info("Initializing UnifiedExecutionEngine...")
        logger.debug(f"Config: {config}")
        
        self.config = config
        
        logger.debug("Initializing entity processor...")
        self.entity_processor = UnifiedEntityProcessor(config)
        
        logger.debug("Initializing performance monitor...")
        self.performance_monitor = UnifiedPerformanceMonitor(config)
        
        logger.debug("Initializing cache...")
        self.cache = UnifiedCache(config)
        
        logger.debug("Initializing error handler...")
        self.error_handler = ErrorHandler(config)
        
        logger.info("Unified execution engine initialized (no strategy logic)")
        logger.debug(f"Initialization complete - components ready: entity_processor, performance_monitor, cache, error_handler")
        # PATCH: Provide a fallback dummy strategy selector
        self.strategy_router = self
        
        # PATCH: Add simple executor registry
        self.executor_registry = self
        self._executors = {
            'got_framework': self._execute_got_strategy,
            'simple': self._execute_simple_strategy,
            'langgraph': self._execute_langgraph_strategy,
            'production_got': self._execute_production_got_strategy,
            'hybrid': self._execute_hybrid_strategy
        }

    def get_executor(self, strategy: str) -> Optional[Callable]:
        """Get executor for strategy"""
        return self._executors.get(strategy)
    
    def get_available_strategies(self) -> List[str]:
        """Get available strategies"""
        return list(self._executors.keys())
    
    async def select_strategy(self, query, entity_context):
        class DummyRecommendation:
            primary_strategy = 'got_framework'
            fallback_strategies = ['simple', 'langgraph']
            confidence = 1.0
            resource_requirements = {}
        return DummyRecommendation()
    
    
    async def execute_query(self, query: str, **kwargs) -> BiomedicalResult:
        """
        Execute a biomedical query using the optimal strategy
        
        Args:
            query: Natural language biomedical query
            **kwargs: Additional execution parameters
            
        Returns:
            Comprehensive biomedical result
        """
        logger.info(f"=== STARTING QUERY EXECUTION ===")
        logger.info(f"Query: {query}")
        logger.debug(f"Additional kwargs: {kwargs}")
        start_time = time.time()
        
        try:
            # Step 1: Process entities
            logger.info("Step 1: Processing entities...")
            entity_start = time.time()
            entity_context = await self.entity_processor.process_entities(query)
            entity_time = time.time() - entity_start
            logger.info(f"Entity processing completed in {entity_time:.2f}s")
            logger.debug(f"Processed entities: {[entity.name for entity in entity_context.entities] if hasattr(entity_context, 'entities') else 'no entities'}")
            
            # Step 2: Select optimal strategy
            logger.info("Step 2: Selecting strategy...")
            strategy_start = time.time()
            recommendation = await self.strategy_router.select_strategy(query, entity_context)
            primary_strategy = recommendation.primary_strategy
            strategy_time = time.time() - strategy_start
            
            logger.info(f"Selected strategy: {primary_strategy} (confidence: {recommendation.confidence:.2f}) in {strategy_time:.2f}s")
            logger.debug(f"Fallback strategies: {recommendation.fallback_strategies}")
            logger.debug(f"Resource requirements: {recommendation.resource_requirements}")
            
            # Step 3: Create execution plan
            logger.info("Step 3: Creating execution plan...")
            plan_start = time.time()
            execution_plan = self._create_execution_plan(
                query, recommendation, entity_context, kwargs
            )
            plan_time = time.time() - plan_start
            logger.info(f"Execution plan created in {plan_time:.2f}s")
            logger.debug(f"Plan configuration keys: {list(execution_plan.configuration.keys())}")
            
            # Step 4: Execute with caching and monitoring
            logger.info("Step 4: Executing with monitoring...")
            async with self.performance_monitor.monitor_query_execution(query, primary_strategy) as execution_data:
                exec_start = time.time()
                result = await self._execute_with_fallbacks(execution_plan)
                exec_time = time.time() - exec_start
                
                logger.info(f"Execution completed in {exec_time:.2f}s")
                logger.debug(f"Result success: {result.success}, confidence: {getattr(result, 'confidence', 'N/A')}")
                
                # Update execution data
                execution_data.update({
                    'quality_score': getattr(result, 'quality_score', 0.0),
                    'confidence': getattr(result, 'confidence', 0.0)
                })
            
            # Step 5: Update performance metrics
            logger.info("Step 5: Updating performance metrics...")
            result.performance_metrics = self.performance_monitor.create_performance_metrics(result.execution_steps)
            
            total_time = time.time() - start_time
            logger.info(f"=== QUERY EXECUTION COMPLETED in {total_time:.2f}s ===")
            logger.info(f"Final result - Success: {result.success}, Answer length: {len(result.final_answer) if result.final_answer else 0} chars")
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"=== QUERY EXECUTION FAILED after {total_time:.2f}s ===")
            logger.error(f"Error: {type(e).__name__}: {str(e)}")
            logger.debug(f"Full execution error:", exc_info=True)
            return create_error_result(query, str(e), "simple")
    
    def _create_execution_plan(self, 
                             query: str, 
                             recommendation, 
                             entity_context: EntityContext,
                             kwargs: Dict[str, Any]) -> ExecutionPlan:
        """Create comprehensive execution plan"""
        logger.debug(f"Creating execution plan for strategy: {recommendation.primary_strategy}")
        
        # Create execution context
        has_kg = hasattr(entity_context, 'knowledge_graph') and entity_context.knowledge_graph is not None
        logger.debug(f"Entity context has knowledge graph: {has_kg}")
        
        execution_context = ExecutionContext(
            query=query,
            strategy=recommendation.primary_strategy,
            entity_context=entity_context,
            knowledge_graph=entity_context.knowledge_graph if has_kg else None,
            config=self.config
        )
        
        # Merge configuration
        logger.debug("Merging configuration...")
        configuration = {
            'strategy_config': self.config.get_strategy_config(recommendation.primary_strategy),
            'performance_config': self.config.performance,
            'quality_config': self.config.quality,
            'domain_config': self.config.domain,
            **kwargs
        }
        
        logger.debug(f"Configuration merged - total keys: {len(configuration)}")
        logger.debug(f"Estimated resources: {recommendation.resource_requirements}")
        
        execution_plan = ExecutionPlan(
            primary_strategy=recommendation.primary_strategy,
            fallback_strategies=recommendation.fallback_strategies,
            entity_context=entity_context,
            execution_context=execution_context,
            configuration=configuration,
            estimated_resources=recommendation.resource_requirements
        )
        
        logger.info(f"Execution plan created for {recommendation.primary_strategy} with {len(recommendation.fallback_strategies)} fallbacks")
        return execution_plan
    
    async def _execute_with_fallbacks(self, plan: ExecutionPlan) -> BiomedicalResult:
        """Execute strategy with fallback support"""
        strategies_to_try = [plan.primary_strategy] + plan.fallback_strategies
        last_error = None
        
        logger.info(f"Executing with fallback support - {len(strategies_to_try)} strategies to try")
        logger.debug(f"Strategy sequence: {strategies_to_try}")
        
        for i, strategy in enumerate(strategies_to_try):
            try:
                logger.info(f"=== STRATEGY {i+1}/{len(strategies_to_try)}: {strategy} ===")
                strategy_start = time.time()
                
                # Check cache first
                logger.debug("Checking cache...")
                config_hash = self._compute_config_hash(plan.configuration)
                logger.debug(f"Config hash: {config_hash}")
                
                cached_result = await self.cache.get(plan.execution_context.query, strategy, config_hash)
                
                if cached_result:
                    cache_time = time.time() - strategy_start
                    logger.info(f"Using cached result (retrieved in {cache_time:.3f}s)")
                    logger.debug(f"Cached result success: {cached_result.success}, answer length: {len(cached_result.final_answer) if cached_result.final_answer else 0}")
                    self.performance_monitor.record_cache_hit(f"strategy_{str(strategy)}")
                    return cached_result
                
                logger.debug(f"No cached result found")
                self.performance_monitor.record_cache_miss(f"strategy_{str(strategy)}")
                
                # Execute strategy with retries
                logger.debug("Executing strategy with retries...")
                result = await self._execute_strategy_with_retries(strategy, plan)
                
                strategy_time = time.time() - strategy_start
                logger.info(f"Strategy {strategy} completed in {strategy_time:.2f}s")
                
                if result and result.success:
                    logger.info(f"Strategy {strategy} succeeded")
                    logger.debug(f"Result confidence: {getattr(result, 'confidence', 'N/A')}, answer length: {len(result.final_answer) if result.final_answer else 0}")
                    
                    # Cache successful result
                    logger.debug("Caching successful result...")
                    await self.cache.put(plan.execution_context.query, strategy, config_hash, result)
                    return result
                else:
                    raise Exception(f"Strategy {strategy} returned unsuccessful result: success={result.success if result else 'None'}")
                    
            except Exception as e:
                last_error = e
                strategy_time = time.time() - strategy_start
                logger.warning(f"Strategy {strategy} failed after {strategy_time:.2f}s: {type(e).__name__}: {str(e)}")
                logger.debug(f"Strategy {strategy} full error:", exc_info=True)
                
                if i < len(strategies_to_try) - 1:
                    next_strategy = strategies_to_try[i+1]
                    logger.info(f"Falling back to next strategy: {next_strategy}")
                    continue
                else:
                    logger.error(f"No more fallback strategies available")
                    break
        
        # All strategies failed
        error_msg = f"All strategies failed. Last error: {str(last_error)}"
        logger.error(f"=== ALL STRATEGIES FAILED ===")
        logger.error(f"Tried strategies: {strategies_to_try}")
        logger.error(f"Final error: {error_msg}")
        return create_error_result(plan.execution_context.query, error_msg, str(plan.primary_strategy))

    async def _execute_strategy_with_retries(self, strategy: str, plan: ExecutionPlan) -> BiomedicalResult:
        """Execute strategy with retry logic"""
        logger.debug(f"Executing strategy {strategy} with retry logic (max retries: {self.config.performance.max_retries})")
        
        executor = self.executor_registry.get_executor(strategy)
        if not executor:
            error_msg = f"No executor registered for strategy: {strategy}"
            logger.error(error_msg)
            logger.debug(f"Available executors: {list(self.executor_registry.get_available_strategies())}")
            raise Exception(error_msg)
        
        logger.debug(f"Found executor for strategy: {strategy}")
        
        for attempt in range(1, self.config.performance.max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt}/{self.config.performance.max_retries} for strategy {strategy}")
                attempt_start = time.time()
                
                # Execute with timeout
                timeout_seconds = self.config.performance.query_timeout_seconds
                logger.debug(f"Using timeout: {timeout_seconds}s")
                timeout_manager = ExecutionTimeout(timeout_seconds)
                
                async with timeout_manager.timeout_context():
                    logger.debug(f"Calling executor for strategy {strategy}...")
                    result = await executor(plan)
                    
                    attempt_time = time.time() - attempt_start
                    logger.debug(f"Executor returned result in {attempt_time:.2f}s")
                    logger.debug(f"Result type: {type(result)}, success: {result.success if result else 'None'}")
                    
                    if result and result.success:
                        logger.info(f"Strategy {strategy} succeeded on attempt {attempt}")
                        return result
                    else:
                        result_info = f"success={result.success}" if result else "result=None"
                        error_msg = f"Strategy execution returned unsuccessful result: {result_info}"
                        logger.warning(error_msg)
                        raise Exception(error_msg)
                        
            except Exception as e:
                attempt_time = time.time() - attempt_start
                logger.warning(f"Attempt {attempt} failed after {attempt_time:.2f}s: {type(e).__name__}: {str(e)}")
                
                should_retry = await self.error_handler.handle_execution_error(
                    e, strategy, plan.execution_context.query, attempt
                )
                
                if not should_retry:
                    logger.error(f"Not retrying - either max attempts reached or non-retryable error")
                    raise
                
                logger.debug(f"Will retry attempt {attempt + 1}...")
        
        final_error = f"Strategy {strategy} failed after {self.config.performance.max_retries} attempts"
        logger.error(final_error)
        raise Exception(final_error)
    
    def _compute_config_hash(self, configuration: Dict[str, Any]) -> str:
        """Compute hash of configuration for caching"""
        import hashlib
        import json
        
        # Create a deterministic string from configuration
        config_str = json.dumps(configuration, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    # Strategy Executors
    async def _execute_simple_strategy(self, plan: ExecutionPlan) -> BiomedicalResult:
        """Execute simple strategy"""
        from ..core.queries.simple_working_optimizer import SimpleWorkingOptimizer
        
        optimizer = SimpleWorkingOptimizer(config=None)  # Will use default config
        result = optimizer.optimize_query(
            plan.execution_context.query,
            entities={entity.name: entity.entity_id for entity in plan.entity_context.entities}
        )
        
        # Convert to unified result format
        return self._convert_legacy_result(result, plan, "basic_adaptive")
    
    async def _execute_got_strategy(self, plan: ExecutionPlan) -> BiomedicalResult:
        """Execute GoT framework strategy"""
        from ..core.queries.got_optimizers import GoTEnhancedSimpleOptimizer
        
        optimizer = GoTEnhancedSimpleOptimizer(enable_got=True, enable_parallel=True)
        result = optimizer.optimize_query(
            plan.execution_context.query,
            entities={entity.name: entity.entity_id for entity in plan.entity_context.entities}
        )
        
        return self._convert_legacy_result(result, plan, "got_framework")
    
    async def _execute_langgraph_strategy(self, plan: ExecutionPlan) -> BiomedicalResult:
        """Execute LangGraph multi-agent strategy"""
        from ..agents.orchestrator import BiomedicalOrchestrator
        
        orchestrator = BiomedicalOrchestrator()
        result = orchestrator.execute_research(
            plan.execution_context.query,
            maxresults=plan.configuration.get('max_results', 50),
            k=plan.configuration.get('k', 5)
        )
        
        return self._convert_langgraph_result(result, plan)
    
    async def _execute_production_got_strategy(self, plan: ExecutionPlan) -> BiomedicalResult:
        """Execute production GoT strategy"""
        from ..core.queries.production_got_optimizer import ProductionGoTOptimizer
        
        optimizer = ProductionGoTOptimizer()
        result, _ = await optimizer.execute_query(plan.execution_context.query)
        
        return self._convert_production_result(result, plan)
    
    async def _execute_enhanced_got_strategy(self, plan: ExecutionPlan) -> BiomedicalResult:
        """Execute enhanced GoT with domain expertise"""
        from ..core.queries.enhanced_got_optimizer import DomainExpertAnswerGenerator, EnhancedConfig
        
        config = EnhancedConfig()
        generator = DomainExpertAnswerGenerator(config)
        
        # This would need to be implemented to work with the enhanced optimizer
        # For now, fall back to production GoT
        return await self._execute_production_got_strategy(plan)
    
    async def _execute_stateful_strategy(self, plan: ExecutionPlan) -> BiomedicalResult:
        """Execute stateful iterative strategy"""
        # Import and use the stateful GoT optimizer
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        try:
            from stateful_got_optimizer import StatefulGoTOptimizer
            
            optimizer = StatefulGoTOptimizer()
            result, _ = await optimizer.execute_query(plan.execution_context.query)
            
            return result  # Already in unified format
            
        except ImportError:
            logger.warning("Stateful optimizer not available, falling back to production GoT")
            return await self._execute_production_got_strategy(plan)
    
    async def _execute_hybrid_strategy(self, plan: ExecutionPlan) -> BiomedicalResult:
        """Execute hybrid adaptive strategy"""
        from ..core.queries.hybrid_optimizer import HybridIntelligentOptimizer
        
        optimizer = HybridIntelligentOptimizer()
        result = optimizer.optimize_query(
            plan.execution_context.query,
            entities={entity.name: entity.entity_id for entity in plan.entity_context.entities}
        )
        
        return self._convert_legacy_result(result, plan, "hybrid_intelligent")
    
    # Result Converters
    def _convert_legacy_result(self, legacy_result, plan: ExecutionPlan, strategy: str) -> BiomedicalResult:
        """Convert legacy optimization result to unified format"""
        # from agentic_bte.core.queries.interfaces import OptimizationResult
        if not isinstance(legacy_result, OptimizationResult):
            # Handle simple results
            return create_error_result(plan.execution_context.query, "Invalid result format", strategy)
        
        # Convert to unified format
        unified_result = BiomedicalResult(
            query=plan.execution_context.query,
            strategy_used=strategy,
            final_answer=legacy_result.final_answer or "No answer generated",
            knowledge_graph=plan.entity_context.knowledge_graph if hasattr(plan.entity_context, 'knowledge_graph') else None,
            entity_context=plan.entity_context,
            execution_steps=self._convert_execution_steps(legacy_result),
            performance_metrics=self._convert_performance_metrics(legacy_result)
        )
        
        unified_result.success = legacy_result.success
        unified_result.confidence = getattr(legacy_result, 'confidence', 0.7)
        
        return unified_result
    
    def _convert_langgraph_result(self, langgraph_result: Dict[str, Any], plan: ExecutionPlan) -> BiomedicalResult:
        """Convert LangGraph result to unified format"""
        # Extract information from LangGraph result
        final_answer = langgraph_result.get('final_answer', 'No answer generated')
        success = langgraph_result.get('success', False)
        
        unified_result = BiomedicalResult(
            query=plan.execution_context.query,
            strategy_used="langgraph_agents",
            final_answer=final_answer,
            knowledge_graph=plan.entity_context.knowledge_graph if hasattr(plan.entity_context, 'knowledge_graph') else None,
            entity_context=plan.entity_context,
            execution_steps=[],  # Would need to extract from LangGraph
            performance_metrics=PerformanceMetrics()
        )
        
        unified_result.success = success
        return unified_result
    
    def _convert_production_result(self, production_result, plan: ExecutionPlan) -> BiomedicalResult:
        """Convert production GoT result to unified format"""
        if hasattr(production_result, 'to_dict'):
            # Already in a good format
            return production_result
        
        # Handle different result formats
        return BiomedicalResult(
            query=plan.execution_context.query,
            strategy_used="parallel_execution",
            final_answer=getattr(production_result, 'final_answer', 'No answer generated'),
            knowledge_graph=plan.entity_context.knowledge_graph if hasattr(plan.entity_context, 'knowledge_graph') else None,
            entity_context=plan.entity_context,
            execution_steps=getattr(production_result, 'execution_steps', []),
            performance_metrics=getattr(production_result, 'performance_metrics', PerformanceMetrics())
        )
    
    def _convert_execution_steps(self, legacy_result) -> List[ExecutionStep]:
        """Convert legacy execution steps to unified format"""
        steps = []
        
        # Extract steps from legacy result if available
        if hasattr(legacy_result, 'execution_steps'):
            for step_data in legacy_result.execution_steps:
                step = ExecutionStep(
                    step_id=getattr(step_data, 'step_id', f"step_{len(steps)}"),
                    step_type=getattr(step_data, 'step_type', 'unknown'),
                    status=ExecutionStatus.SUCCESS if getattr(step_data, 'success', True) else ExecutionStatus.FAILED
                )
                
                if hasattr(step_data, 'execution_time'):
                    step.execution_time = step_data.execution_time
                
                if hasattr(step_data, 'confidence'):
                    step.confidence = step_data.confidence
                
                steps.append(step)
        
        return steps
    
    def _convert_performance_metrics(self, legacy_result) -> PerformanceMetrics:
        """Convert legacy performance metrics to unified format"""
        metrics = PerformanceMetrics()
        
        if hasattr(legacy_result, 'metrics'):
            legacy_metrics = legacy_result.metrics
            
            # Map common metrics
            if hasattr(legacy_metrics, 'total_execution_time'):
                metrics.total_execution_time = legacy_metrics.total_execution_time
            
            if hasattr(legacy_metrics, 'cache_hits'):
                metrics.cache_hits = legacy_metrics.cache_hits
                metrics.cache_misses = getattr(legacy_metrics, 'cache_misses', 0)
            
            if hasattr(legacy_metrics, 'error_count'):
                metrics.error_count = legacy_metrics.error_count
        
        return metrics
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        return {
            'cache_stats': self.cache.get_stats(),
            'performance_stats': self.performance_monitor.get_comprehensive_report(),
            'available_strategies': list(self.executor_registry.get_available_strategies()),
            'error_stats': {
                'total_retries': len(self.error_handler.retry_counts),
                'retry_counts': self.error_handler.retry_counts
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the execution engine"""
        logger.info("=== INITIALIZING UNIFIED EXECUTION ENGINE ===")
        
        # Check component status
        logger.debug("Checking component initialization status...")
        
        components_status = {
            'entity_processor': self.entity_processor is not None,
            'performance_monitor': self.performance_monitor is not None,
            'cache': self.cache is not None,
            'error_handler': self.error_handler is not None
        }
        
        logger.debug(f"Component status: {components_status}")
        
        # Test cache functionality if enabled
        if self.config.caching.enable_caching:
            logger.debug("Testing cache functionality...")
            cache_stats = self.cache.get_stats()
            logger.debug(f"Initial cache stats: {cache_stats}")
        
        # Test performance monitor
        logger.debug("Testing performance monitor...")
        try:
            perf_report = self.performance_monitor.get_comprehensive_report()
            logger.debug(f"Performance monitor ready - report keys: {list(perf_report.keys())}")
        except Exception as e:
            logger.warning(f"Performance monitor test failed: {e}")
        
        logger.info("UnifiedExecutionEngine initialization completed")
        logger.info(f"Ready with components: {', '.join([k for k, v in components_status.items() if v])}")
    
    async def execute(
        self,
        strategy: str,
        queries: List[Any],
        context: Any
    ) -> List[Any]:
        """Execute queries using specified strategy"""
        logger.info(f"=== EXECUTING {len(queries)} QUERIES with strategy {strategy} ===")
        logger.debug(f"Queries: {queries}")
        logger.debug(f"Context: {context}")
        
        # Simple mock implementation for demo
        results = []
        
        for i, query in enumerate(queries):
            logger.debug(f"Processing query {i+1}/{len(queries)}: {query}")
            
            # Create mock result
            result = type('MockResult', (), {
                'subject_entity': 'diabetes',
                'predicate': 'biolink:treats',
                'object_entity': 'metformin',
                'confidence': 0.85,
                'evidence_count': 10,
                'source_databases': ['demo_db']
            })()
            results.append(result)
            
            logger.debug(f"Created mock result for query {i+1}: {result.subject_entity} {result.predicate} {result.object_entity}")
        
        logger.info(f"Completed execution - {len(results)} results generated")
        return results
