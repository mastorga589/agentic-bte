"""
Tests for the unified execution engine
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import time

from ..config import UnifiedConfig
from ..types import BiomedicalResult, EntityContext, create_error_result
from ..execution_engine import (
    UnifiedExecutionEngine, UnifiedCache, ErrorHandler, ExecutionTimeout, ExecutionPlan
)


@pytest.fixture
def mock_config():
    """Create a mock unified config"""
    config = Mock(spec=UnifiedConfig)
    
    # Mock caching config
    config.caching = Mock()
    config.caching.enable_caching = True
    config.caching.cache_ttl = 3600
    config.caching.backend.value = "memory"
    
    # Mock performance config
    config.performance = Mock()
    config.performance.max_retries = 3
    config.performance.retry_delay_seconds = 1.0
    config.performance.backoff_multiplier = 2.0
    config.performance.query_timeout_seconds = 300.0
    
    # Mock strategy config
    config.get_strategy_config = Mock(return_value={})
    
    return config


@pytest.fixture
def mock_entity_context():
    """Create a mock entity context"""
    context = Mock(spec=EntityContext)
    context.entities = []
    context.query = "test query"
    context.confidence = 0.8
    return context


class TestUnifiedCache:
    """Test the unified caching system"""
    
    def test_cache_initialization(self, mock_config):
        """Test cache initialization"""
        cache = UnifiedCache(mock_config)
        
        assert cache.config == mock_config
        assert cache.memory_cache == {}
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0
        assert cache.redis_client is None
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, mock_config):
        """Test cache miss scenario"""
        cache = UnifiedCache(mock_config)
        
        result = await cache.get("test query", ExecutionStrategy.SIMPLE, "hash123")
        
        assert result is None
        assert cache.cache_misses == 1
        assert cache.cache_hits == 0
    
    @pytest.mark.asyncio
    async def test_cache_put_and_hit(self, mock_config):
        """Test cache put and subsequent hit"""
        cache = UnifiedCache(mock_config)
        
        # Create a mock result
        mock_result = Mock(spec=BiomedicalResult)
        mock_result.success = True
        
        # Put in cache
        await cache.put("test query", ExecutionStrategy.SIMPLE, "hash123", mock_result)
        
        # Get from cache
        result = await cache.get("test query", ExecutionStrategy.SIMPLE, "hash123")
        
        assert result == mock_result
        assert cache.cache_hits == 1
        assert cache.cache_misses == 0
    
    @pytest.mark.asyncio
    async def test_cache_expiry(self, mock_config):
        """Test cache expiry functionality"""
        # Set very short TTL
        mock_config.caching.cache_ttl = 0.1
        cache = UnifiedCache(mock_config)
        
        # Create and cache a result
        mock_result = Mock(spec=BiomedicalResult)
        mock_result.success = True
        
        await cache.put("test query", ExecutionStrategy.SIMPLE, "hash123", mock_result)
        
        # Wait for expiry
        await asyncio.sleep(0.2)
        
        # Should be cache miss due to expiry
        result = await cache.get("test query", ExecutionStrategy.SIMPLE, "hash123")
        assert result is None
        assert cache.cache_misses == 1
    
    def test_cache_stats(self, mock_config):
        """Test cache statistics"""
        cache = UnifiedCache(mock_config)
        cache.cache_hits = 5
        cache.cache_misses = 3
        
        stats = cache.get_stats()
        
        assert stats['cache_hits'] == 5
        assert stats['cache_misses'] == 3
        assert stats['hit_rate'] == 5/8
        assert stats['memory_cache_size'] == 0
        assert stats['redis_available'] is False


class TestErrorHandler:
    """Test the error handling system"""
    
    def test_error_handler_initialization(self, mock_config):
        """Test error handler initialization"""
        handler = ErrorHandler(mock_config)
        
        assert handler.config == mock_config
        assert handler.retry_counts == {}
    
    @pytest.mark.asyncio
    async def test_retryable_error_handling(self, mock_config):
        """Test handling of retryable errors"""
        handler = ErrorHandler(mock_config)
        
        # Test with timeout error (retryable)
        error = TimeoutError("Connection timeout")
        should_retry = await handler.handle_execution_error(
            error, ExecutionStrategy.SIMPLE, "test query", 1
        )
        
        assert should_retry is True
        assert "simple:test query" in handler.retry_counts
    
    @pytest.mark.asyncio
    async def test_non_retryable_error_handling(self, mock_config):
        """Test handling of non-retryable errors"""
        handler = ErrorHandler(mock_config)
        
        # Test with value error (non-retryable)
        error = ValueError("Invalid input")
        should_retry = await handler.handle_execution_error(
            error, ExecutionStrategy.SIMPLE, "test query", 1
        )
        
        assert should_retry is False
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, mock_config):
        """Test behavior when max retries is exceeded"""
        handler = ErrorHandler(mock_config)
        
        error = TimeoutError("Connection timeout")
        should_retry = await handler.handle_execution_error(
            error, ExecutionStrategy.SIMPLE, "test query", 
            mock_config.performance.max_retries + 1
        )
        
        assert should_retry is False


class TestExecutionTimeout:
    """Test execution timeout management"""
    
    @pytest.mark.asyncio
    async def test_successful_execution_within_timeout(self):
        """Test successful execution within timeout"""
        timeout_manager = ExecutionTimeout(1.0)
        
        async def quick_task():
            await asyncio.sleep(0.1)
            return "success"
        
        async with timeout_manager.timeout_context():
            result = await quick_task()
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_execution_timeout(self):
        """Test execution that exceeds timeout"""
        timeout_manager = ExecutionTimeout(0.1)
        
        async def slow_task():
            await asyncio.sleep(1.0)
            return "success"
        
        with pytest.raises(asyncio.TimeoutError):
            async with timeout_manager.timeout_context():
                await slow_task()


# StrategyExecutorRegistry tests removed (no strategy logic remains)


class TestUnifiedExecutionEngine:
    """Test the main execution engine"""
    
    @pytest.fixture
    def mock_components(self, mock_config, mock_entity_context):
        """Create mocked components for the execution engine"""
        with patch.multiple(
            'agentic_bte.unified.execution_engine',
            UnifiedEntityProcessor=Mock(),
            UnifiedPerformanceMonitor=Mock(),
            UnifiedStrategyRouter=Mock(),
        ) as mocks:
            
            # Mock entity processor
            mocks['UnifiedEntityProcessor'].return_value.process_entities = AsyncMock(
                return_value=mock_entity_context
            )
            
            # Mock performance monitor
            mock_perf_monitor = Mock()
            mock_perf_monitor.monitor_query_execution = AsyncMock()
            mock_perf_monitor.create_performance_metrics = Mock()
            mock_perf_monitor.record_cache_hit = Mock()
            mock_perf_monitor.record_cache_miss = Mock()
            mocks['UnifiedPerformanceMonitor'].return_value = mock_perf_monitor
            
            # Mock strategy router
            mock_recommendation = Mock()
            mock_recommendation.primary_strategy = ExecutionStrategy.SIMPLE
            mock_recommendation.fallback_strategies = []
            mock_recommendation.confidence = 0.8
            mock_recommendation.resource_requirements = {}
            
            mock_router = Mock()
            mock_router.select_strategy = AsyncMock(return_value=mock_recommendation)
            mocks['UnifiedStrategyRouter'].return_value = mock_router
            
            yield mocks
    
    def test_execution_engine_initialization(self, mock_config, mock_components):
        """Test execution engine initialization"""
        engine = UnifiedExecutionEngine(mock_config)
        
        assert engine.config == mock_config
        assert engine.entity_processor is not None
        assert engine.performance_monitor is not None
        assert engine.strategy_router is not None
        assert engine.cache is not None
        assert engine.error_handler is not None
        assert engine.executor_registry is not None
    
    @pytest.mark.asyncio
    async def test_successful_query_execution(self, mock_config, mock_components):
        """Test successful query execution"""
        engine = UnifiedExecutionEngine(mock_config)
        
        # Mock a successful strategy execution
        mock_result = Mock(spec=BiomedicalResult)
        mock_result.success = True
        mock_result.quality_score = 0.9
        mock_result.confidence = 0.8
        mock_result.execution_steps = []
        
        # Register mock executor
        mock_executor = AsyncMock(return_value=mock_result)
        engine.executor_registry.register_executor(ExecutionStrategy.SIMPLE, mock_executor)
        
        # Mock performance monitor context
        engine.performance_monitor.monitor_query_execution.return_value.__aenter__ = AsyncMock(return_value={})
        engine.performance_monitor.monitor_query_execution.return_value.__aexit__ = AsyncMock(return_value=None)
        
        result = await engine.execute_query("test query")
        
        assert result is not None
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_query_execution_with_cache_hit(self, mock_config, mock_components):
        """Test query execution with cache hit"""
        engine = UnifiedExecutionEngine(mock_config)
        
        # Pre-populate cache
        mock_cached_result = Mock(spec=BiomedicalResult)
        mock_cached_result.success = True
        
        # Mock cache hit
        engine.cache.get = AsyncMock(return_value=mock_cached_result)
        
        # Mock performance monitor context
        engine.performance_monitor.monitor_query_execution.return_value.__aenter__ = AsyncMock(return_value={})
        engine.performance_monitor.monitor_query_execution.return_value.__aexit__ = AsyncMock(return_value=None)
        
        result = await engine.execute_query("test query")
        
        assert result == mock_cached_result
        engine.performance_monitor.record_cache_hit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_execution_fallback(self, mock_config, mock_components):
        """Test query execution with fallback strategy"""
        engine = UnifiedExecutionEngine(mock_config)
        
        # Update mock router to return fallback strategies
        mock_recommendation = Mock()
        mock_recommendation.primary_strategy = ExecutionStrategy.SIMPLE
        mock_recommendation.fallback_strategies = [ExecutionStrategy.GOT_FRAMEWORK]
        mock_recommendation.confidence = 0.8
        mock_recommendation.resource_requirements = {}
        
        engine.strategy_router.select_strategy = AsyncMock(return_value=mock_recommendation)
        
        # Mock first strategy failure
        failing_executor = AsyncMock(side_effect=Exception("Primary strategy failed"))
        engine.executor_registry.register_executor(ExecutionStrategy.SIMPLE, failing_executor)
        
        # Mock successful fallback
        mock_result = Mock(spec=BiomedicalResult)
        mock_result.success = True
        mock_result.quality_score = 0.7
        mock_result.confidence = 0.6
        mock_result.execution_steps = []
        
        successful_executor = AsyncMock(return_value=mock_result)
        engine.executor_registry.register_executor(ExecutionStrategy.GOT_FRAMEWORK, successful_executor)
        
        # Mock performance monitor context
        engine.performance_monitor.monitor_query_execution.return_value.__aenter__ = AsyncMock(return_value={})
        engine.performance_monitor.monitor_query_execution.return_value.__aexit__ = AsyncMock(return_value=None)
        
        result = await engine.execute_query("test query")
        
        assert result is not None
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_query_execution_all_strategies_fail(self, mock_config, mock_components):
        """Test query execution when all strategies fail"""
        engine = UnifiedExecutionEngine(mock_config)
        
        # Register failing executor
        failing_executor = AsyncMock(side_effect=Exception("Strategy failed"))
        engine.executor_registry.register_executor(ExecutionStrategy.SIMPLE, failing_executor)
        
        # Mock performance monitor context
        engine.performance_monitor.monitor_query_execution.return_value.__aenter__ = AsyncMock(return_value={})
        engine.performance_monitor.monitor_query_execution.return_value.__aexit__ = AsyncMock(return_value=None)
        
        result = await engine.execute_query("test query")
        
        assert result is not None
        assert result.success is False
        assert "All strategies failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_strategy_execution_with_retries(self, mock_config):
        """Test strategy execution with retry logic"""
        engine = UnifiedExecutionEngine(mock_config)
        
        # Mock execution plan
        mock_plan = Mock(spec=ExecutionPlan)
        mock_plan.execution_context.query = "test query"
        
        # Mock executor that fails twice then succeeds
        call_count = 0
        
        async def mock_executor(plan):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise TimeoutError("Temporary failure")
            
            result = Mock(spec=BiomedicalResult)
            result.success = True
            return result
        
        engine.executor_registry.register_executor(ExecutionStrategy.SIMPLE, mock_executor)
        
        result = await engine._execute_strategy_with_retries(ExecutionStrategy.SIMPLE, mock_plan)
        
        assert result is not None
        assert result.success is True
        assert call_count == 3  # Failed twice, succeeded on third try
    
    def test_config_hash_computation(self, mock_config):
        """Test configuration hash computation"""
        engine = UnifiedExecutionEngine(mock_config)
        
        config1 = {'param1': 'value1', 'param2': 'value2'}
        config2 = {'param2': 'value2', 'param1': 'value1'}  # Same but different order
        config3 = {'param1': 'value1', 'param2': 'different'}
        
        hash1 = engine._compute_config_hash(config1)
        hash2 = engine._compute_config_hash(config2)
        hash3 = engine._compute_config_hash(config3)
        
        # Same configs should have same hash regardless of key order
        assert hash1 == hash2
        
        # Different configs should have different hashes
        assert hash1 != hash3
        
        # Hashes should be strings
        assert isinstance(hash1, str)
        assert len(hash1) == 16  # MD5 truncated to 16 chars
    
    def test_get_execution_statistics(self, mock_config, mock_components):
        """Test getting execution statistics"""
        engine = UnifiedExecutionEngine(mock_config)
        
        # Mock some data
        engine.cache.get_stats = Mock(return_value={'cache_hits': 10})
        engine.performance_monitor.get_comprehensive_report = Mock(return_value={'avg_time': 1.5})
        
        stats = engine.get_execution_statistics()
        
        assert 'cache_stats' in stats
        assert 'performance_stats' in stats
        assert 'available_strategies' in stats
        assert 'error_stats' in stats
        
        assert stats['cache_stats']['cache_hits'] == 10
        assert stats['performance_stats']['avg_time'] == 1.5


# Integration test
class TestExecutionEngineIntegration:
    """Integration tests for the execution engine"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_execution(self):
        """Test end-to-end execution with minimal mocking"""
        # This would be a more comprehensive test with real components
        # For now, we'll create a basic integration test
        
        config = UnifiedConfig()
        
        # Note: This test would require actual implementations
        # of the various components, so we'll keep it simple
        with patch('agentic_bte.unified.execution_engine.UnifiedEntityProcessor'), \
             patch('agentic_bte.unified.execution_engine.UnifiedPerformanceMonitor'), \
             patch('agentic_bte.unified.execution_engine.UnifiedStrategyRouter'):
            
            engine = UnifiedExecutionEngine(config)
            
            # Test that engine can be created without errors
            assert engine is not None
            assert len(engine.executor_registry.get_available_strategies()) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])