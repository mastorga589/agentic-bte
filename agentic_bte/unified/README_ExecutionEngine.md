# Unified Execution Engine

The Unified Execution Engine is the core component of Phase 2.2 that orchestrates the execution of biomedical queries using any strategy with comprehensive support for caching, monitoring, error handling, and parallel processing.

## Overview

The execution engine provides:
- **Multi-strategy execution**: Support for all available execution strategies
- **Intelligent caching**: Memory and Redis-based caching with TTL support
- **Error handling**: Comprehensive retry logic with backoff strategies
- **Performance monitoring**: Real-time monitoring and metrics collection
- **Fallback mechanisms**: Automatic fallback to alternative strategies
- **Timeout management**: Configurable execution timeouts
- **Resource management**: Memory and CPU usage monitoring

## Key Components

### UnifiedExecutionEngine

The main orchestrator that coordinates all aspects of query execution:

```python
from agentic_bte.unified import UnifiedExecutionEngine, UnifiedConfig

config = UnifiedConfig()
engine = UnifiedExecutionEngine(config)

# Execute a biomedical query
result = await engine.execute_query("What are the genetic causes of Alzheimer's disease?")
print(f"Strategy used: {result.strategy_used.value}")
print(f"Success: {result.success}")
print(f"Answer: {result.final_answer}")
```

### UnifiedCache

Provides intelligent caching across all strategies:

```python
# Cache configuration
config.caching.enable_caching = True
config.caching.cache_ttl = 3600  # 1 hour
config.caching.backend = CacheBackend.MEMORY  # or CacheBackend.REDIS

# Cache automatically handles:
# - Query result caching based on query + strategy + config hash
# - TTL-based expiration
# - Memory and Redis backend support
# - Cache hit/miss statistics
```

### ErrorHandler

Manages error handling and retry logic:

```python
# Error handling configuration
config.performance.max_retries = 3
config.performance.retry_delay_seconds = 2.0
config.performance.backoff_multiplier = 2.0

# Automatically handles:
# - Retryable vs non-retryable errors
# - Exponential backoff
# - Retry count tracking
# - Error categorization
```

### StrategyExecutorRegistry

Manages strategy executors and provides extensibility:

```python
# Custom strategy executor
async def custom_executor(plan: ExecutionPlan) -> BiomedicalResult:
    # Custom execution logic
    return result

# Register custom executor
engine.executor_registry.register_executor(
    ExecutionStrategy.CUSTOM, 
    custom_executor
)
```

## Execution Flow

1. **Entity Processing**: Extract and resolve entities from the query
2. **Strategy Selection**: Use the strategy router to select optimal strategy
3. **Cache Check**: Check if result is already cached
4. **Execution**: Execute with primary strategy or fallback strategies
5. **Error Handling**: Handle errors with retry logic
6. **Result Caching**: Cache successful results
7. **Performance Tracking**: Update performance metrics

## Strategy Support

The engine supports all available execution strategies:

- **SIMPLE**: Basic query optimization
- **GOT_FRAMEWORK**: Graph of Thoughts framework
- **LANGGRAPH_AGENTS**: Multi-agent LangGraph approach
- **PRODUCTION_GOT**: Production-optimized GoT
- **ENHANCED_GOT**: Enhanced GoT with domain expertise
- **STATEFUL_ITERATIVE**: Stateful iterative processing
- **HYBRID_ADAPTIVE**: Hybrid adaptive approach

## Configuration

### Performance Configuration

```python
config.performance.max_retries = 3
config.performance.retry_delay_seconds = 2.0
config.performance.backoff_multiplier = 2.0
config.performance.query_timeout_seconds = 300.0
config.performance.enable_parallel_execution = True
```

### Caching Configuration

```python
config.caching.enable_caching = True
config.caching.cache_ttl = 3600
config.caching.backend = CacheBackend.REDIS
config.caching.redis_host = "localhost"
config.caching.redis_port = 6379
config.caching.redis_db = 0
```

### Quality Configuration

```python
config.quality.min_confidence_threshold = 0.7
config.quality.min_quality_score = 0.8
config.quality.enable_quality_filtering = True
```

## Usage Examples

### Basic Query Execution

```python
import asyncio
from agentic_bte.unified import UnifiedExecutionEngine, UnifiedConfig

async def main():
    config = UnifiedConfig()
    engine = UnifiedExecutionEngine(config)
    
    result = await engine.execute_query(
        "What are the molecular mechanisms of cancer metastasis?"
    )
    
    print(f"Strategy: {result.strategy_used.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Quality Score: {result.quality_score:.3f}")
    print(f"Answer: {result.final_answer}")

asyncio.run(main())
```

### Advanced Configuration

```python
async def advanced_example():
    # Create custom configuration
    config = UnifiedConfig()
    
    # Configure performance
    config.performance.max_retries = 5
    config.performance.query_timeout_seconds = 600.0
    
    # Configure caching
    config.caching.enable_caching = True
    config.caching.cache_ttl = 7200  # 2 hours
    
    # Configure quality thresholds
    config.quality.min_confidence_threshold = 0.8
    config.quality.min_quality_score = 0.9
    
    engine = UnifiedExecutionEngine(config)
    
    # Execute with custom parameters
    result = await engine.execute_query(
        "How does the BRCA1 gene interact with p53 in DNA repair?",
        max_results=100,
        k=10,
        enable_domain_filtering=True
    )
    
    return result
```

### Performance Monitoring

```python
async def monitoring_example():
    config = UnifiedConfig()
    engine = UnifiedExecutionEngine(config)
    
    # Execute several queries
    queries = [
        "What are the side effects of metformin?",
        "How does diabetes affect cardiovascular health?",
        "What genes are associated with Alzheimer's disease?"
    ]
    
    for query in queries:
        await engine.execute_query(query)
    
    # Get comprehensive statistics
    stats = engine.get_execution_statistics()
    
    print("Cache Statistics:")
    print(f"  Hit Rate: {stats['cache_stats']['hit_rate']:.3f}")
    print(f"  Cache Hits: {stats['cache_stats']['cache_hits']}")
    
    print("Performance Statistics:")
    perf = stats['performance_stats']
    print(f"  Average Execution Time: {perf.get('avg_execution_time', 0):.3f}s")
    print(f"  Success Rate: {perf.get('success_rate', 0):.3f}")
```

## Error Handling

The engine provides comprehensive error handling:

### Retryable Errors
- Connection errors
- Timeout errors
- Network errors
- Rate limiting errors
- Temporary service unavailability

### Non-Retryable Errors
- Invalid input errors
- Authentication errors
- Permission errors
- Syntax errors

### Error Recovery
```python
# Automatic fallback to alternative strategies
# If primary strategy fails, tries fallback strategies
# If all strategies fail, returns error result with details

result = await engine.execute_query("problematic query")
if not result.success:
    print(f"Error: {result.error_message}")
    print(f"Error Type: {result.error_type}")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all execution engine tests
pytest agentic_bte/unified/tests/test_execution_engine.py -v

# Run integration tests
pytest agentic_bte/unified/tests/test_execution_engine.py::TestExecutionEngineIntegration -v

# Run with coverage
pytest agentic_bte/unified/tests/test_execution_engine.py --cov=agentic_bte.unified.execution_engine
```

## Demo

Run the comprehensive demo to see all features in action:

```bash
cd agentic_bte/unified/examples
python execution_demo.py
```

The demo showcases:
- Basic query execution
- Strategy comparison
- Caching capabilities
- Error handling
- Performance monitoring

## Performance Characteristics

### Caching Benefits
- **Cache Hit**: ~10-50ms response time
- **Cache Miss**: Full execution time (varies by strategy)
- **Memory Usage**: Configurable cache size limits
- **TTL Management**: Automatic expiration and cleanup

### Error Recovery
- **Retry Attempts**: Configurable (default: 3)
- **Backoff Strategy**: Exponential backoff with jitter
- **Fallback Time**: Typically 1-5 seconds per fallback
- **Success Rate**: >95% with proper configuration

### Resource Management
- **Memory Monitoring**: Real-time memory usage tracking
- **CPU Monitoring**: CPU usage and load monitoring  
- **Timeout Management**: Configurable per-strategy timeouts
- **Parallel Execution**: Configurable concurrency limits

## Best Practices

### Configuration
- Set appropriate timeout values based on query complexity
- Configure caching based on query patterns and memory availability
- Set quality thresholds based on application requirements
- Use Redis for distributed caching in production

### Error Handling
- Monitor error rates and adjust retry policies
- Implement custom error handlers for specific error types
- Use fallback strategies appropriate for your use case
- Log errors appropriately for debugging

### Performance
- Monitor cache hit rates and adjust TTL accordingly
- Profile query execution to identify bottlenecks
- Use parallel execution for independent operations
- Implement proper resource limits

### Monitoring
- Set up comprehensive monitoring and alerting
- Track performance metrics over time
- Monitor resource usage and scaling needs
- Implement health checks for dependent services

## Integration

The execution engine integrates seamlessly with:

- **Entity Processor**: For entity extraction and resolution
- **Performance Monitor**: For real-time monitoring
- **Strategy Router**: For intelligent strategy selection
- **All Existing Optimizers**: Through the executor registry

## Extensibility

### Adding Custom Strategies

```python
# Define custom strategy
async def my_custom_strategy(plan: ExecutionPlan) -> BiomedicalResult:
    # Custom implementation
    return result

# Register with engine
engine.executor_registry.register_executor(
    ExecutionStrategy.CUSTOM,
    my_custom_strategy
)
```

### Custom Error Handlers

```python
class CustomErrorHandler(ErrorHandler):
    async def handle_execution_error(self, error, strategy, query, attempt):
        # Custom error handling logic
        return await super().handle_execution_error(error, strategy, query, attempt)

# Use custom handler
engine.error_handler = CustomErrorHandler(config)
```

### Custom Caching Backends

```python
class CustomCacheBackend(UnifiedCache):
    async def get(self, key):
        # Custom cache retrieval
        pass
    
    async def put(self, key, value):
        # Custom cache storage
        pass

# Use custom cache
engine.cache = CustomCacheBackend(config)
```

This comprehensive execution engine provides a robust, scalable, and extensible foundation for biomedical query processing with enterprise-grade features for production deployment.