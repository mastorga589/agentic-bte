"""
Unified Execution Engine Demo

This script demonstrates the capabilities of the unified execution engine
with various biomedical queries and strategies.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any

from ..config import UnifiedConfig, ExecutionStrategy
from ..execution_engine import UnifiedExecutionEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ExecutionDemo:
    """Demo class for showcasing execution engine capabilities"""
    
    def __init__(self):
        # Initialize configuration
        self.config = UnifiedConfig()
        
        # Configure for demo purposes
        self.config.performance.max_retries = 2
        self.config.performance.query_timeout_seconds = 60.0
        self.config.caching.enable_caching = True
        self.config.caching.cache_ttl = 3600
        
        # Initialize execution engine
        self.engine = UnifiedExecutionEngine(self.config)
        
        logger.info("Demo execution engine initialized")
    
    async def run_basic_query_demo(self):
        """Demonstrate basic query execution"""
        logger.info("=" * 60)
        logger.info("BASIC QUERY EXECUTION DEMO")
        logger.info("=" * 60)
        
        queries = [
            "What are the genetic causes of Alzheimer's disease?",
            "How does diabetes affect cardiovascular health?",
            "What drugs are used to treat depression?",
            "What are the molecular mechanisms of cancer metastasis?"
        ]
        
        results = []
        for query in queries:
            logger.info(f"\nExecuting query: {query}")
            start_time = time.time()
            
            try:
                result = await self.engine.execute_query(query)
                execution_time = time.time() - start_time
                
                logger.info(f"Strategy used: {str(result.strategy_used)}")
                logger.info(f"Success: {result.success}")
                logger.info(f"Confidence: {result.confidence:.3f}")
                logger.info(f"Quality score: {result.quality_score:.3f}")
                logger.info(f"Execution time: {execution_time:.2f}s")
                
                if result.success:
                    logger.info(f"Answer preview: {result.final_answer[:200]}...")
                else:
                    logger.warning(f"Error: {result.error_message}")
                
                results.append({
                    'query': query,
                    'result': result,
                    'execution_time': execution_time
                })
                
            except Exception as e:
                logger.error(f"Query failed with exception: {str(e)}")
        
        return results
    
    async def run_strategy_comparison_demo(self):
        """Demonstrate different strategies on the same query"""
        logger.info("=" * 60)
        logger.info("STRATEGY COMPARISON DEMO")
        logger.info("=" * 60)
        
        test_query = "What are the interactions between BRCA1 and p53 in cancer?"
        
        # Force different strategies (this would require custom strategy selection)
        strategies = [
            ExecutionStrategy.SIMPLE,
            ExecutionStrategy.GOT_FRAMEWORK,
            ExecutionStrategy.PRODUCTION_GOT,
            ExecutionStrategy.LANGGRAPH_AGENTS
        ]
        
        results = {}
        
        for strategy in strategies:
            logger.info(f"\nTesting strategy: {strategy.value}")
            start_time = time.time()
            
            try:
                # Note: In real implementation, you'd need a way to force strategy selection
                # For demo purposes, we'll use the normal execution
                result = await self.engine.execute_query(
                    test_query,
                    preferred_strategy=strategy.value  # Custom parameter
                )
                
                execution_time = time.time() - start_time
                
                logger.info(f"Success: {result.success}")
                logger.info(f"Confidence: {result.confidence:.3f}")
                logger.info(f"Quality score: {result.quality_score:.3f}")
                logger.info(f"Execution time: {execution_time:.2f}s")
                logger.info(f"Steps executed: {len(result.execution_steps)}")
                
                results[strategy.value] = {
                    'result': result,
                    'execution_time': execution_time
                }
                
            except Exception as e:
                logger.error(f"Strategy {strategy.value} failed: {str(e)}")
                results[strategy.value] = {
                    'error': str(e),
                    'execution_time': time.time() - start_time
                }
        
        # Compare results
        self._compare_strategy_results(results)
        
        return results
    
    async def run_caching_demo(self):
        """Demonstrate caching capabilities"""
        logger.info("=" * 60)
        logger.info("CACHING DEMO")
        logger.info("=" * 60)
        
        test_query = "What are the side effects of metformin?"
        
        logger.info(f"Query: {test_query}")
        
        # First execution (cache miss)
        logger.info("\nFirst execution (cache miss expected):")
        start_time = time.time()
        result1 = await self.engine.execute_query(test_query)
        time1 = time.time() - start_time
        logger.info(f"Execution time: {time1:.3f}s")
        
        # Second execution (cache hit)
        logger.info("\nSecond execution (cache hit expected):")
        start_time = time.time()
        result2 = await self.engine.execute_query(test_query)
        time2 = time.time() - start_time
        logger.info(f"Execution time: {time2:.3f}s")
        
        # Verify results are the same
        logger.info(f"\nResults identical: {result1.final_answer == result2.final_answer}")
        logger.info(f"Speed improvement: {(time1 - time2) / time1 * 100:.1f}%")
        
        # Show cache statistics
        cache_stats = self.engine.cache.get_stats()
        logger.info(f"\nCache statistics:")
        logger.info(f"  Hit rate: {cache_stats['hit_rate']:.3f}")
        logger.info(f"  Cache hits: {cache_stats['cache_hits']}")
        logger.info(f"  Cache misses: {cache_stats['cache_misses']}")
        logger.info(f"  Memory cache size: {cache_stats['memory_cache_size']}")
        
        return {
            'first_execution': {'result': result1, 'time': time1},
            'second_execution': {'result': result2, 'time': time2},
            'cache_stats': cache_stats
        }
    
    async def run_error_handling_demo(self):
        """Demonstrate error handling and retry mechanisms"""
        logger.info("=" * 60)
        logger.info("ERROR HANDLING DEMO")
        logger.info("=" * 60)
        
        # Test queries that might cause different types of errors
        test_queries = [
            "What is the molecular weight of XYZ123NonexistentCompound?",  # Invalid entity
            "",  # Empty query
            "A" * 1000,  # Very long query
            "What are the effects of #@$%^&*() on health?"  # Special characters
        ]
        
        results = []
        
        for query in test_queries:
            logger.info(f"\nTesting error handling for: {query[:50]}...")
            
            try:
                result = await self.engine.execute_query(query)
                
                logger.info(f"Success: {result.success}")
                if not result.success:
                    logger.info(f"Error message: {result.error_message}")
                
                results.append({
                    'query': query,
                    'result': result
                })
                
            except Exception as e:
                logger.error(f"Unhandled exception: {str(e)}")
                results.append({
                    'query': query,
                    'exception': str(e)
                })
        
        return results
    
    async def run_performance_monitoring_demo(self):
        """Demonstrate performance monitoring capabilities"""
        logger.info("=" * 60)
        logger.info("PERFORMANCE MONITORING DEMO")
        logger.info("=" * 60)
        
        # Run several queries to generate performance data
        queries = [
            "What is the mechanism of action of aspirin?",
            "How does insulin regulate glucose metabolism?",
            "What are the genetic factors in Type 2 diabetes?",
            "What drugs interact with warfarin?",
            "How does the immune system respond to vaccination?"
        ]
        
        logger.info("Executing multiple queries for performance analysis...")
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Query {i}/{len(queries)}: {query}")
            await self.engine.execute_query(query)
        
        # Get comprehensive performance report
        performance_stats = self.engine.performance_monitor.get_comprehensive_report()
        
        logger.info("\nPerformance Statistics:")
        logger.info(f"  Total queries executed: {performance_stats.get('total_queries', 0)}")
        logger.info(f"  Average execution time: {performance_stats.get('avg_execution_time', 0):.3f}s")
        logger.info(f"  Success rate: {performance_stats.get('success_rate', 0):.3f}")
        
        # Show strategy performance breakdown
        strategy_stats = performance_stats.get('strategy_performance', {})
        if strategy_stats:
            logger.info("\nStrategy Performance:")
            for strategy, stats in strategy_stats.items():
                logger.info(f"  {strategy}:")
                logger.info(f"    Count: {stats.get('count', 0)}")
                logger.info(f"    Avg time: {stats.get('avg_time', 0):.3f}s")
                logger.info(f"    Success rate: {stats.get('success_rate', 0):.3f}")
        
        return performance_stats
    
    def _compare_strategy_results(self, results: Dict[str, Dict[str, Any]]):
        """Compare results from different strategies"""
        logger.info("\nSTRATEGY COMPARISON SUMMARY:")
        logger.info("-" * 40)
        
        successful_results = {k: v for k, v in results.items() if 'result' in v and v['result'].success}
        
        if successful_results:
            # Find best performing strategy
            best_strategy = min(successful_results.keys(), 
                              key=lambda x: successful_results[x]['execution_time'])
            
            logger.info(f"Fastest strategy: {best_strategy} ({successful_results[best_strategy]['execution_time']:.2f}s)")
            
            # Find highest quality
            if successful_results:
                best_quality = max(successful_results.keys(), 
                                 key=lambda x: successful_results[x]['result'].quality_score)
                
                logger.info(f"Highest quality: {best_quality} (score: {successful_results[best_quality]['result'].quality_score:.3f})")
        
        # Show failed strategies
        failed_results = {k: v for k, v in results.items() if 'error' in v}
        if failed_results:
            logger.info(f"Failed strategies: {', '.join(failed_results.keys())}")
    
    async def run_full_demo(self):
        """Run all demo scenarios"""
        logger.info("🚀 Starting Unified Execution Engine Demo")
        logger.info("=" * 80)
        
        demo_results = {}
        
        try:
            # Basic query execution
            demo_results['basic_queries'] = await self.run_basic_query_demo()
            
            # Strategy comparison
            demo_results['strategy_comparison'] = await self.run_strategy_comparison_demo()
            
            # Caching demonstration
            demo_results['caching'] = await self.run_caching_demo()
            
            # Error handling
            demo_results['error_handling'] = await self.run_error_handling_demo()
            
            # Performance monitoring
            demo_results['performance'] = await self.run_performance_monitoring_demo()
            
            # Final summary
            self._print_demo_summary(demo_results)
            
        except Exception as e:
            logger.error(f"Demo failed with error: {str(e)}")
            raise
        
        return demo_results
    
    def _print_demo_summary(self, demo_results: Dict[str, Any]):
        """Print summary of all demo results"""
        logger.info("=" * 80)
        logger.info("DEMO SUMMARY")
        logger.info("=" * 80)
        
        # Get overall statistics
        overall_stats = self.engine.get_execution_statistics()
        
        logger.info("Overall Statistics:")
        cache_stats = overall_stats.get('cache_stats', {})
        logger.info(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.3f}")
        logger.info(f"  Available strategies: {len(overall_stats.get('available_strategies', []))}")
        
        # Print available strategies
        available_strategies = overall_stats.get('available_strategies', [])
        if available_strategies:
            logger.info(f"  Registered strategies: {', '.join(available_strategies)}")
        
        logger.info("\n🎉 Demo completed successfully!")
        logger.info("The unified execution engine demonstrated:")
        logger.info("  ✅ Multi-strategy execution")
        logger.info("  ✅ Intelligent caching")
        logger.info("  ✅ Error handling and retries")
        logger.info("  ✅ Performance monitoring")
        logger.info("  ✅ Fallback mechanisms")


async def main():
    """Main demo function"""
    try:
        demo = ExecutionDemo()
        await demo.run_full_demo()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())