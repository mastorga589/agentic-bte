"""
Unified Parallel Executor Demo

This script demonstrates the capabilities of the unified parallel executor
including concurrent predicate execution, parallel API calls, batch processing,
and intelligent resource management.
"""

import asyncio
import logging
import time
import random
from typing import List, Dict, Any

from ..config import UnifiedConfig
from ..types import EntityContext, Entity, EntityType
from ..parallel_executor import (
    UnifiedParallelExecutor, ExecutionTask, ExecutionBatch, 
    ExecutionMode, TaskPriority
)
from ..knowledge_manager import TRAPIQuery, UnifiedKnowledgeManager
from ...core.knowledge.predicate_strategy import QueryIntent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ParallelExecutorDemo:
    """Demo class for showcasing parallel executor capabilities"""
    
    def __init__(self):
        # Initialize configuration
        self.config = UnifiedConfig()
        
        # Configure for demo purposes
        self.config.performance.max_concurrent_queries = 8
        self.config.performance.max_concurrent_api_calls = 12
        self.config.performance.max_worker_threads = 6
        self.config.performance.max_worker_processes = 3
        self.config.performance.query_timeout_seconds = 60.0
        self.config.performance.api_timeout_seconds = 30.0
        
        # Initialize parallel executor
        self.parallel_executor = UnifiedParallelExecutor(self.config)
        
        logger.info("Demo parallel executor initialized")
    
    async def run_basic_concurrent_execution_demo(self):
        """Demonstrate basic concurrent task execution"""
        logger.info("=" * 60)
        logger.info("BASIC CONCURRENT EXECUTION DEMO")
        logger.info("=" * 60)
        
        # Create simple computational tasks
        async def compute_fibonacci(n):
            """Simulate computational work"""
            await asyncio.sleep(0.01)  # Simulate some async work
            if n <= 1:
                return n
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
        
        async def compute_factorial(n):
            """Simulate another computational task"""
            await asyncio.sleep(0.01)
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        
        async def compute_squares(numbers):
            """Compute squares of numbers"""
            await asyncio.sleep(0.01)
            return [x * x for x in numbers]
        
        # Create execution tasks
        tasks = [
            ExecutionTask("fib_10", "computation", compute_fibonacci, args=(10,), priority=TaskPriority.HIGH),
            ExecutionTask("fact_8", "computation", compute_factorial, args=(8,), priority=TaskPriority.MEDIUM),
            ExecutionTask("squares", "computation", compute_squares, args=([1, 2, 3, 4, 5],), priority=TaskPriority.LOW),
            ExecutionTask("fib_15", "computation", compute_fibonacci, args=(15,), priority=TaskPriority.HIGH),
            ExecutionTask("fact_6", "computation", compute_factorial, args=(6,), priority=TaskPriority.MEDIUM)
        ]
        
        # Create execution batch
        batch = ExecutionBatch(
            "basic_concurrent_demo",
            tasks,
            ExecutionMode.CONCURRENT,
            max_concurrency=3
        )
        
        logger.info(f"Executing {len(tasks)} tasks concurrently (max concurrency: 3)")
        start_time = time.time()
        
        result = await self.parallel_executor.execute_batch(batch)
        
        execution_time = time.time() - start_time
        
        logger.info(f"\nBatch execution completed in {execution_time:.3f}s")
        logger.info(f"Success rate: {result.success_rate:.2%}")
        logger.info(f"Total execution time: {result.total_execution_time:.3f}s")
        
        # Show individual results
        logger.info("\nTask Results:")
        for task_result in result.task_results:
            status = "✓" if task_result.success else "✗"
            logger.info(f"  {status} {task_result.task_id}: {task_result.result} "
                       f"({task_result.execution_time:.3f}s)")
        
        return result
    
    async def run_parallel_predicates_demo(self):
        """Demonstrate parallel predicate execution"""
        logger.info("=" * 60)
        logger.info("PARALLEL PREDICATES DEMO")
        logger.info("=" * 60)
        
        # Create sample TRAPI queries
        sample_queries = []
        predicates = ["biolink:treats", "biolink:affects", "biolink:associated_with", "biolink:related_to"]
        
        for i, predicate in enumerate(predicates):
            query = TRAPIQuery(
                query_graph={
                    "nodes": {"n0": {"categories": ["biolink:Disease"]}, "n1": {"categories": ["biolink:SmallMolecule"]}},
                    "edges": {"e01": {"subject": "n0", "object": "n1", "predicates": [predicate]}}
                },
                query_id=f"query_{i}",
                predicate=predicate,
                entities=["diabetes", "drugs"],
                confidence=0.8,
                source_intent=QueryIntent.THERAPEUTIC,
                estimated_results=50 * (i + 1)
            )
            sample_queries.append(query)
        
        # Mock execution function that simulates API calls
        async def mock_trapi_execution(query: TRAPIQuery):
            # Simulate variable execution times
            delay = random.uniform(0.1, 0.5)
            await asyncio.sleep(delay)
            
            # Simulate some queries failing occasionally
            if random.random() < 0.1:  # 10% failure rate
                raise Exception(f"Simulated API failure for {query.predicate}")
            
            # Return mock results
            return {
                "query_id": query.query_id,
                "predicate": query.predicate,
                "results_count": random.randint(10, query.estimated_results),
                "execution_time": delay
            }
        
        logger.info(f"Executing {len(sample_queries)} TRAPI queries in parallel")
        
        # Test different execution modes
        for mode in [ExecutionMode.CONCURRENT, ExecutionMode.THREADED]:
            logger.info(f"\nTesting {mode.value} execution mode:")
            start_time = time.time()
            
            results = await self.parallel_executor.execute_parallel_predicates(
                sample_queries,
                mock_trapi_execution,
                execution_mode=mode,
                max_concurrency=3
            )
            
            execution_time = time.time() - start_time
            
            successful_results = [r for r in results if r[1] is not None]
            logger.info(f"  Completed in {execution_time:.3f}s")
            logger.info(f"  Successful queries: {len(successful_results)}/{len(sample_queries)}")
            
            # Show results
            for query, result in successful_results[:3]:  # Show first 3 results
                if result:
                    logger.info(f"    {query.predicate}: {result['results_count']} results")
        
        return results
    
    async def run_parallel_api_calls_demo(self):
        """Demonstrate parallel API call execution"""
        logger.info("=" * 60)
        logger.info("PARALLEL API CALLS DEMO")
        logger.info("=" * 60)
        
        # Create mock API calls with different characteristics
        async def fast_api_call():
            await asyncio.sleep(0.1)
            return {"api": "fast", "data": "quick response"}
        
        async def slow_api_call():
            await asyncio.sleep(0.5)
            return {"api": "slow", "data": "detailed response"}
        
        async def unreliable_api_call():
            await asyncio.sleep(0.2)
            if random.random() < 0.3:  # 30% failure rate
                raise Exception("API temporarily unavailable")
            return {"api": "unreliable", "data": "eventually consistent"}
        
        async def timeout_prone_api_call():
            delay = random.uniform(0.1, 1.0)
            await asyncio.sleep(delay)
            return {"api": "timeout_prone", "data": f"responded after {delay:.2f}s"}
        
        # Create multiple instances of each API call
        api_calls = []
        for i in range(3):
            api_calls.extend([
                fast_api_call,
                slow_api_call,
                unreliable_api_call,
                timeout_prone_api_call
            ])
        
        logger.info(f"Executing {len(api_calls)} API calls in parallel")
        
        # Test with different concurrency limits
        concurrency_limits = [2, 4, 8]
        
        for max_concurrency in concurrency_limits:
            logger.info(f"\nTesting with max concurrency: {max_concurrency}")
            start_time = time.time()
            
            results = await self.parallel_executor.execute_parallel_api_calls(
                api_calls,
                ExecutionMode.CONCURRENT,
                max_concurrency=max_concurrency
            )
            
            execution_time = time.time() - start_time
            
            logger.info(f"  Completed in {execution_time:.3f}s")
            logger.info(f"  Successful calls: {len(results)}/{len(api_calls)}")
            
            # Count API types in results
            api_types = {}
            for result in results:
                api_type = result['api']
                api_types[api_type] = api_types.get(api_type, 0) + 1
            
            logger.info("  Results by API type:")
            for api_type, count in api_types.items():
                logger.info(f"    {api_type}: {count}")
        
        return results
    
    async def run_execution_mode_comparison_demo(self):
        """Compare different execution modes"""
        logger.info("=" * 60)
        logger.info("EXECUTION MODE COMPARISON DEMO")
        logger.info("=" * 60)
        
        # Create a mix of I/O and CPU intensive tasks
        async def io_task(task_id, delay=0.1):
            await asyncio.sleep(delay)
            return f"IO task {task_id} completed"
        
        def cpu_task(task_id, iterations=100000):
            # CPU intensive task
            result = 0
            for i in range(iterations):
                result += i * i
            return f"CPU task {task_id} completed with result {result}"
        
        # Create mixed workload
        tasks = []
        for i in range(6):
            if i % 2 == 0:
                # I/O task
                task = ExecutionTask(
                    f"io_task_{i}",
                    "api_call",
                    lambda tid=i: io_task(tid, 0.2),
                    priority=TaskPriority.MEDIUM
                )
            else:
                # CPU task
                task = ExecutionTask(
                    f"cpu_task_{i}",
                    "computation",
                    lambda tid=i: cpu_task(tid, 50000),
                    priority=TaskPriority.MEDIUM
                )
            tasks.append(task)
        
        # Test different execution modes
        modes = [
            ExecutionMode.SEQUENTIAL,
            ExecutionMode.CONCURRENT,
            ExecutionMode.THREADED,
            ExecutionMode.HYBRID
        ]
        
        results = {}
        
        for mode in modes:
            logger.info(f"\nTesting {mode.value} execution mode:")
            
            batch = ExecutionBatch(
                f"comparison_{mode.value}",
                tasks.copy(),  # Copy to avoid issues with task reuse
                mode,
                max_concurrency=4
            )
            
            start_time = time.time()
            result = await self.parallel_executor.execute_batch(batch)
            execution_time = time.time() - start_time
            
            results[mode.value] = {
                'execution_time': execution_time,
                'batch_execution_time': result.total_execution_time,
                'success_rate': result.success_rate,
                'successful_tasks': len([r for r in result.task_results if r.success])
            }
            
            logger.info(f"  Wall clock time: {execution_time:.3f}s")
            logger.info(f"  Batch execution time: {result.total_execution_time:.3f}s")
            logger.info(f"  Success rate: {result.success_rate:.2%}")
            logger.info(f"  Speedup vs sequential: {results['sequential']['execution_time'] / execution_time:.2f}x" 
                       if mode != ExecutionMode.SEQUENTIAL else "  (baseline)")
        
        # Summary comparison
        logger.info("\nExecution Mode Comparison Summary:")
        logger.info("Mode           | Wall Time | Success Rate | Speedup")
        logger.info("-" * 55)
        
        sequential_time = results['sequential']['execution_time']
        for mode_name, data in results.items():
            speedup = sequential_time / data['execution_time'] if mode_name != 'sequential' else 1.0
            logger.info(f"{mode_name:13} | {data['execution_time']:8.3f}s | {data['success_rate']:10.2%} | {speedup:6.2f}x")
        
        return results
    
    async def run_resource_monitoring_demo(self):
        """Demonstrate resource monitoring and throttling"""
        logger.info("=" * 60)
        logger.info("RESOURCE MONITORING DEMO")
        logger.info("=" * 60)
        
        # Create resource-intensive tasks
        def memory_intensive_task(size_mb):
            # Create a large list to consume memory
            data = [i for i in range(size_mb * 1000)]  # Approximate MB
            return f"Processed {len(data)} items"
        
        async def cpu_intensive_task(duration):
            # CPU intensive task with async yield points
            start_time = time.time()
            result = 0
            while time.time() - start_time < duration:
                for _ in range(10000):
                    result += 1
                await asyncio.sleep(0.001)  # Yield control
            return f"CPU task completed: {result} iterations"
        
        # Create tasks with different resource requirements
        tasks = [
            ExecutionTask("memory_1", "computation", memory_intensive_task, args=(10,)),
            ExecutionTask("cpu_1", "computation", cpu_intensive_task, args=(0.2,)),
            ExecutionTask("memory_2", "computation", memory_intensive_task, args=(5,)),
            ExecutionTask("cpu_2", "computation", cpu_intensive_task, args=(0.1,)),
            ExecutionTask("memory_3", "computation", memory_intensive_task, args=(15,)),
            ExecutionTask("cpu_3", "computation", cpu_intensive_task, args=(0.3,))
        ]
        
        # Monitor resource usage during execution
        logger.info("Starting resource monitoring...")
        
        batch = ExecutionBatch(
            "resource_monitoring_test",
            tasks,
            ExecutionMode.CONCURRENT,
            max_concurrency=3
        )
        
        # Get initial resource usage
        initial_usage = self.parallel_executor.resource_monitor.get_current_usage()
        logger.info(f"Initial resource usage:")
        logger.info(f"  CPU: {initial_usage['cpu_percent']:.1f}%")
        logger.info(f"  Memory: {initial_usage['memory_percent']:.1f}%")
        logger.info(f"  Threads: {initial_usage['active_threads']}")
        
        start_time = time.time()
        result = await self.parallel_executor.execute_batch(batch)
        execution_time = time.time() - start_time
        
        # Get final resource usage
        final_usage = self.parallel_executor.resource_monitor.get_current_usage()
        logger.info(f"\nFinal resource usage:")
        logger.info(f"  CPU: {final_usage['cpu_percent']:.1f}%")
        logger.info(f"  Memory: {final_usage['memory_percent']:.1f}%")
        logger.info(f"  Threads: {final_usage['active_threads']}")
        
        logger.info(f"\nExecution completed in {execution_time:.3f}s")
        logger.info(f"Success rate: {result.success_rate:.2%}")
        
        # Show resource usage from batch result
        if result.resource_summary:
            logger.info(f"\nResource summary from batch:")
            for resource, value in result.resource_summary.items():
                logger.info(f"  {resource}: {value}")
        
        return result
    
    async def run_error_handling_and_recovery_demo(self):
        """Demonstrate error handling and recovery mechanisms"""
        logger.info("=" * 60)
        logger.info("ERROR HANDLING AND RECOVERY DEMO")
        logger.info("=" * 60)
        
        # Create tasks with different failure modes
        async def reliable_task(task_id):
            await asyncio.sleep(0.1)
            return f"Task {task_id} completed successfully"
        
        async def unreliable_task(task_id, failure_rate=0.5):
            await asyncio.sleep(0.1)
            if random.random() < failure_rate:
                raise Exception(f"Task {task_id} failed randomly")
            return f"Task {task_id} succeeded against odds"
        
        async def timeout_task(task_id, delay=2.0):
            await asyncio.sleep(delay)
            return f"Task {task_id} completed after {delay}s"
        
        async def memory_error_task(task_id):
            await asyncio.sleep(0.1)
            raise MemoryError(f"Task {task_id} ran out of memory")
        
        # Create mixed batch with different error types
        tasks = [
            ExecutionTask("reliable_1", "api_call", reliable_task, args=("reliable_1",)),
            ExecutionTask("unreliable_1", "api_call", unreliable_task, args=("unreliable_1", 0.7)),
            ExecutionTask("reliable_2", "api_call", reliable_task, args=("reliable_2",)),
            ExecutionTask("timeout_1", "api_call", timeout_task, args=("timeout_1", 1.0), timeout=0.5),
            ExecutionTask("unreliable_2", "api_call", unreliable_task, args=("unreliable_2", 0.3)),
            ExecutionTask("memory_error_1", "computation", memory_error_task, args=("memory_error_1",)),
            ExecutionTask("reliable_3", "api_call", reliable_task, args=("reliable_3",))
        ]
        
        # Test with different error tolerance levels
        tolerance_levels = [0.0, 0.3, 0.5, 1.0]  # 0%, 30%, 50%, 100% error tolerance
        
        for tolerance in tolerance_levels:
            logger.info(f"\nTesting with {tolerance:.0%} error tolerance:")
            
            batch = ExecutionBatch(
                f"error_test_{tolerance}",
                tasks.copy(),
                ExecutionMode.CONCURRENT,
                max_concurrency=4,
                error_tolerance=tolerance
            )
            
            start_time = time.time()
            result = await self.parallel_executor.execute_batch(batch)
            execution_time = time.time() - start_time
            
            successful_tasks = [r for r in result.task_results if r.success]
            failed_tasks = [r for r in result.task_results if not r.success]
            
            logger.info(f"  Execution time: {execution_time:.3f}s")
            logger.info(f"  Total tasks: {len(result.task_results)}")
            logger.info(f"  Successful: {len(successful_tasks)}")
            logger.info(f"  Failed: {len(failed_tasks)}")
            logger.info(f"  Success rate: {result.success_rate:.2%}")
            
            # Show error types
            if failed_tasks:
                error_types = {}
                for task_result in failed_tasks:
                    error_type = type(task_result.error).__name__
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                logger.info("  Error breakdown:")
                for error_type, count in error_types.items():
                    logger.info(f"    {error_type}: {count}")
        
        return result
    
    async def run_intelligent_mode_selection_demo(self):
        """Demonstrate automatic execution mode selection"""
        logger.info("=" * 60)
        logger.info("INTELLIGENT MODE SELECTION DEMO")
        logger.info("=" * 60)
        
        # Create different workload patterns
        workload_patterns = {
            "mostly_io": [
                ExecutionTask(f"api_call_{i}", "api_call", lambda: None) for i in range(8)
            ] + [
                ExecutionTask(f"cpu_task_{i}", "computation", lambda: None) for i in range(2)
            ],
            
            "mostly_cpu": [
                ExecutionTask(f"cpu_task_{i}", "computation", lambda: None) for i in range(8)
            ] + [
                ExecutionTask(f"api_call_{i}", "api_call", lambda: None) for i in range(2)
            ],
            
            "mixed_workload": [
                ExecutionTask(f"api_call_{i}", "api_call", lambda: None) for i in range(3)
            ] + [
                ExecutionTask(f"cpu_task_{i}", "computation", lambda: None) for i in range(3)
            ] + [
                ExecutionTask(f"db_task_{i}", "database", lambda: None) for i in range(2)
            ] + [
                ExecutionTask(f"analysis_task_{i}", "analysis", lambda: None) for i in range(2)
            ],
            
            "single_task": [
                ExecutionTask("single_task", "computation", lambda: None)
            ],
            
            "few_tasks": [
                ExecutionTask("task_1", "api_call", lambda: None),
                ExecutionTask("task_2", "computation", lambda: None)
            ]
        }
        
        logger.info("Testing intelligent execution mode selection...")
        
        for pattern_name, tasks in workload_patterns.items():
            logger.info(f"\nWorkload pattern: {pattern_name}")
            logger.info(f"  Tasks: {len(tasks)}")
            
            # Count task types
            task_types = {}
            for task in tasks:
                task_type = task.task_type
                task_types[task_type] = task_types.get(task_type, 0) + 1
            
            logger.info("  Task breakdown:")
            for task_type, count in task_types.items():
                logger.info(f"    {task_type}: {count}")
            
            # Get recommended execution mode
            recommended_mode = self.parallel_executor.optimize_execution_mode(tasks)
            logger.info(f"  Recommended mode: {recommended_mode.value}")
            
            # Explain the recommendation
            if len(tasks) == 1:
                logger.info("    Reason: Single task - sequential execution is optimal")
            elif len(tasks) <= 3:
                logger.info("    Reason: Few tasks - concurrent execution is sufficient")
            elif task_types.get("api_call", 0) / len(tasks) > 0.7:
                logger.info("    Reason: I/O heavy workload - async concurrency is optimal")
            elif task_types.get("computation", 0) / len(tasks) > 0.7:
                logger.info("    Reason: CPU heavy workload - thread/process pool is optimal")
            elif len(task_types) > 2:
                logger.info("    Reason: Mixed workload - hybrid approach is optimal")
            else:
                logger.info("    Reason: Balanced workload - concurrent execution is suitable")
        
        return workload_patterns
    
    def run_performance_statistics_demo(self):
        """Show comprehensive performance statistics"""
        logger.info("=" * 60)
        logger.info("PERFORMANCE STATISTICS DEMO")
        logger.info("=" * 60)
        
        # Get current execution statistics
        stats = self.parallel_executor.get_execution_statistics()
        
        logger.info("Unified Parallel Executor Statistics:")
        logger.info(f"  Total tasks executed: {stats['total_tasks_executed']}")
        logger.info(f"  Successful tasks: {stats['successful_tasks']}")
        logger.info(f"  Failed tasks: {stats['failed_tasks']}")
        logger.info(f"  Overall success rate: {stats['success_rate']:.2%}")
        logger.info(f"  Total execution time: {stats['total_execution_time']:.3f}s")
        logger.info(f"  Average task execution time: {stats['average_execution_time']:.3f}s")
        
        logger.info("\nExecution mode breakdown:")
        logger.info(f"  Concurrent executions: {stats['concurrent_executions']}")
        logger.info(f"  Thread executions: {stats['thread_executions']}")
        logger.info(f"  Process executions: {stats['process_executions']}")
        
        logger.info("\nCurrent resource usage:")
        resource_usage = stats['current_resource_usage']
        for resource, value in resource_usage.items():
            if isinstance(value, float):
                logger.info(f"  {resource}: {value:.1f}{'%' if 'percent' in resource else ''}")
            else:
                logger.info(f"  {resource}: {value}")
        
        logger.info("\nTask scheduler status:")
        scheduler_status = stats['scheduler_status']
        for status, count in scheduler_status.items():
            logger.info(f"  {status}: {count}")
        
        return stats
    
    async def run_full_demo(self):
        """Run all parallel executor demo scenarios"""
        logger.info("🚀 Starting Unified Parallel Executor Demo")
        logger.info("=" * 80)
        
        demo_results = {}
        
        try:
            # Basic concurrent execution
            demo_results['basic_concurrent'] = await self.run_basic_concurrent_execution_demo()
            
            # Parallel predicate execution
            demo_results['parallel_predicates'] = await self.run_parallel_predicates_demo()
            
            # Parallel API calls
            demo_results['parallel_api_calls'] = await self.run_parallel_api_calls_demo()
            
            # Execution mode comparison
            demo_results['mode_comparison'] = await self.run_execution_mode_comparison_demo()
            
            # Resource monitoring
            demo_results['resource_monitoring'] = await self.run_resource_monitoring_demo()
            
            # Error handling and recovery
            demo_results['error_handling'] = await self.run_error_handling_and_recovery_demo()
            
            # Intelligent mode selection
            demo_results['intelligent_selection'] = await self.run_intelligent_mode_selection_demo()
            
            # Performance statistics
            demo_results['performance_stats'] = self.run_performance_statistics_demo()
            
            # Final summary
            self._print_demo_summary(demo_results)
            
        except Exception as e:
            logger.error(f"Demo failed with error: {str(e)}")
            raise
        finally:
            # Cleanup resources
            self.parallel_executor.shutdown()
        
        return demo_results
    
    def _print_demo_summary(self, demo_results: Dict[str, Any]):
        """Print summary of all demo results"""
        logger.info("=" * 80)
        logger.info("DEMO SUMMARY")
        logger.info("=" * 80)
        
        logger.info("Unified Parallel Executor demonstrated:")
        logger.info("  ✅ Basic concurrent task execution")
        logger.info("  ✅ Parallel TRAPI predicate execution")
        logger.info("  ✅ Parallel API call management")
        logger.info("  ✅ Multiple execution modes (concurrent, threaded, multiprocess, hybrid)")
        logger.info("  ✅ Intelligent execution mode selection")
        logger.info("  ✅ Resource monitoring and throttling")
        logger.info("  ✅ Comprehensive error handling and recovery")
        logger.info("  ✅ Task scheduling with priority and dependencies")
        logger.info("  ✅ Performance statistics and monitoring")
        
        # Show key performance metrics
        perf_stats = demo_results.get('performance_stats', {})
        if perf_stats:
            logger.info(f"\nFinal Performance Metrics:")
            logger.info(f"  Total tasks processed: {perf_stats['total_tasks_executed']}")
            logger.info(f"  Overall success rate: {perf_stats['success_rate']:.2%}")
            logger.info(f"  Average execution time: {perf_stats['average_execution_time']:.3f}s")
        
        # Show execution mode performance comparison
        mode_comparison = demo_results.get('mode_comparison', {})
        if mode_comparison:
            logger.info(f"\nBest performing execution mode comparison:")
            best_mode = min(mode_comparison.keys(), key=lambda k: mode_comparison[k]['execution_time'])
            best_time = mode_comparison[best_mode]['execution_time']
            logger.info(f"  Fastest mode: {best_mode} ({best_time:.3f}s)")
        
        logger.info("\n🎉 Parallel Executor Demo completed successfully!")


async def main():
    """Main demo function"""
    try:
        demo = ParallelExecutorDemo()
        await demo.run_full_demo()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())