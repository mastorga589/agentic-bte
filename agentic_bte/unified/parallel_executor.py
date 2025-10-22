"""
Unified Parallel Execution Manager

This module provides comprehensive parallel execution capabilities for biomedical queries,
including concurrent predicate execution, parallel API calls, batch processing,
and resource-aware scheduling across all execution strategies.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import asynccontextmanager
import multiprocessing
import threading

from .config import UnifiedConfig
from .types import (
    BiomedicalResult, EntityContext, ExecutionContext, ExecutionStep,
    ExecutionStatus, PerformanceMetrics
)
from .knowledge_manager import TRAPIQuery

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Types of parallel execution modes"""
    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"        # Async concurrent execution
    THREADED = "threaded"           # Thread-based parallelism
    MULTIPROCESS = "multiprocess"   # Process-based parallelism
    HYBRID = "hybrid"               # Mixed concurrent + threaded


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ExecutionTask:
    """A single execution task"""
    task_id: str
    task_type: str  # predicate_execution, api_call, query_processing, etc.
    task_func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout: Optional[float] = None
    dependencies: Set[str] = field(default_factory=set)
    estimated_duration: float = 1.0  # seconds
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionBatch:
    """A batch of related execution tasks"""
    batch_id: str
    tasks: List[ExecutionTask]
    execution_mode: ExecutionMode
    max_concurrency: int = 10
    timeout: Optional[float] = None
    error_tolerance: float = 0.1  # Percentage of tasks that can fail


@dataclass
class TaskResult:
    """Result of a single task execution"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of a batch execution"""
    batch_id: str
    task_results: List[TaskResult]
    total_execution_time: float
    success_rate: float
    failed_tasks: List[str]
    resource_summary: Dict[str, Any]


class ResourceMonitor:
    """Monitor system resources during execution"""
    
    def __init__(self):
        self.cpu_threshold = 80.0  # Percentage
        self.memory_threshold = 80.0  # Percentage
        self._monitoring = False
        self._resource_history = []
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self._monitoring = True
        # In a real implementation, this would start a background thread
        # to monitor CPU, memory, etc.
        logger.debug("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring = False
        logger.debug("Resource monitoring stopped")
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'active_threads': threading.active_count(),
                'active_processes': len(psutil.pids())
            }
        except ImportError:
            # Fallback if psutil is not available
            return {
                'cpu_percent': 50.0,  # Estimated
                'memory_percent': 60.0,  # Estimated
                'active_threads': threading.active_count(),
                'active_processes': multiprocessing.cpu_count()
            }
    
    def should_throttle(self) -> bool:
        """Check if execution should be throttled due to high resource usage"""
        usage = self.get_current_usage()
        return (usage['cpu_percent'] > self.cpu_threshold or 
                usage['memory_percent'] > self.memory_threshold)


class TaskScheduler:
    """Intelligent task scheduler with dependency resolution"""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.pending_tasks: Dict[str, ExecutionTask] = {}
        self.completed_tasks: Set[str] = set()
        self.running_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
    
    def add_task(self, task: ExecutionTask):
        """Add a task to the scheduler"""
        self.pending_tasks[task.task_id] = task
        logger.debug(f"Added task {task.task_id} to scheduler")
    
    def get_ready_tasks(self) -> List[ExecutionTask]:
        """Get tasks that are ready to execute (dependencies satisfied)"""
        ready_tasks = []
        
        for task_id, task in self.pending_tasks.items():
            if task_id not in self.running_tasks:
                # Check if all dependencies are completed
                if all(dep in self.completed_tasks for dep in task.dependencies):
                    ready_tasks.append(task)
        
        # Sort by priority (higher priority first)
        ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        
        # Limit by max concurrent tasks
        available_slots = self.max_concurrent_tasks - len(self.running_tasks)
        return ready_tasks[:available_slots]
    
    def mark_task_started(self, task_id: str):
        """Mark a task as started"""
        if task_id in self.pending_tasks:
            self.running_tasks.add(task_id)
            logger.debug(f"Task {task_id} started")
    
    def mark_task_completed(self, task_id: str, success: bool = True):
        """Mark a task as completed"""
        self.running_tasks.discard(task_id)
        
        if success:
            self.completed_tasks.add(task_id)
            logger.debug(f"Task {task_id} completed successfully")
        else:
            self.failed_tasks.add(task_id)
            logger.warning(f"Task {task_id} failed")
        
        # Remove from pending
        self.pending_tasks.pop(task_id, None)
    
    def has_pending_tasks(self) -> bool:
        """Check if there are pending tasks"""
        return len(self.pending_tasks) > 0 or len(self.running_tasks) > 0
    
    def get_status(self) -> Dict[str, int]:
        """Get scheduler status"""
        return {
            'pending': len(self.pending_tasks),
            'running': len(self.running_tasks),
            'completed': len(self.completed_tasks),
            'failed': len(self.failed_tasks)
        }


class UnifiedParallelExecutor:
    """
    Unified Parallel Execution Manager that provides comprehensive concurrent
    processing capabilities across all biomedical query execution strategies.
    """
    
    def __init__(self, config: UnifiedConfig):
        """
        Initialize the parallel executor
        
        Args:
            config: Unified configuration
        """
        self.config = config
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Task scheduling
        self.task_scheduler = TaskScheduler(
            max_concurrent_tasks=config.performance.max_concurrent_queries
        )
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.performance.max_worker_threads
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=config.performance.max_worker_processes
        )
        
        # Performance tracking
        self.execution_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0,
            'concurrent_executions': 0,
            'thread_executions': 0,
            'process_executions': 0
        }
        
        logger.info("Unified Parallel Executor initialized")
    
    async def execute_parallel_predicates(self, 
                                        trapi_queries: List[TRAPIQuery],
                                        execution_func: Callable,
                                        execution_mode: ExecutionMode = ExecutionMode.CONCURRENT,
                                        max_concurrency: int = None) -> List[Tuple[TRAPIQuery, Any]]:
        """
        Execute multiple TRAPI queries with different predicates in parallel
        
        Args:
            trapi_queries: List of TRAPI queries to execute
            execution_func: Function to execute each query
            execution_mode: Type of parallel execution
            max_concurrency: Maximum concurrent executions
            
        Returns:
            List of (query, result) tuples
        """
        logger.info(f"Executing {len(trapi_queries)} TRAPI queries in parallel using {execution_mode.value} mode")
        
        if not trapi_queries:
            return []
        
        max_concurrency = max_concurrency or self.config.performance.max_concurrent_queries
        
        # Create execution tasks
        tasks = []
        for i, query in enumerate(trapi_queries):
            task = ExecutionTask(
                task_id=f"predicate_execution_{query.query_id}",
                task_type="predicate_execution",
                task_func=execution_func,
                args=(query,),
                priority=TaskPriority.HIGH,
                timeout=self.config.performance.query_timeout_seconds,
                estimated_duration=self._estimate_query_duration(query)
            )
            tasks.append(task)
        
        # Create execution batch
        batch = ExecutionBatch(
            batch_id=f"parallel_predicates_{int(time.time())}",
            tasks=tasks,
            execution_mode=execution_mode,
            max_concurrency=max_concurrency,
            timeout=self.config.performance.query_timeout_seconds * 2
        )
        
        # Execute batch
        batch_result = await self.execute_batch(batch)
        
        # Process results
        results = []
        for task_result in batch_result.task_results:
            # Find corresponding query
            query = next((q for q in trapi_queries if f"predicate_execution_{q.query_id}" == task_result.task_id), None)
            if query:
                results.append((query, task_result.result if task_result.success else None))
        
        logger.info(f"Parallel predicate execution completed: {batch_result.success_rate:.2%} success rate")
        return results
    
    async def execute_batch(self, batch: ExecutionBatch) -> BatchResult:
        """
        Execute a batch of tasks
        
        Args:
            batch: Execution batch to process
            
        Returns:
            Batch execution result
        """
        logger.info(f"Executing batch {batch.batch_id} with {len(batch.tasks)} tasks using {batch.execution_mode.value} mode")
        
        start_time = time.time()
        
        # Add tasks to scheduler
        for task in batch.tasks:
            self.task_scheduler.add_task(task)
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Execute based on mode
        if batch.execution_mode == ExecutionMode.CONCURRENT:
            task_results = await self._execute_concurrent_batch(batch)
        elif batch.execution_mode == ExecutionMode.THREADED:
            task_results = await self._execute_threaded_batch(batch)
        elif batch.execution_mode == ExecutionMode.MULTIPROCESS:
            task_results = await self._execute_process_batch(batch)
        elif batch.execution_mode == ExecutionMode.HYBRID:
            task_results = await self._execute_hybrid_batch(batch)
        else:  # SEQUENTIAL
            task_results = await self._execute_sequential_batch(batch)
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        total_time = time.time() - start_time
        
        # Calculate results
        successful_tasks = [r for r in task_results if r.success]
        failed_tasks = [r.task_id for r in task_results if not r.success]
        success_rate = len(successful_tasks) / len(task_results) if task_results else 0.0
        
        # Update stats
        self.execution_stats['total_tasks'] += len(task_results)
        self.execution_stats['successful_tasks'] += len(successful_tasks)
        self.execution_stats['failed_tasks'] += len(failed_tasks)
        self.execution_stats['total_execution_time'] += total_time
        
        if batch.execution_mode == ExecutionMode.CONCURRENT:
            self.execution_stats['concurrent_executions'] += 1
        elif batch.execution_mode == ExecutionMode.THREADED:
            self.execution_stats['thread_executions'] += 1
        elif batch.execution_mode == ExecutionMode.MULTIPROCESS:
            self.execution_stats['process_executions'] += 1
        
        batch_result = BatchResult(
            batch_id=batch.batch_id,
            task_results=task_results,
            total_execution_time=total_time,
            success_rate=success_rate,
            failed_tasks=failed_tasks,
            resource_summary=self.resource_monitor.get_current_usage()
        )
        
        logger.info(f"Batch {batch.batch_id} completed: {success_rate:.2%} success rate in {total_time:.2f}s")
        return batch_result
    
    async def _execute_concurrent_batch(self, batch: ExecutionBatch) -> List[TaskResult]:
        """Execute batch using async concurrency"""
        semaphore = asyncio.Semaphore(batch.max_concurrency)
        tasks = []
        
        async def execute_task_with_semaphore(task: ExecutionTask) -> TaskResult:
            async with semaphore:
                return await self._execute_single_task_async(task)
        
        # Create async tasks
        for task in batch.tasks:
            async_task = asyncio.create_task(execute_task_with_semaphore(task))
            tasks.append(async_task)
        
        # Wait for completion with timeout
        if batch.timeout:
            task_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=batch.timeout
            )
        else:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = []
        for i, result in enumerate(task_results):
            if isinstance(result, TaskResult):
                results.append(result)
            else:
                # Handle exceptions
                error_result = TaskResult(
                    task_id=batch.tasks[i].task_id,
                    success=False,
                    error=result if isinstance(result, Exception) else Exception(str(result))
                )
                results.append(error_result)
        
        return results
    
    async def _execute_threaded_batch(self, batch: ExecutionBatch) -> List[TaskResult]:
        """Execute batch using thread pool"""
        loop = asyncio.get_event_loop()
        futures = []
        
        for task in batch.tasks:
            future = loop.run_in_executor(
                self.thread_pool,
                self._execute_single_task_sync,
                task
            )
            futures.append(future)
        
        # Wait for completion
        if batch.timeout:
            task_results = await asyncio.wait_for(
                asyncio.gather(*futures),
                timeout=batch.timeout
            )
        else:
            task_results = await asyncio.gather(*futures)
        
        return task_results
    
    async def _execute_process_batch(self, batch: ExecutionBatch) -> List[TaskResult]:
        """Execute batch using process pool"""
        loop = asyncio.get_event_loop()
        futures = []
        
        for task in batch.tasks:
            # Only use process pool for CPU-intensive tasks
            if task.task_type in ['computation', 'analysis', 'processing']:
                future = loop.run_in_executor(
                    self.process_pool,
                    self._execute_single_task_sync,
                    task
                )
            else:
                # Use thread pool for I/O tasks
                future = loop.run_in_executor(
                    self.thread_pool,
                    self._execute_single_task_sync,
                    task
                )
            futures.append(future)
        
        # Wait for completion
        if batch.timeout:
            task_results = await asyncio.wait_for(
                asyncio.gather(*futures),
                timeout=batch.timeout
            )
        else:
            task_results = await asyncio.gather(*futures)
        
        return task_results
    
    async def _execute_hybrid_batch(self, batch: ExecutionBatch) -> List[TaskResult]:
        """Execute batch using hybrid approach (async + threaded)"""
        # Separate I/O and CPU tasks
        io_tasks = [t for t in batch.tasks if t.task_type in ['api_call', 'network', 'database']]
        cpu_tasks = [t for t in batch.tasks if t.task_type in ['computation', 'analysis', 'processing']]
        other_tasks = [t for t in batch.tasks if t not in io_tasks and t not in cpu_tasks]
        
        results = []
        
        # Execute I/O tasks concurrently
        if io_tasks:
            io_batch = ExecutionBatch(
                batch_id=f"{batch.batch_id}_io",
                tasks=io_tasks,
                execution_mode=ExecutionMode.CONCURRENT,
                max_concurrency=batch.max_concurrency
            )
            io_results = await self._execute_concurrent_batch(io_batch)
            results.extend(io_results)
        
        # Execute CPU tasks in thread pool
        if cpu_tasks:
            cpu_batch = ExecutionBatch(
                batch_id=f"{batch.batch_id}_cpu",
                tasks=cpu_tasks,
                execution_mode=ExecutionMode.THREADED,
                max_concurrency=min(batch.max_concurrency, self.config.performance.max_worker_threads)
            )
            cpu_results = await self._execute_threaded_batch(cpu_batch)
            results.extend(cpu_results)
        
        # Execute other tasks concurrently
        if other_tasks:
            other_batch = ExecutionBatch(
                batch_id=f"{batch.batch_id}_other",
                tasks=other_tasks,
                execution_mode=ExecutionMode.CONCURRENT,
                max_concurrency=batch.max_concurrency
            )
            other_results = await self._execute_concurrent_batch(other_batch)
            results.extend(other_results)
        
        return results
    
    async def _execute_sequential_batch(self, batch: ExecutionBatch) -> List[TaskResult]:
        """Execute batch sequentially"""
        results = []
        
        for task in batch.tasks:
            result = await self._execute_single_task_async(task)
            results.append(result)
            
            # Check if we should stop on error
            if not result.success and batch.error_tolerance == 0:
                logger.warning(f"Stopping sequential execution due to task failure: {task.task_id}")
                break
        
        return results
    
    async def _execute_single_task_async(self, task: ExecutionTask) -> TaskResult:
        """Execute a single task asynchronously"""
        start_time = time.time()
        
        try:
            # Mark task as started
            self.task_scheduler.mark_task_started(task.task_id)
            
            # Check resource constraints
            if self.resource_monitor.should_throttle():
                logger.warning(f"Throttling task {task.task_id} due to high resource usage")
                await asyncio.sleep(0.1)  # Brief pause
            
            # Execute task
            if asyncio.iscoroutinefunction(task.task_func):
                if task.timeout:
                    result = await asyncio.wait_for(
                        task.task_func(*task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                else:
                    result = await task.task_func(*task.args, **task.kwargs)
            else:
                # Run sync function in thread
                loop = asyncio.get_event_loop()
                if task.timeout:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, task.task_func, *task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                else:
                    result = await loop.run_in_executor(None, task.task_func, *task.args, **task.kwargs)
            
            execution_time = time.time() - start_time
            
            # Mark task as completed
            self.task_scheduler.mark_task_completed(task.task_id, success=True)
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                resource_usage=self.resource_monitor.get_current_usage()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            
            # Mark task as failed
            self.task_scheduler.mark_task_completed(task.task_id, success=False)
            
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=e,
                execution_time=execution_time,
                resource_usage=self.resource_monitor.get_current_usage()
            )
    
    def _execute_single_task_sync(self, task: ExecutionTask) -> TaskResult:
        """Execute a single task synchronously (for thread/process pools)"""
        start_time = time.time()
        
        try:
            # Execute task
            result = task.task_func(*task.args, **task.kwargs)
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=e,
                execution_time=execution_time
            )
    
    def _estimate_query_duration(self, query: TRAPIQuery) -> float:
        """Estimate execution duration for a TRAPI query"""
        # Base duration estimate
        base_duration = 2.0
        
        # Adjust based on query complexity
        if query.estimated_results > 100:
            base_duration += 1.0
        if query.estimated_results > 500:
            base_duration += 2.0
        
        # Adjust based on predicate type
        if 'related_to' in query.predicate:
            base_duration += 0.5  # Related_to queries tend to be slower
        
        return base_duration
    
    async def execute_parallel_api_calls(self,
                                       api_calls: List[Callable],
                                       execution_mode: ExecutionMode = ExecutionMode.CONCURRENT,
                                       max_concurrency: int = None) -> List[Any]:
        """
        Execute multiple API calls in parallel
        
        Args:
            api_calls: List of API call functions
            execution_mode: Type of parallel execution
            max_concurrency: Maximum concurrent calls
            
        Returns:
            List of API call results
        """
        logger.info(f"Executing {len(api_calls)} API calls in parallel")
        
        max_concurrency = max_concurrency or self.config.performance.max_concurrent_api_calls
        
        # Create execution tasks
        tasks = []
        for i, api_call in enumerate(api_calls):
            task = ExecutionTask(
                task_id=f"api_call_{i}",
                task_type="api_call",
                task_func=api_call,
                priority=TaskPriority.HIGH,
                timeout=self.config.performance.api_timeout_seconds
            )
            tasks.append(task)
        
        # Create execution batch
        batch = ExecutionBatch(
            batch_id=f"parallel_api_calls_{int(time.time())}",
            tasks=tasks,
            execution_mode=execution_mode,
            max_concurrency=max_concurrency
        )
        
        # Execute batch
        batch_result = await self.execute_batch(batch)
        
        # Extract results
        results = [task_result.result for task_result in batch_result.task_results 
                  if task_result.success]
        
        logger.info(f"Parallel API calls completed: {len(results)} successful out of {len(api_calls)}")
        return results
    
    @asynccontextmanager
    async def parallel_context(self, max_concurrency: int = None):
        """
        Context manager for parallel execution
        
        Args:
            max_concurrency: Maximum concurrent operations
        """
        max_concurrency = max_concurrency or self.config.performance.max_concurrent_queries
        semaphore = asyncio.Semaphore(max_concurrency)
        
        original_max = self.task_scheduler.max_concurrent_tasks
        self.task_scheduler.max_concurrent_tasks = max_concurrency
        
        try:
            self.resource_monitor.start_monitoring()
            yield semaphore
        finally:
            self.resource_monitor.stop_monitoring()
            self.task_scheduler.max_concurrent_tasks = original_max
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        total_tasks = self.execution_stats['total_tasks']
        
        stats = {
            'total_tasks_executed': total_tasks,
            'successful_tasks': self.execution_stats['successful_tasks'],
            'failed_tasks': self.execution_stats['failed_tasks'],
            'success_rate': (self.execution_stats['successful_tasks'] / total_tasks 
                           if total_tasks > 0 else 0.0),
            'total_execution_time': self.execution_stats['total_execution_time'],
            'average_execution_time': (self.execution_stats['total_execution_time'] / total_tasks 
                                     if total_tasks > 0 else 0.0),
            'concurrent_executions': self.execution_stats['concurrent_executions'],
            'thread_executions': self.execution_stats['thread_executions'],
            'process_executions': self.execution_stats['process_executions'],
            'current_resource_usage': self.resource_monitor.get_current_usage(),
            'scheduler_status': self.task_scheduler.get_status()
        }
        
        return stats
    
    def optimize_execution_mode(self, tasks: List[ExecutionTask]) -> ExecutionMode:
        """
        Automatically select optimal execution mode based on task characteristics
        
        Args:
            tasks: List of tasks to analyze
            
        Returns:
            Recommended execution mode
        """
        if not tasks:
            return ExecutionMode.SEQUENTIAL
        
        # Analyze task types
        io_tasks = len([t for t in tasks if t.task_type in ['api_call', 'network', 'database']])
        cpu_tasks = len([t for t in tasks if t.task_type in ['computation', 'analysis', 'processing']])
        total_tasks = len(tasks)
        
        # Get current resource usage
        usage = self.resource_monitor.get_current_usage()
        
        # Decision logic
        if total_tasks == 1:
            return ExecutionMode.SEQUENTIAL
        elif total_tasks <= 3:
            return ExecutionMode.CONCURRENT
        elif io_tasks / total_tasks > 0.7:
            # Mostly I/O tasks - use concurrency
            return ExecutionMode.CONCURRENT
        elif cpu_tasks / total_tasks > 0.7:
            # Mostly CPU tasks - use threads or processes
            if usage['cpu_percent'] > 70:
                return ExecutionMode.THREADED  # Don't overload CPU
            else:
                return ExecutionMode.MULTIPROCESS
        elif io_tasks > 0 and cpu_tasks > 0:
            # Mixed workload - use hybrid
            return ExecutionMode.HYBRID
        else:
            # Default to concurrent
            return ExecutionMode.CONCURRENT
    
    def shutdown(self):
        """Shutdown the parallel executor and cleanup resources"""
        logger.info("Shutting down parallel executor")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Shutdown process pool
        self.process_pool.shutdown(wait=True)
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        logger.info("Parallel executor shutdown complete")