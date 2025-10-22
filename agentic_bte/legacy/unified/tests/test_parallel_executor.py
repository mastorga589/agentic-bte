"""
Tests for the unified parallel executor
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from concurrent.futures import Future

from ..config import UnifiedConfig
from ..parallel_executor import (
    UnifiedParallelExecutor, ExecutionTask, ExecutionBatch, TaskResult, 
    BatchResult, ExecutionMode, TaskPriority, ResourceMonitor, TaskScheduler
)
from ..knowledge_manager import TRAPIQuery
from ...core.knowledge.predicate_strategy import QueryIntent


@pytest.fixture
def mock_config():
    """Create a mock unified config"""
    config = Mock(spec=UnifiedConfig)
    
    # Performance config
    config.performance = Mock()
    config.performance.max_concurrent_queries = 5
    config.performance.max_concurrent_api_calls = 10
    config.performance.max_worker_threads = 4
    config.performance.max_worker_processes = 2
    config.performance.query_timeout_seconds = 30.0
    config.performance.api_timeout_seconds = 15.0
    
    return config


@pytest.fixture
def sample_execution_task():
    """Create a sample execution task"""
    def dummy_func(x):
        return x * 2
    
    return ExecutionTask(
        task_id="test_task_1",
        task_type="computation",
        task_func=dummy_func,
        args=(5,),
        priority=TaskPriority.HIGH,
        timeout=10.0,
        estimated_duration=2.0
    )


@pytest.fixture
def sample_trapi_queries():
    """Create sample TRAPI queries"""
    queries = []
    
    for i in range(3):
        query = Mock(spec=TRAPIQuery)
        query.query_id = f"query_{i}"
        query.predicate = "biolink:treats"
        query.estimated_results = 50 * (i + 1)
        queries.append(query)
    
    return queries


class TestResourceMonitor:
    """Test resource monitoring functionality"""
    
    def test_resource_monitor_initialization(self):
        """Test resource monitor initialization"""
        monitor = ResourceMonitor()
        
        assert monitor.cpu_threshold == 80.0
        assert monitor.memory_threshold == 80.0
        assert not monitor._monitoring
        assert monitor._resource_history == []
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        monitor = ResourceMonitor()
        
        # Test starting
        monitor.start_monitoring()
        assert monitor._monitoring is True
        
        # Test stopping
        monitor.stop_monitoring()
        assert monitor._monitoring is False
    
    @patch('agentic_bte.unified.parallel_executor.psutil')
    def test_get_current_usage_with_psutil(self, mock_psutil):
        """Test getting current usage with psutil available"""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 75.0
        mock_psutil.virtual_memory.return_value.percent = 60.0
        mock_psutil.pids.return_value = [1, 2, 3, 4, 5]
        
        monitor = ResourceMonitor()
        usage = monitor.get_current_usage()
        
        assert usage['cpu_percent'] == 75.0
        assert usage['memory_percent'] == 60.0
        assert 'active_threads' in usage
        assert usage['active_processes'] == 5
    
    def test_get_current_usage_without_psutil(self):
        """Test getting current usage without psutil (fallback)"""
        monitor = ResourceMonitor()
        usage = monitor.get_current_usage()
        
        # Should have fallback values
        assert 'cpu_percent' in usage
        assert 'memory_percent' in usage
        assert 'active_threads' in usage
        assert 'active_processes' in usage
    
    def test_should_throttle(self):
        """Test throttling decision"""
        monitor = ResourceMonitor()
        
        # Mock high resource usage
        with patch.object(monitor, 'get_current_usage') as mock_usage:
            # High CPU usage
            mock_usage.return_value = {'cpu_percent': 85.0, 'memory_percent': 50.0}
            assert monitor.should_throttle() is True
            
            # High memory usage
            mock_usage.return_value = {'cpu_percent': 50.0, 'memory_percent': 85.0}
            assert monitor.should_throttle() is True
            
            # Normal usage
            mock_usage.return_value = {'cpu_percent': 50.0, 'memory_percent': 50.0}
            assert monitor.should_throttle() is False


class TestTaskScheduler:
    """Test task scheduling functionality"""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization"""
        scheduler = TaskScheduler(max_concurrent_tasks=5)
        
        assert scheduler.max_concurrent_tasks == 5
        assert len(scheduler.pending_tasks) == 0
        assert len(scheduler.completed_tasks) == 0
        assert len(scheduler.running_tasks) == 0
        assert len(scheduler.failed_tasks) == 0
    
    def test_add_task(self, sample_execution_task):
        """Test adding tasks to scheduler"""
        scheduler = TaskScheduler()
        
        scheduler.add_task(sample_execution_task)
        
        assert sample_execution_task.task_id in scheduler.pending_tasks
        assert scheduler.pending_tasks[sample_execution_task.task_id] == sample_execution_task
    
    def test_get_ready_tasks_no_dependencies(self, sample_execution_task):
        """Test getting ready tasks without dependencies"""
        scheduler = TaskScheduler(max_concurrent_tasks=2)
        
        # Add multiple tasks
        task1 = sample_execution_task
        task2 = ExecutionTask("task_2", "computation", lambda x: x, priority=TaskPriority.LOW)
        task3 = ExecutionTask("task_3", "computation", lambda x: x, priority=TaskPriority.CRITICAL)
        
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        scheduler.add_task(task3)
        
        ready_tasks = scheduler.get_ready_tasks()
        
        # Should return tasks ordered by priority, limited by max_concurrent
        assert len(ready_tasks) <= 2
        assert ready_tasks[0].priority.value >= ready_tasks[1].priority.value  # Higher priority first
    
    def test_get_ready_tasks_with_dependencies(self):
        """Test getting ready tasks with dependencies"""
        scheduler = TaskScheduler()
        
        # Create tasks with dependencies
        task1 = ExecutionTask("task_1", "computation", lambda: None)
        task2 = ExecutionTask("task_2", "computation", lambda: None, dependencies={"task_1"})
        
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        
        # Initially only task1 should be ready
        ready_tasks = scheduler.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "task_1"
        
        # Complete task1
        scheduler.mark_task_completed("task_1", success=True)
        
        # Now task2 should be ready
        ready_tasks = scheduler.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "task_2"
    
    def test_mark_task_lifecycle(self, sample_execution_task):
        """Test task lifecycle management"""
        scheduler = TaskScheduler()
        scheduler.add_task(sample_execution_task)
        
        task_id = sample_execution_task.task_id
        
        # Mark as started
        scheduler.mark_task_started(task_id)
        assert task_id in scheduler.running_tasks
        assert task_id in scheduler.pending_tasks
        
        # Mark as completed successfully
        scheduler.mark_task_completed(task_id, success=True)
        assert task_id in scheduler.completed_tasks
        assert task_id not in scheduler.running_tasks
        assert task_id not in scheduler.pending_tasks
    
    def test_mark_task_failed(self, sample_execution_task):
        """Test marking task as failed"""
        scheduler = TaskScheduler()
        scheduler.add_task(sample_execution_task)
        
        task_id = sample_execution_task.task_id
        
        scheduler.mark_task_started(task_id)
        scheduler.mark_task_completed(task_id, success=False)
        
        assert task_id in scheduler.failed_tasks
        assert task_id not in scheduler.completed_tasks
        assert task_id not in scheduler.running_tasks
    
    def test_has_pending_tasks(self, sample_execution_task):
        """Test checking for pending tasks"""
        scheduler = TaskScheduler()
        
        assert not scheduler.has_pending_tasks()
        
        scheduler.add_task(sample_execution_task)
        assert scheduler.has_pending_tasks()
        
        scheduler.mark_task_started(sample_execution_task.task_id)
        assert scheduler.has_pending_tasks()  # Still has running task
        
        scheduler.mark_task_completed(sample_execution_task.task_id, success=True)
        assert not scheduler.has_pending_tasks()
    
    def test_get_status(self, sample_execution_task):
        """Test getting scheduler status"""
        scheduler = TaskScheduler()
        
        status = scheduler.get_status()
        assert status['pending'] == 0
        assert status['running'] == 0
        assert status['completed'] == 0
        assert status['failed'] == 0
        
        scheduler.add_task(sample_execution_task)
        scheduler.mark_task_started(sample_execution_task.task_id)
        
        status = scheduler.get_status()
        assert status['pending'] == 1  # Still in pending until completed
        assert status['running'] == 1


class TestUnifiedParallelExecutor:
    """Test the unified parallel executor"""
    
    def test_executor_initialization(self, mock_config):
        """Test executor initialization"""
        executor = UnifiedParallelExecutor(mock_config)
        
        assert executor.config == mock_config
        assert executor.resource_monitor is not None
        assert executor.task_scheduler is not None
        assert executor.thread_pool is not None
        assert executor.process_pool is not None
        assert 'total_tasks' in executor.execution_stats
    
    @pytest.mark.asyncio
    async def test_execute_single_task_async_success(self, mock_config):
        """Test successful async task execution"""
        executor = UnifiedParallelExecutor(mock_config)
        
        async def async_task(x):
            return x * 2
        
        task = ExecutionTask("test_task", "computation", async_task, args=(5,))
        
        result = await executor._execute_single_task_async(task)
        
        assert result.success is True
        assert result.result == 10
        assert result.task_id == "test_task"
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_execute_single_task_async_failure(self, mock_config):
        """Test failed async task execution"""
        executor = UnifiedParallelExecutor(mock_config)
        
        async def failing_task():
            raise ValueError("Test error")
        
        task = ExecutionTask("failing_task", "computation", failing_task)
        
        result = await executor._execute_single_task_async(task)
        
        assert result.success is False
        assert isinstance(result.error, ValueError)
        assert result.task_id == "failing_task"
    
    @pytest.mark.asyncio
    async def test_execute_single_task_async_timeout(self, mock_config):
        """Test task execution with timeout"""
        executor = UnifiedParallelExecutor(mock_config)
        
        async def slow_task():
            await asyncio.sleep(2.0)
            return "completed"
        
        task = ExecutionTask("slow_task", "computation", slow_task, timeout=0.1)
        
        result = await executor._execute_single_task_async(task)
        
        assert result.success is False
        assert isinstance(result.error, asyncio.TimeoutError)
    
    def test_execute_single_task_sync(self, mock_config):
        """Test synchronous task execution"""
        executor = UnifiedParallelExecutor(mock_config)
        
        def sync_task(x, y):
            return x + y
        
        task = ExecutionTask("sync_task", "computation", sync_task, args=(3, 4))
        
        result = executor._execute_single_task_sync(task)
        
        assert result.success is True
        assert result.result == 7
        assert result.task_id == "sync_task"
    
    def test_estimate_query_duration(self, mock_config, sample_trapi_queries):
        """Test query duration estimation"""
        executor = UnifiedParallelExecutor(mock_config)
        
        query = sample_trapi_queries[0]
        duration = executor._estimate_query_duration(query)
        
        assert duration > 0
        assert isinstance(duration, float)
        
        # Test with complex query
        query.estimated_results = 600
        query.predicate = "biolink:related_to"
        
        complex_duration = executor._estimate_query_duration(query)
        assert complex_duration > duration  # Should be longer
    
    @pytest.mark.asyncio
    async def test_execute_concurrent_batch(self, mock_config):
        """Test concurrent batch execution"""
        executor = UnifiedParallelExecutor(mock_config)
        
        # Create simple tasks
        tasks = []
        for i in range(3):
            task = ExecutionTask(
                f"task_{i}",
                "computation", 
                lambda x=i: x * 2,  # Use default argument to capture i
                priority=TaskPriority.MEDIUM
            )
            tasks.append(task)
        
        batch = ExecutionBatch("test_batch", tasks, ExecutionMode.CONCURRENT, max_concurrency=2)
        
        results = await executor._execute_concurrent_batch(batch)
        
        assert len(results) == 3
        assert all(isinstance(r, TaskResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_execute_parallel_predicates(self, mock_config, sample_trapi_queries):
        """Test parallel predicate execution"""
        executor = UnifiedParallelExecutor(mock_config)
        
        # Mock execution function
        async def mock_execution_func(query):
            await asyncio.sleep(0.01)  # Simulate work
            return f"result_for_{query.query_id}"
        
        results = await executor.execute_parallel_predicates(
            sample_trapi_queries,
            mock_execution_func,
            ExecutionMode.CONCURRENT,
            max_concurrency=2
        )
        
        assert len(results) == 3
        assert all(isinstance(item, tuple) for item in results)
        assert all(len(item) == 2 for item in results)
        
        # Check that we got results for all queries
        query_ids = {result[0].query_id for result in results if result[0] is not None}
        expected_ids = {q.query_id for q in sample_trapi_queries}
        assert query_ids == expected_ids
    
    @pytest.mark.asyncio
    async def test_execute_parallel_api_calls(self, mock_config):
        """Test parallel API call execution"""
        executor = UnifiedParallelExecutor(mock_config)
        
        # Create mock API calls
        async def api_call_1():
            await asyncio.sleep(0.01)
            return "api_result_1"
        
        async def api_call_2():
            await asyncio.sleep(0.01) 
            return "api_result_2"
        
        async def failing_api_call():
            raise Exception("API failed")
        
        api_calls = [api_call_1, api_call_2, failing_api_call]
        
        results = await executor.execute_parallel_api_calls(
            api_calls,
            ExecutionMode.CONCURRENT,
            max_concurrency=3
        )
        
        # Should get results only from successful calls
        assert len(results) == 2
        assert "api_result_1" in results
        assert "api_result_2" in results
    
    @pytest.mark.asyncio 
    async def test_execute_batch_different_modes(self, mock_config):
        """Test batch execution with different modes"""
        executor = UnifiedParallelExecutor(mock_config)
        
        # Create simple tasks
        tasks = []
        for i in range(2):
            task = ExecutionTask(
                f"task_{i}",
                "computation",
                lambda x=i: x + 1,
                priority=TaskPriority.MEDIUM
            )
            tasks.append(task)
        
        # Test different execution modes
        modes_to_test = [
            ExecutionMode.SEQUENTIAL,
            ExecutionMode.CONCURRENT,
            ExecutionMode.THREADED
        ]
        
        for mode in modes_to_test:
            batch = ExecutionBatch(f"batch_{mode.value}", tasks, mode, max_concurrency=2)
            
            result = await executor.execute_batch(batch)
            
            assert isinstance(result, BatchResult)
            assert result.batch_id == f"batch_{mode.value}"
            assert len(result.task_results) == 2
            assert result.success_rate > 0
    
    @pytest.mark.asyncio
    async def test_parallel_context_manager(self, mock_config):
        """Test parallel context manager"""
        executor = UnifiedParallelExecutor(mock_config)
        
        async with executor.parallel_context(max_concurrency=3) as semaphore:
            assert isinstance(semaphore, asyncio.Semaphore)
            assert semaphore._value == 3  # Check semaphore limit
        
        # Context should be cleaned up after exiting
    
    def test_optimize_execution_mode(self, mock_config):
        """Test automatic execution mode optimization"""
        executor = UnifiedParallelExecutor(mock_config)
        
        # Test empty tasks
        assert executor.optimize_execution_mode([]) == ExecutionMode.SEQUENTIAL
        
        # Test single task
        single_task = [ExecutionTask("task_1", "computation", lambda: None)]
        assert executor.optimize_execution_mode(single_task) == ExecutionMode.SEQUENTIAL
        
        # Test mostly I/O tasks
        io_tasks = [
            ExecutionTask(f"io_task_{i}", "api_call", lambda: None)
            for i in range(5)
        ]
        assert executor.optimize_execution_mode(io_tasks) == ExecutionMode.CONCURRENT
        
        # Test mostly CPU tasks
        cpu_tasks = [
            ExecutionTask(f"cpu_task_{i}", "computation", lambda: None)
            for i in range(5)
        ]
        # Mode depends on current CPU usage, but should be either THREADED or MULTIPROCESS
        mode = executor.optimize_execution_mode(cpu_tasks)
        assert mode in [ExecutionMode.THREADED, ExecutionMode.MULTIPROCESS]
        
        # Test mixed tasks
        mixed_tasks = [
            ExecutionTask("io_task", "api_call", lambda: None),
            ExecutionTask("cpu_task", "computation", lambda: None),
            ExecutionTask("other_task", "other", lambda: None)
        ]
        assert executor.optimize_execution_mode(mixed_tasks) == ExecutionMode.HYBRID
    
    def test_get_execution_statistics(self, mock_config):
        """Test getting execution statistics"""
        executor = UnifiedParallelExecutor(mock_config)
        
        # Update some stats
        executor.execution_stats['total_tasks'] = 10
        executor.execution_stats['successful_tasks'] = 8
        executor.execution_stats['failed_tasks'] = 2
        executor.execution_stats['total_execution_time'] = 25.0
        executor.execution_stats['concurrent_executions'] = 3
        
        stats = executor.get_execution_statistics()
        
        assert stats['total_tasks_executed'] == 10
        assert stats['successful_tasks'] == 8
        assert stats['failed_tasks'] == 2
        assert stats['success_rate'] == 0.8
        assert stats['total_execution_time'] == 25.0
        assert stats['average_execution_time'] == 2.5
        assert stats['concurrent_executions'] == 3
        assert 'current_resource_usage' in stats
        assert 'scheduler_status' in stats
    
    def test_shutdown(self, mock_config):
        """Test executor shutdown"""
        executor = UnifiedParallelExecutor(mock_config)
        
        # Mock the pools to verify shutdown is called
        with patch.object(executor.thread_pool, 'shutdown') as mock_thread_shutdown, \
             patch.object(executor.process_pool, 'shutdown') as mock_process_shutdown, \
             patch.object(executor.resource_monitor, 'stop_monitoring') as mock_stop_monitoring:
            
            executor.shutdown()
            
            mock_thread_shutdown.assert_called_once_with(wait=True)
            mock_process_shutdown.assert_called_once_with(wait=True)
            mock_stop_monitoring.assert_called_once()


class TestDataStructures:
    """Test data structures used by parallel executor"""
    
    def test_execution_task_creation(self):
        """Test ExecutionTask creation"""
        def dummy_func():
            return "test"
        
        task = ExecutionTask(
            task_id="test_task",
            task_type="computation",
            task_func=dummy_func,
            args=(1, 2),
            kwargs={'key': 'value'},
            priority=TaskPriority.HIGH,
            timeout=30.0,
            dependencies={'dep1', 'dep2'},
            estimated_duration=5.0,
            resource_requirements={'cpu': 50, 'memory': 100}
        )
        
        assert task.task_id == "test_task"
        assert task.task_type == "computation"
        assert task.task_func == dummy_func
        assert task.args == (1, 2)
        assert task.kwargs == {'key': 'value'}
        assert task.priority == TaskPriority.HIGH
        assert task.timeout == 30.0
        assert task.dependencies == {'dep1', 'dep2'}
        assert task.estimated_duration == 5.0
        assert task.resource_requirements == {'cpu': 50, 'memory': 100}
    
    def test_execution_batch_creation(self):
        """Test ExecutionBatch creation"""
        tasks = [
            ExecutionTask("task_1", "computation", lambda: None),
            ExecutionTask("task_2", "api_call", lambda: None)
        ]
        
        batch = ExecutionBatch(
            batch_id="test_batch",
            tasks=tasks,
            execution_mode=ExecutionMode.CONCURRENT,
            max_concurrency=5,
            timeout=60.0,
            error_tolerance=0.2
        )
        
        assert batch.batch_id == "test_batch"
        assert len(batch.tasks) == 2
        assert batch.execution_mode == ExecutionMode.CONCURRENT
        assert batch.max_concurrency == 5
        assert batch.timeout == 60.0
        assert batch.error_tolerance == 0.2
    
    def test_task_result_creation(self):
        """Test TaskResult creation"""
        result = TaskResult(
            task_id="test_task",
            success=True,
            result="task_output",
            error=None,
            execution_time=2.5,
            resource_usage={'cpu': 45.0, 'memory': 60.0}
        )
        
        assert result.task_id == "test_task"
        assert result.success is True
        assert result.result == "task_output"
        assert result.error is None
        assert result.execution_time == 2.5
        assert result.resource_usage == {'cpu': 45.0, 'memory': 60.0}
    
    def test_batch_result_creation(self):
        """Test BatchResult creation"""
        task_results = [
            TaskResult("task_1", True, "result_1"),
            TaskResult("task_2", False, None, Exception("error"))
        ]
        
        batch_result = BatchResult(
            batch_id="test_batch",
            task_results=task_results,
            total_execution_time=10.0,
            success_rate=0.5,
            failed_tasks=["task_2"],
            resource_summary={'cpu': 70.0, 'memory': 80.0}
        )
        
        assert batch_result.batch_id == "test_batch"
        assert len(batch_result.task_results) == 2
        assert batch_result.total_execution_time == 10.0
        assert batch_result.success_rate == 0.5
        assert batch_result.failed_tasks == ["task_2"]
        assert batch_result.resource_summary == {'cpu': 70.0, 'memory': 80.0}


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mixed_workload_execution(self, mock_config):
        """Test executing a mixed workload with different task types"""
        executor = UnifiedParallelExecutor(mock_config)
        
        # Create mixed tasks
        async def api_task():
            await asyncio.sleep(0.01)
            return "api_result"
        
        def cpu_task(n):
            return sum(range(n))
        
        tasks = [
            ExecutionTask("api_1", "api_call", api_task),
            ExecutionTask("cpu_1", "computation", cpu_task, args=(100,)),
            ExecutionTask("api_2", "api_call", api_task),
            ExecutionTask("cpu_2", "computation", cpu_task, args=(200,))
        ]
        
        batch = ExecutionBatch(
            "mixed_workload",
            tasks,
            ExecutionMode.HYBRID,
            max_concurrency=4
        )
        
        result = await executor.execute_batch(batch)
        
        assert result.success_rate > 0
        assert len(result.task_results) == 4
        
        # Check that different task types completed successfully
        api_results = [r for r in result.task_results if r.task_id.startswith("api_")]
        cpu_results = [r for r in result.task_results if r.task_id.startswith("cpu_")]
        
        assert len(api_results) == 2
        assert len(cpu_results) == 2
    
    @pytest.mark.asyncio
    @pytest.mark.integration 
    async def test_error_handling_and_recovery(self, mock_config):
        """Test error handling and partial success scenarios"""
        executor = UnifiedParallelExecutor(mock_config)
        
        # Create tasks with mixed success/failure
        async def success_task():
            return "success"
        
        async def failure_task():
            raise ValueError("Simulated failure")
        
        async def timeout_task():
            await asyncio.sleep(1.0)  # Will timeout with short timeout setting
            return "timeout"
        
        tasks = [
            ExecutionTask("success_1", "api_call", success_task),
            ExecutionTask("failure_1", "api_call", failure_task),
            ExecutionTask("success_2", "api_call", success_task),
            ExecutionTask("timeout_1", "api_call", timeout_task, timeout=0.1)
        ]
        
        batch = ExecutionBatch(
            "error_handling_test",
            tasks,
            ExecutionMode.CONCURRENT,
            max_concurrency=4,
            error_tolerance=0.5  # Allow 50% failures
        )
        
        result = await executor.execute_batch(batch)
        
        # Should have some successes and some failures
        successful_tasks = [r for r in result.task_results if r.success]
        failed_tasks = [r for r in result.task_results if not r.success]
        
        assert len(successful_tasks) >= 2  # At least the success tasks
        assert len(failed_tasks) >= 2      # At least the failure and timeout tasks
        assert len(result.failed_tasks) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])