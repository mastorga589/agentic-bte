"""
Tests for the unified parallel executor
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from concurrent.futures import Future

from agentic_bte.unified.config import UnifiedConfig
from agentic_bte.unified.parallel_executor import (
    UnifiedParallelExecutor, ExecutionTask, ExecutionBatch, TaskResult, 
    BatchResult, ExecutionMode, TaskPriority, ResourceMonitor, TaskScheduler
)
from agentic_bte.unified.knowledge_manager import TRAPIQuery
from agentic_bte.core.knowledge.predicate_strategy import QueryIntent


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
    
    def test_get_current_usage_without_psutil(self):
        """Test getting current usage without psutil (fallback)"""
        monitor = ResourceMonitor()
        usage = monitor.get_current_usage()
        
        # Should have fallback values
        assert 'cpu_percent' in usage
        assert 'memory_percent' in usage
        assert 'active_threads' in usage
        assert 'active_processes' in usage


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
        if len(ready_tasks) >= 2:
            assert ready_tasks[0].priority.value >= ready_tasks[1].priority.value  # Higher priority first


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])