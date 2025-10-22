"""
Unified Performance Framework

This module provides comprehensive performance monitoring, metrics collection,
and benchmarking capabilities that work consistently across all execution
strategies and components.
"""

import logging
import time
import psutil
import asyncio
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import json
import statistics

from .config import UnifiedConfig
from .types import PerformanceMetrics, ExecutionStep, ExecutionStatus

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage"""
    timestamp: float
    memory_usage_mb: float
    cpu_percent: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0


@dataclass
class OperationTiming:
    """Timing information for a specific operation"""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, metadata: Optional[Dict[str, Any]] = None):
        """Mark operation as finished"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if metadata:
            self.metadata.update(metadata)


@dataclass
class StrategyPerformanceHistory:
    """Historical performance data for a strategy"""
    strategy: str
    executions: List[Dict[str, Any]] = field(default_factory=list)
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    average_quality_score: float = 0.0
    average_confidence: float = 0.0
    total_uses: int = 0
    
    def add_execution(self, execution_data: Dict[str, Any]):
        """Add execution data to history"""
        self.executions.append(execution_data)
        self.total_uses += 1
        self._recalculate_metrics()
    
    def _recalculate_metrics(self):
        """Recalculate aggregate metrics"""
        if not self.executions:
            return
        
        successful = [e for e in self.executions if e.get('success', False)]
        self.success_rate = len(successful) / len(self.executions)
        
        if successful:
            self.average_execution_time = statistics.mean([e.get('execution_time', 0) for e in successful])
            self.average_quality_score = statistics.mean([e.get('quality_score', 0) for e in successful])
            self.average_confidence = statistics.mean([e.get('confidence', 0) for e in successful])


class ResourceMonitor:
    """System resource monitoring"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.snapshots: deque = deque(maxlen=1000)  # Keep last 1000 snapshots
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._baseline_resources: Optional[ResourceSnapshot] = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous resource monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._baseline_resources = self._take_snapshot()
        
        def monitor_loop():
            while self.monitoring:
                try:
                    snapshot = self._take_snapshot()
                    self.snapshots.append(snapshot)
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    time.sleep(interval * 2)  # Back off on error
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        # Get disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
        except:
            disk_read_mb = disk_write_mb = 0
        
        # Get network I/O
        try:
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent if net_io else 0
            net_recv = net_io.bytes_recv if net_io else 0
        except:
            net_sent = net_recv = 0
        
        return ResourceSnapshot(
            timestamp=time.time(),
            memory_usage_mb=memory_info.rss / (1024 * 1024),
            cpu_percent=cpu_percent,
            memory_percent=process.memory_percent(),
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_bytes_sent=net_sent,
            network_bytes_recv=net_recv
        )
    
    def get_current_usage(self) -> ResourceSnapshot:
        """Get current resource usage"""
        return self._take_snapshot()
    
    def get_peak_usage(self, duration_seconds: Optional[float] = None) -> ResourceSnapshot:
        """Get peak resource usage over specified duration"""
        cutoff_time = time.time() - (duration_seconds or 300)  # Default 5 minutes
        relevant_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        
        if not relevant_snapshots:
            return self._take_snapshot()
        
        peak_memory = max(relevant_snapshots, key=lambda s: s.memory_usage_mb)
        peak_cpu = max(relevant_snapshots, key=lambda s: s.cpu_percent)
        
        return ResourceSnapshot(
            timestamp=time.time(),
            memory_usage_mb=peak_memory.memory_usage_mb,
            cpu_percent=peak_cpu.cpu_percent,
            memory_percent=peak_memory.memory_percent,
            disk_io_read_mb=peak_memory.disk_io_read_mb,
            disk_io_write_mb=peak_memory.disk_io_write_mb
        )
    
    def get_average_usage(self, duration_seconds: Optional[float] = None) -> ResourceSnapshot:
        """Get average resource usage over specified duration"""
        cutoff_time = time.time() - (duration_seconds or 300)
        relevant_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        
        if not relevant_snapshots:
            return self._take_snapshot()
        
        avg_memory = statistics.mean([s.memory_usage_mb for s in relevant_snapshots])
        avg_cpu = statistics.mean([s.cpu_percent for s in relevant_snapshots])
        avg_mem_percent = statistics.mean([s.memory_percent for s in relevant_snapshots])
        
        return ResourceSnapshot(
            timestamp=time.time(),
            memory_usage_mb=avg_memory,
            cpu_percent=avg_cpu,
            memory_percent=avg_mem_percent,
            disk_io_read_mb=0,  # Not meaningful for average
            disk_io_write_mb=0
        )


class PerformanceProfiler:
    """Detailed performance profiling for operations"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.active_operations: Dict[str, OperationTiming] = {}
        self.completed_operations: List[OperationTiming] = []
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0, 'total_time': 0.0, 'min_time': float('inf'), 
            'max_time': 0.0, 'errors': 0
        })
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start timing an operation"""
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        operation = OperationTiming(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        self.active_operations[operation_id] = operation
        return operation_id
    
    def finish_operation(self, operation_id: str, success: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """Finish timing an operation"""
        if operation_id not in self.active_operations:
            logger.warning(f"Unknown operation ID: {operation_id}")
            return
        
        operation = self.active_operations.pop(operation_id)
        operation.finish(metadata)
        
        self.completed_operations.append(operation)
        
        # Update statistics
        stats = self.operation_stats[operation.operation_name]
        stats['count'] += 1
        stats['total_time'] += operation.duration
        stats['min_time'] = min(stats['min_time'], operation.duration)
        stats['max_time'] = max(stats['max_time'], operation.duration)
        
        if not success:
            stats['errors'] += 1
    
    @asynccontextmanager
    async def profile_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for profiling operations"""
        operation_id = self.start_operation(operation_name, metadata)
        success = True
        
        try:
            yield operation_id
        except Exception as e:
            success = False
            raise
        finally:
            self.finish_operation(operation_id, success)
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for operations"""
        if operation_name:
            if operation_name not in self.operation_stats:
                return {}
            
            stats = self.operation_stats[operation_name]
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            error_rate = stats['errors'] / stats['count'] if stats['count'] > 0 else 0
            
            return {
                'operation_name': operation_name,
                'total_executions': stats['count'],
                'total_time': stats['total_time'],
                'average_time': avg_time,
                'min_time': stats['min_time'] if stats['min_time'] != float('inf') else 0,
                'max_time': stats['max_time'],
                'error_count': stats['errors'],
                'error_rate': error_rate
            }
        else:
            # Return stats for all operations
            all_stats = {}
            for op_name in self.operation_stats:
                all_stats[op_name] = self.get_operation_stats(op_name)
            return all_stats


class CacheMetricsCollector:
    """Collect and track caching metrics"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.cache_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            'hits': 0, 'misses': 0, 'puts': 0, 'evictions': 0
        })
        self.response_times: Dict[str, List[float]] = defaultdict(list)
    
    def record_cache_hit(self, cache_type: str, response_time: float = 0.0):
        """Record a cache hit"""
        self.cache_stats[cache_type]['hits'] += 1
        if response_time > 0:
            self.response_times[cache_type].append(response_time)
    
    def record_cache_miss(self, cache_type: str, response_time: float = 0.0):
        """Record a cache miss"""
        self.cache_stats[cache_type]['misses'] += 1
        if response_time > 0:
            self.response_times[cache_type].append(response_time)
    
    def record_cache_put(self, cache_type: str):
        """Record a cache put operation"""
        self.cache_stats[cache_type]['puts'] += 1
    
    def record_cache_eviction(self, cache_type: str):
        """Record a cache eviction"""
        self.cache_stats[cache_type]['evictions'] += 1
    
    def get_cache_stats(self, cache_type: Optional[str] = None) -> Dict[str, Any]:
        """Get cache statistics"""
        if cache_type:
            if cache_type not in self.cache_stats:
                return {}
            
            stats = self.cache_stats[cache_type]
            total_requests = stats['hits'] + stats['misses']
            hit_rate = stats['hits'] / total_requests if total_requests > 0 else 0
            
            response_times = self.response_times[cache_type]
            avg_response_time = statistics.mean(response_times) if response_times else 0
            
            return {
                'cache_type': cache_type,
                'hits': stats['hits'],
                'misses': stats['misses'],
                'puts': stats['puts'],
                'evictions': stats['evictions'],
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'average_response_time': avg_response_time
            }
        else:
            # Return stats for all cache types
            all_stats = {}
            for cache_name in self.cache_stats:
                all_stats[cache_name] = self.get_cache_stats(cache_name)
            return all_stats


class StrategyPerformanceTracker:
    """Track performance of different execution strategies"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.strategy_history: Dict[str, StrategyPerformanceHistory] = {}

    def record_execution(self, strategy: str, execution_data: Dict[str, Any]):
        """Record execution results for a strategy"""
        if strategy not in self.strategy_history:
            self.strategy_history[strategy] = StrategyPerformanceHistory(strategy=strategy)
        self.strategy_history[strategy].add_execution(execution_data)
        logger.debug(f"Recorded execution for strategy {strategy}")

    def get_strategy_performance(self, strategy: str) -> StrategyPerformanceHistory:
        """Get performance history for a strategy"""
        return self.strategy_history[strategy]

    def get_best_strategy(self, metric: str = 'success_rate', min_executions: int = 3) -> str:
        """Get the best performing strategy based on specified metric"""
        candidates = [
            (strategy, history) for strategy, history in self.strategy_history.items()
            if history.total_uses >= min_executions
        ]

        if not candidates:
            return 'simple'  # Default fallback

        if metric == 'success_rate':
            best_strategy, _ = max(candidates, key=lambda x: x[1].success_rate)
        elif metric == 'execution_time':
            best_strategy, _ = min(candidates, key=lambda x: x[1].average_execution_time or float('inf'))
        elif metric == 'quality_score':
            best_strategy, _ = max(candidates, key=lambda x: x[1].average_quality_score)
        elif metric == 'confidence':
            best_strategy, _ = max(candidates, key=lambda x: x[1].average_confidence)
        else:
            # Default to success rate
            best_strategy, _ = max(candidates, key=lambda x: x[1].success_rate)

        return best_strategy

    def get_strategy_rankings(self) -> List[Tuple[str, float]]:
        """Get strategies ranked by overall performance"""
        rankings = []

        for strategy, history in self.strategy_history.items():
            if history.total_uses < 1:
                score = 0.0
            else:
                # Composite score: weighted combination of metrics
                score = (
                    0.4 * history.success_rate +
                    0.3 * min(1.0, 1.0 / max(history.average_execution_time, 1.0)) +
                    0.2 * history.average_quality_score +
                    0.1 * history.average_confidence
                )
            rankings.append((strategy, score))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


class UnifiedPerformanceMonitor:
    """
    Unified performance monitoring system that consolidates all performance
    tracking capabilities
    """
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        
        # Initialize components
        self.resource_monitor = ResourceMonitor(config)
        self.profiler = PerformanceProfiler(config)
        self.cache_metrics = CacheMetricsCollector(config)
        self.strategy_tracker = StrategyPerformanceTracker(config)
        
        # Overall metrics
        self.session_start_time = time.time()
        self.query_count = 0
        self.total_execution_time = 0.0
        self.error_count = 0
        
        # Start resource monitoring if enabled
        if config.debug.enable_performance_profiling:
            self.resource_monitor.start_monitoring()
        
        logger.info("Unified performance monitor initialized")
    
    def __del__(self):
        """Cleanup when monitor is destroyed"""
        self.shutdown()
    
    def shutdown(self):
        """Shutdown the performance monitor"""
        self.resource_monitor.stop_monitoring()
    
    @asynccontextmanager
    async def monitor_query_execution(self, query: str, strategy: str):
        """Monitor complete query execution"""
        start_time = time.time()
        self.query_count += 1
        initial_resources = self.resource_monitor.get_current_usage()
        
        execution_data = {
            'query': query[:100],  # Truncate for storage
'strategy': strategy,
            'start_time': start_time,
            'success': False,
            'execution_time': 0.0,
            'quality_score': 0.0,
            'confidence': 0.0,
            'error_message': None,
            'resource_usage': {}
        }
        
        try:
            async with self.profiler.profile_operation(f"query_execution_{strategy}"):
                yield execution_data
            # Execution succeeded
            execution_data['success'] = True
        except Exception as e:
            execution_data['error_message'] = str(e)
            self.error_count += 1
            raise
        finally:
            # Calculate final metrics
            end_time = time.time()
            execution_time = end_time - start_time
            execution_data['execution_time'] = execution_time
            self.total_execution_time += execution_time
            # Get resource usage
            final_resources = self.resource_monitor.get_current_usage()
            execution_data['resource_usage'] = {
                'initial_memory_mb': initial_resources.memory_usage_mb,
                'final_memory_mb': final_resources.memory_usage_mb,
                'memory_delta_mb': final_resources.memory_usage_mb - initial_resources.memory_usage_mb,
                'peak_memory_mb': self.resource_monitor.get_peak_usage(execution_time).memory_usage_mb,
                'average_cpu_percent': self.resource_monitor.get_average_usage(execution_time).cpu_percent
            }
            # Record strategy performance
            self.strategy_tracker.record_execution(strategy, execution_data)
    
    def create_performance_metrics(self, execution_steps: List[ExecutionStep]) -> PerformanceMetrics:
        """Create comprehensive performance metrics from execution steps"""
        metrics = PerformanceMetrics()
        
        # Calculate timing metrics
        for step in execution_steps:
            if step.execution_time:
                if step.step_type == 'entity_extraction':
                    metrics.entity_extraction_time += step.execution_time
                elif step.step_type in ['trapi_building', 'query_building']:
                    metrics.query_building_time += step.execution_time
                elif step.step_type in ['api_execution', 'bte_api']:
                    metrics.api_execution_time += step.execution_time
                elif step.step_type in ['result_processing', 'aggregation']:
                    metrics.result_processing_time += step.execution_time
                
                metrics.total_execution_time += step.execution_time
            
            # Count errors and retries
            if step.status == ExecutionStatus.FAILED:
                metrics.error_count += 1
            
            # Collect confidence scores
            if step.confidence > 0:
                metrics.add_confidence_score(step.confidence)
        
        # Get current resource usage
        current_resources = self.resource_monitor.get_current_usage()
        metrics.memory_usage_mb = current_resources.memory_usage_mb
        metrics.cpu_usage_percent = current_resources.cpu_percent
        
        # Get cache metrics
        cache_stats = self.cache_metrics.get_cache_stats()
        for cache_type, stats in cache_stats.items():
            metrics.cache_hits += stats.get('hits', 0)
            metrics.cache_misses += stats.get('misses', 0)
        
        return metrics
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'query_count': self.query_count,
            'total_execution_time': self.total_execution_time,
            'error_count': self.error_count,
            'session_duration': time.time() - self.session_start_time,
            'resource_usage': self.resource_monitor.get_current_usage().__dict__,
            'cache_stats': self.cache_metrics.get_cache_stats()
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        session_duration = time.time() - self.session_start_time
        
        return {
            'session_info': {
                'start_time': self.session_start_time,
                'duration_seconds': session_duration,
                'query_count': self.query_count,
                'total_execution_time': self.total_execution_time,
                'error_count': self.error_count,
                'error_rate': self.error_count / self.query_count if self.query_count > 0 else 0,
                'average_query_time': self.total_execution_time / self.query_count if self.query_count > 0 else 0
            },
            'resource_usage': {
                'current': self.resource_monitor.get_current_usage().__dict__,
                'peak': self.resource_monitor.get_peak_usage(session_duration).__dict__,
                'average': self.resource_monitor.get_average_usage(session_duration).__dict__
            },
            'operation_performance': self.profiler.get_operation_stats(),
            'cache_performance': self.cache_metrics.get_cache_stats(),
            'strategy_performance': {
                strategy: history.__dict__ 
                for strategy, history in self.strategy_tracker.strategy_history.items()
            },
            'strategy_rankings': [
                {'strategy': strategy, 'score': score}
                for strategy, score in self.strategy_tracker.get_strategy_rankings()
            ]
        }
    
    def save_report(self, filepath: str):
        """Save performance report to file"""
        report = self.get_comprehensive_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Performance report saved to {filepath}")
    
    # Convenience methods for external components
    def record_cache_hit(self, cache_type: str, response_time: float = 0.0):
        """Record cache hit"""
        self.cache_metrics.record_cache_hit(cache_type, response_time)
    
    def record_cache_miss(self, cache_type: str, response_time: float = 0.0):
        """Record cache miss"""
        self.cache_metrics.record_cache_miss(cache_type, response_time)
    
    def get_best_strategy(self, metric: str = 'success_rate') -> str:
        """Get best performing strategy"""
        return self.strategy_tracker.get_best_strategy(metric)
    
    def profile_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Profile an operation"""
        return self.profiler.profile_operation(operation_name, metadata)