"""
Performance Monitoring and Metrics System

This module provides comprehensive performance tracking, quality metrics,
and execution monitoring for query optimizers.
"""

import time
import logging
import statistics
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

# Avoid circular imports by importing at runtime
# We'll import the enums dynamically when needed

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """Categories of metrics to track"""
    PERFORMANCE = "performance"    # Speed, throughput, efficiency
    QUALITY = "quality"           # Accuracy, completeness, relevance
    RESOURCE = "resource"         # API calls, memory, cost
    RELIABILITY = "reliability"   # Success rate, error rates, retries
    USER_EXPERIENCE = "user_experience"  # Response time, satisfaction


@dataclass
class MetricPoint:
    """A single metric observation"""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesMetric:
    """Time series of metric observations"""
    name: str
    category: MetricCategory
    unit: str = ""
    description: str = ""
    points: deque = field(default_factory=lambda: deque(maxlen=1000))  # Keep last 1000 points
    
    def add_point(self, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Add a new metric point"""
        self.points.append(MetricPoint(
            timestamp=time.time(),
            value=value,
            metadata=metadata or {}
        ))
    
    def get_recent_values(self, seconds: int = 3600) -> List[float]:
        """Get values from the last N seconds"""
        cutoff = time.time() - seconds
        return [p.value for p in self.points if p.timestamp >= cutoff]
    
    def get_stats(self, seconds: int = 3600) -> Dict[str, float]:
        """Get statistical summary of recent values"""
        values = self.get_recent_values(seconds)
        if not values:
            return {}
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0
        }


@dataclass
class OptimizerPerformanceProfile:
    """Performance profile for a specific optimizer"""
    strategy: Any  # Will be OptimizationStrategy at runtime
    total_queries: int = 0
    successful_queries: int = 0
    total_execution_time: float = 0.0
    total_results: int = 0
    total_api_calls: int = 0
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    total_retries: int = 0
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    avg_quality_score: float = 0.0
    avg_relevance_score: float = 0.0
    
    # Time series metrics
    execution_times: deque = field(default_factory=lambda: deque(maxlen=100))
    quality_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    result_counts: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, result: Any):
        """Update profile with a new result"""
        self.total_queries += 1
        
        if result.success:
            self.successful_queries += 1
        
        self.total_execution_time += result.metrics.execution_time
        self.total_results += result.metrics.total_results
        self.total_api_calls += result.metrics.api_calls_made
        self.total_cache_hits += result.metrics.cache_hits
        self.total_cache_misses += result.metrics.cache_misses
        self.total_retries += result.metrics.retry_count
        
        # Track errors
        for error in result.errors:
            # Simplify error for categorization
            error_type = error.split(':')[0] if ':' in error else error[:50]
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Update time series
        self.execution_times.append(result.metrics.execution_time)
        self.quality_scores.append(result.metrics.quality_score)
        self.result_counts.append(result.metrics.total_results)
        
        # Update averages
        if self.successful_queries > 0:
            self.avg_quality_score = (
                (self.avg_quality_score * (self.successful_queries - 1) + result.metrics.quality_score) 
                / self.successful_queries
            )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries
    
    @property
    def avg_execution_time(self) -> float:
        """Calculate average execution time"""
        if self.total_queries == 0:
            return 0.0
        return self.total_execution_time / self.total_queries
    
    @property
    def avg_results_per_query(self) -> float:
        """Calculate average results per query"""
        if self.total_queries == 0:
            return 0.0
        return self.total_results / self.total_queries
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_cache_requests = self.total_cache_hits + self.total_cache_misses
        if total_cache_requests == 0:
            return 0.0
        return self.total_cache_hits / total_cache_requests


class PerformanceAlert:
    """Performance alerting system"""
    
    def __init__(self, name: str, threshold: float, comparison: str = "gt"):
        self.name = name
        self.threshold = threshold
        self.comparison = comparison  # "gt", "lt", "eq"
        self.triggered_count = 0
        self.last_triggered = None
    
    def check(self, value: float) -> bool:
        """Check if alert should trigger"""
        triggered = False
        
        if self.comparison == "gt" and value > self.threshold:
            triggered = True
        elif self.comparison == "lt" and value < self.threshold:
            triggered = True
        elif self.comparison == "eq" and abs(value - self.threshold) < 0.001:
            triggered = True
        
        if triggered:
            self.triggered_count += 1
            self.last_triggered = time.time()
            logger.warning(f"Performance alert '{self.name}' triggered: {value} {self.comparison} {self.threshold}")
        
        return triggered


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    
    Tracks metrics across all optimizers and provides insights into
    performance trends, bottlenecks, and optimization opportunities.
    """
    
    def __init__(self):
        """Initialize the performance monitor"""
        # Thread safety
        self._lock = threading.RLock()
        
        # Optimizer profiles
        self.optimizer_profiles: Dict[OptimizationStrategy, OptimizerPerformanceProfile] = {}
        
        # System-wide metrics
        self.system_metrics: Dict[str, TimeSeriesMetric] = {}
        
        # Performance alerts
        self.alerts: List[PerformanceAlert] = []
        
        # Session tracking
        self.session_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Initialize system metrics
        self._initialize_system_metrics()
        
        # Initialize alerts
        self._initialize_alerts()
        
        logger.info("Performance monitor initialized")
    
    def _initialize_system_metrics(self):
        """Initialize system-wide metrics"""
        metrics = [
            ("total_queries_per_minute", MetricCategory.PERFORMANCE, "queries/min", "Total queries executed per minute"),
            ("avg_response_time", MetricCategory.PERFORMANCE, "seconds", "Average response time across all optimizers"),
            ("total_api_calls_per_minute", MetricCategory.RESOURCE, "calls/min", "Total API calls made per minute"),
            ("cache_hit_rate", MetricCategory.RESOURCE, "percentage", "System-wide cache hit rate"),
            ("error_rate", MetricCategory.RELIABILITY, "percentage", "System-wide error rate"),
            ("avg_quality_score", MetricCategory.QUALITY, "score", "Average quality score across all results"),
            ("concurrent_queries", MetricCategory.PERFORMANCE, "count", "Number of concurrent queries being processed"),
        ]
        
        for name, category, unit, description in metrics:
            self.system_metrics[name] = TimeSeriesMetric(
                name=name,
                category=category,
                unit=unit,
                description=description
            )
    
    def _initialize_alerts(self):
        """Initialize performance alerts"""
        self.alerts = [
            PerformanceAlert("high_response_time", 30.0, "gt"),  # > 30 seconds
            PerformanceAlert("low_success_rate", 0.8, "lt"),     # < 80%
            PerformanceAlert("high_error_rate", 0.1, "gt"),      # > 10%
            PerformanceAlert("low_cache_hit_rate", 0.3, "lt"),   # < 30%
        ]
    
    def record_optimization_result(self, result: Any, session_id: Optional[str] = None):
        """
        Record the results of an optimization execution
        
        Args:
            result: The optimization result to record
            session_id: Optional session identifier
        """
        with self._lock:
            # Update optimizer profile
            strategy = result.strategy
            if strategy not in self.optimizer_profiles:
                self.optimizer_profiles[strategy] = OptimizerPerformanceProfile(strategy=strategy)
            
            self.optimizer_profiles[strategy].update(result)
            
            # Update system metrics
            self._update_system_metrics(result)
            
            # Track session metrics if provided
            if session_id:
                self._update_session_metrics(result, session_id)
            
            # Check alerts
            self._check_alerts()
            
            logger.debug(f"Recorded optimization result: {strategy.value}, success: {result.success}, time: {result.metrics.execution_time:.2f}s")
    
    def _update_system_metrics(self, result: Any):
        """Update system-wide metrics"""
        current_time = time.time()
        
        # Update query count (we'll aggregate to per-minute in get_stats)
        self.system_metrics["total_queries_per_minute"].add_point(1.0)
        
        # Update response time
        self.system_metrics["avg_response_time"].add_point(result.metrics.execution_time)
        
        # Update API calls
        if result.metrics.api_calls_made > 0:
            self.system_metrics["total_api_calls_per_minute"].add_point(result.metrics.api_calls_made)
        
        # Update cache metrics
        total_cache_requests = result.metrics.cache_hits + result.metrics.cache_misses
        if total_cache_requests > 0:
            hit_rate = result.metrics.cache_hits / total_cache_requests
            self.system_metrics["cache_hit_rate"].add_point(hit_rate * 100)  # Convert to percentage
        
        # Update error rate
        error_rate = 1.0 if result.errors else 0.0
        self.system_metrics["error_rate"].add_point(error_rate)
        
        # Update quality score
        self.system_metrics["avg_quality_score"].add_point(result.metrics.quality_score)
    
    def _update_session_metrics(self, result: Any, session_id: str):
        """Update session-specific metrics"""
        if session_id not in self.session_metrics:
            self.session_metrics[session_id] = {
                "start_time": time.time(),
                "query_count": 0,
                "success_count": 0,
                "total_execution_time": 0.0,
                "total_results": 0,
                "strategies_used": set(),
                "avg_quality": 0.0
            }
        
        session_data = self.session_metrics[session_id]
        session_data["query_count"] += 1
        
        if result.success:
            session_data["success_count"] += 1
        
        session_data["total_execution_time"] += result.metrics.execution_time
        session_data["total_results"] += result.metrics.total_results
        session_data["strategies_used"].add(result.strategy.value)
        
        # Update average quality
        if session_data["success_count"] > 0:
            session_data["avg_quality"] = (
                (session_data["avg_quality"] * (session_data["success_count"] - 1) + result.metrics.quality_score)
                / session_data["success_count"]
            )
    
    def _check_alerts(self):
        """Check performance alerts"""
        # Get recent system metrics for alert checking
        response_times = self.system_metrics["avg_response_time"].get_recent_values(300)  # Last 5 minutes
        if response_times:
            avg_response_time = statistics.mean(response_times)
            for alert in self.alerts:
                if alert.name == "high_response_time":
                    alert.check(avg_response_time)
        
        # Check success rates across optimizers
        total_queries = sum(profile.total_queries for profile in self.optimizer_profiles.values())
        successful_queries = sum(profile.successful_queries for profile in self.optimizer_profiles.values())
        
        if total_queries > 0:
            system_success_rate = successful_queries / total_queries
            for alert in self.alerts:
                if alert.name == "low_success_rate":
                    alert.check(system_success_rate)
        
        # Check cache hit rates
        cache_hit_rates = self.system_metrics["cache_hit_rate"].get_recent_values(300)
        if cache_hit_rates:
            avg_cache_hit_rate = statistics.mean(cache_hit_rates) / 100.0  # Convert back from percentage
            for alert in self.alerts:
                if alert.name == "low_cache_hit_rate":
                    alert.check(avg_cache_hit_rate)
    
    def get_optimizer_performance(self, strategy: Any) -> Optional[OptimizerPerformanceProfile]:
        """Get performance profile for a specific optimizer"""
        with self._lock:
            return self.optimizer_profiles.get(strategy)
    
    def get_system_summary(self, time_window_hours: int = 1) -> Dict[str, Any]:
        """
        Get system-wide performance summary
        
        Args:
            time_window_hours: Time window to analyze (in hours)
            
        Returns:
            Dictionary with system performance summary
        """
        with self._lock:
            seconds = time_window_hours * 3600
            summary = {}
            
            # Overall statistics
            total_queries = sum(profile.total_queries for profile in self.optimizer_profiles.values())
            successful_queries = sum(profile.successful_queries for profile in self.optimizer_profiles.values())
            
            summary["total_queries"] = total_queries
            summary["successful_queries"] = successful_queries
            summary["success_rate"] = successful_queries / total_queries if total_queries > 0 else 0.0
            
            # Average metrics across optimizers
            if self.optimizer_profiles:
                summary["avg_execution_time"] = statistics.mean(
                    profile.avg_execution_time for profile in self.optimizer_profiles.values()
                    if profile.total_queries > 0
                ) if any(p.total_queries > 0 for p in self.optimizer_profiles.values()) else 0.0
                
                summary["avg_quality_score"] = statistics.mean(
                    profile.avg_quality_score for profile in self.optimizer_profiles.values()
                    if profile.successful_queries > 0
                ) if any(p.successful_queries > 0 for p in self.optimizer_profiles.values()) else 0.0
            
            # System metric stats
            for name, metric in self.system_metrics.items():
                stats = metric.get_stats(seconds)
                if stats:
                    summary[f"{name}_stats"] = stats
            
            # Alert summary
            summary["active_alerts"] = [
                {
                    "name": alert.name,
                    "triggered_count": alert.triggered_count,
                    "last_triggered": alert.last_triggered
                }
                for alert in self.alerts if alert.triggered_count > 0
            ]
            
            return summary
    
    def get_optimizer_comparison(self) -> Dict[str, Dict[str, float]]:
        """Get performance comparison across optimizers"""
        with self._lock:
            comparison = {}
            
            for strategy, profile in self.optimizer_profiles.items():
                if profile.total_queries > 0:
                    comparison[strategy.value] = {
                        "success_rate": profile.success_rate,
                        "avg_execution_time": profile.avg_execution_time,
                        "avg_results_per_query": profile.avg_results_per_query,
                        "avg_quality_score": profile.avg_quality_score,
                        "cache_hit_rate": profile.cache_hit_rate,
                        "total_queries": profile.total_queries
                    }
            
            return comparison
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get performance summary for a specific session"""
        with self._lock:
            if session_id not in self.session_metrics:
                return None
            
            session_data = self.session_metrics[session_id].copy()
            
            # Calculate derived metrics
            if session_data["query_count"] > 0:
                session_data["success_rate"] = session_data["success_count"] / session_data["query_count"]
                session_data["avg_execution_time"] = session_data["total_execution_time"] / session_data["query_count"]
                session_data["avg_results_per_query"] = session_data["total_results"] / session_data["query_count"]
            
            # Convert set to list for JSON serialization
            session_data["strategies_used"] = list(session_data["strategies_used"])
            
            # Add session duration
            session_data["session_duration"] = time.time() - session_data["start_time"]
            
            return session_data
    
    def get_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        with self._lock:
            # Analyze optimizer performance
            if len(self.optimizer_profiles) > 1:
                # Find best and worst performing optimizers
                profiles_with_data = {
                    strategy: profile for strategy, profile in self.optimizer_profiles.items()
                    if profile.total_queries >= 5  # Need sufficient data
                }
                
                if profiles_with_data:
                    # Find optimizer with highest success rate
                    best_optimizer = max(profiles_with_data.items(), key=lambda x: x[1].success_rate)
                    worst_optimizer = min(profiles_with_data.items(), key=lambda x: x[1].success_rate)
                    
                    if best_optimizer[1].success_rate - worst_optimizer[1].success_rate > 0.1:
                        recommendations.append(
                            f"Consider using {best_optimizer[0].value} optimizer more often - "
                            f"it has a {best_optimizer[1].success_rate:.1%} success rate vs "
                            f"{worst_optimizer[1].success_rate:.1%} for {worst_optimizer[0].value}"
                        )
                    
                    # Find fastest optimizer
                    fastest_optimizer = min(profiles_with_data.items(), key=lambda x: x[1].avg_execution_time)
                    slowest_optimizer = max(profiles_with_data.items(), key=lambda x: x[1].avg_execution_time)
                    
                    if slowest_optimizer[1].avg_execution_time > fastest_optimizer[1].avg_execution_time * 2:
                        recommendations.append(
                            f"For faster responses, consider {fastest_optimizer[0].value} optimizer - "
                            f"it averages {fastest_optimizer[1].avg_execution_time:.1f}s vs "
                            f"{slowest_optimizer[1].avg_execution_time:.1f}s for {slowest_optimizer[0].value}"
                        )
            
            # Analyze cache performance
            total_hits = sum(profile.total_cache_hits for profile in self.optimizer_profiles.values())
            total_misses = sum(profile.total_cache_misses for profile in self.optimizer_profiles.values())
            
            if total_hits + total_misses > 0:
                cache_hit_rate = total_hits / (total_hits + total_misses)
                if cache_hit_rate < 0.5:
                    recommendations.append(
                        f"Cache hit rate is low ({cache_hit_rate:.1%}). "
                        "Consider increasing cache TTL or reviewing query patterns."
                    )
            
            # Analyze error patterns
            all_errors = defaultdict(int)
            for profile in self.optimizer_profiles.values():
                for error, count in profile.error_counts.items():
                    all_errors[error] += count
            
            if all_errors:
                most_common_error = max(all_errors.items(), key=lambda x: x[1])
                if most_common_error[1] > 5:  # More than 5 occurrences
                    recommendations.append(
                        f"Most common error: '{most_common_error[0]}' ({most_common_error[1]} times). "
                        "Consider investigating root cause."
                    )
        
        return recommendations
    
    def export_metrics(self, format: str = "dict") -> Any:
        """Export all metrics in specified format"""
        with self._lock:
            if format == "dict":
                return {
                    "system_summary": self.get_system_summary(),
                    "optimizer_comparison": self.get_optimizer_comparison(),
                    "optimizer_profiles": {
                        strategy.value: {
                            "total_queries": profile.total_queries,
                            "successful_queries": profile.successful_queries,
                            "success_rate": profile.success_rate,
                            "avg_execution_time": profile.avg_execution_time,
                            "avg_quality_score": profile.avg_quality_score,
                            "total_api_calls": profile.total_api_calls,
                            "cache_hit_rate": profile.cache_hit_rate,
                            "error_counts": dict(profile.error_counts)
                        }
                        for strategy, profile in self.optimizer_profiles.items()
                    },
                    "recommendations": self.get_recommendations(),
                    "timestamp": time.time()
                }
        
        # Add more formats as needed (JSON, CSV, etc.)
        return None


# Singleton instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor