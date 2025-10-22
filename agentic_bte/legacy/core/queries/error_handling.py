"""
Enhanced Error Handling and Recovery System

This module provides comprehensive error handling, graceful fallbacks,
and recovery mechanisms for query optimizers.
"""

import logging
import time
import traceback
import functools
from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import asyncio

# Import at runtime to avoid circular imports

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Minor issues, can continue
    MEDIUM = "medium"     # Significant issues, may need fallback
    HIGH = "high"         # Critical issues, immediate fallback needed
    CRITICAL = "critical" # System-threatening issues


class ErrorCategory(Enum):
    """Categories of errors for handling strategies"""
    NETWORK = "network"                 # Network connectivity issues
    API_LIMIT = "api_limit"            # API rate limits or quotas
    TIMEOUT = "timeout"                # Request timeouts
    AUTHENTICATION = "authentication"  # Auth/permission issues
    DATA_FORMAT = "data_format"        # Data parsing/format issues
    RESOURCE = "resource"              # Resource constraints (memory, etc.)
    LOGIC = "logic"                    # Business logic errors
    EXTERNAL_SERVICE = "external_service"  # External service failures
    CONFIGURATION = "configuration"    # Configuration errors
    UNKNOWN = "unknown"                # Unclassified errors


@dataclass
class ErrorPattern:
    """Pattern for matching and handling specific errors"""
    pattern: str                       # Error message pattern to match
    category: ErrorCategory
    severity: ErrorSeverity
    retry_strategy: str = "exponential"  # exponential, linear, immediate, none
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    fallback_strategy: Optional[str] = None
    
    def matches(self, error_message: str) -> bool:
        """Check if error message matches this pattern"""
        return self.pattern.lower() in error_message.lower()


@dataclass
class ErrorInstance:
    """Instance of an error that occurred"""
    timestamp: float
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    strategy: Any  # OptimizationStrategy at runtime
    query: str
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_exception(cls, exception: Exception, strategy: Any, 
                      query: str, context: Dict[str, Any] = None) -> 'ErrorInstance':
        """Create ErrorInstance from an exception"""
        error_handler = get_error_handler()
        category, severity = error_handler.classify_error(exception)
        
        return cls(
            timestamp=time.time(),
            error_type=type(exception).__name__,
            error_message=str(exception),
            category=category,
            severity=severity,
            strategy=strategy,
            query=query,
            traceback=traceback.format_exc(),
            context=context or {}
        )


@dataclass
class RecoveryAction:
    """Represents a recovery action that can be taken"""
    name: str
    description: str
    action: Callable
    conditions: List[Callable] = field(default_factory=list)
    success_rate: float = 0.0
    avg_recovery_time: float = 0.0
    usage_count: int = 0
    
    def can_execute(self, error: ErrorInstance) -> bool:
        """Check if this recovery action can be executed for the given error"""
        return all(condition(error) for condition in self.conditions)


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker recovered - transitioning to CLOSED")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise


class ErrorHandler:
    """
    Comprehensive error handling system for query optimizers
    
    Provides error classification, recovery strategies, circuit breakers,
    and intelligent fallback mechanisms.
    """
    
    def __init__(self):
        """Initialize the error handler"""
        # Error patterns for classification
        self.error_patterns = self._initialize_error_patterns()
        
        # Error tracking
        self.error_history: List[ErrorInstance] = []
        self.error_counts: Dict[ErrorCategory, int] = defaultdict(int)
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        
        # Circuit breakers for different services
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            "bte_api": CircuitBreaker(failure_threshold=5, recovery_timeout=300.0),
            "openai_api": CircuitBreaker(failure_threshold=3, recovery_timeout=600.0),
            "entity_extraction": CircuitBreaker(failure_threshold=5, recovery_timeout=180.0),
        }
        
        # Recovery strategies
        self._initialize_recovery_actions()
        
        logger.info("Error handler initialized")
    
    def _initialize_error_patterns(self) -> List[ErrorPattern]:
        """Initialize error patterns for classification"""
        return [
            # Network errors
            ErrorPattern("connection timeout", ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, max_retries=3, base_delay=2.0),
            ErrorPattern("connection refused", ErrorCategory.NETWORK, ErrorSeverity.HIGH, max_retries=2, base_delay=5.0),
            ErrorPattern("network unreachable", ErrorCategory.NETWORK, ErrorSeverity.HIGH, max_retries=2, base_delay=10.0),
            ErrorPattern("dns resolution", ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, max_retries=3, base_delay=1.0),
            
            # API limits
            ErrorPattern("rate limit", ErrorCategory.API_LIMIT, ErrorSeverity.MEDIUM, max_retries=3, base_delay=60.0),
            ErrorPattern("quota exceeded", ErrorCategory.API_LIMIT, ErrorSeverity.HIGH, max_retries=2, base_delay=300.0),
            ErrorPattern("too many requests", ErrorCategory.API_LIMIT, ErrorSeverity.MEDIUM, max_retries=3, base_delay=30.0),
            
            # Timeouts
            ErrorPattern("request timeout", ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM, max_retries=2, base_delay=5.0),
            ErrorPattern("read timeout", ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM, max_retries=2, base_delay=3.0),
            ErrorPattern("execution timeout", ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM, max_retries=1, base_delay=10.0),
            
            # Authentication
            ErrorPattern("unauthorized", ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH, max_retries=1, base_delay=1.0),
            ErrorPattern("invalid api key", ErrorCategory.AUTHENTICATION, ErrorSeverity.CRITICAL, max_retries=0),
            ErrorPattern("permission denied", ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH, max_retries=1, base_delay=1.0),
            
            # Data format
            ErrorPattern("json decode error", ErrorCategory.DATA_FORMAT, ErrorSeverity.MEDIUM, max_retries=1, base_delay=1.0),
            ErrorPattern("invalid response", ErrorCategory.DATA_FORMAT, ErrorSeverity.MEDIUM, max_retries=2, base_delay=1.0),
            ErrorPattern("parsing error", ErrorCategory.DATA_FORMAT, ErrorSeverity.MEDIUM, max_retries=1, base_delay=1.0),
            
            # Resource constraints
            ErrorPattern("out of memory", ErrorCategory.RESOURCE, ErrorSeverity.HIGH, max_retries=1, base_delay=5.0),
            ErrorPattern("disk space", ErrorCategory.RESOURCE, ErrorSeverity.HIGH, max_retries=0),
            ErrorPattern("resource exhausted", ErrorCategory.RESOURCE, ErrorSeverity.HIGH, max_retries=1, base_delay=10.0),
            
            # External services
            ErrorPattern("service unavailable", ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.HIGH, max_retries=3, base_delay=30.0),
            ErrorPattern("internal server error", ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.MEDIUM, max_retries=2, base_delay=10.0),
            ErrorPattern("bad gateway", ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.MEDIUM, max_retries=3, base_delay=5.0),
            
            # Configuration
            ErrorPattern("missing configuration", ErrorCategory.CONFIGURATION, ErrorSeverity.CRITICAL, max_retries=0),
            ErrorPattern("invalid configuration", ErrorCategory.CONFIGURATION, ErrorSeverity.CRITICAL, max_retries=0),
        ]
    
    def _initialize_recovery_actions(self):
        """Initialize recovery actions"""
        self.recovery_actions = {
            "retry_with_backoff": RecoveryAction(
                name="retry_with_backoff",
                description="Retry with exponential backoff",
                action=self._retry_with_backoff,
                conditions=[lambda e: e.severity != ErrorSeverity.CRITICAL]
            ),
            "fallback_to_basic": RecoveryAction(
                name="fallback_to_basic",
                description="Fallback to basic adaptive optimizer",
                action=self._fallback_to_basic_optimizer,
                conditions=[lambda e: e.category in [ErrorCategory.EXTERNAL_SERVICE, ErrorCategory.NETWORK]]
            ),
            "reduce_complexity": RecoveryAction(
                name="reduce_complexity",
                description="Reduce query complexity and retry",
                action=self._reduce_query_complexity,
                conditions=[lambda e: e.category in [ErrorCategory.TIMEOUT, ErrorCategory.RESOURCE]]
            ),
            "use_cache_only": RecoveryAction(
                name="use_cache_only",
                description="Use cached results only",
                action=self._use_cache_only,
                conditions=[lambda e: e.category in [ErrorCategory.API_LIMIT, ErrorCategory.NETWORK]]
            ),
            "refresh_auth": RecoveryAction(
                name="refresh_auth",
                description="Refresh authentication and retry",
                action=self._refresh_authentication,
                conditions=[lambda e: e.category == ErrorCategory.AUTHENTICATION]
            ),
        }
    
    def classify_error(self, error: Union[Exception, str]) -> tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify error into category and severity
        
        Args:
            error: Exception or error message string
            
        Returns:
            Tuple of (ErrorCategory, ErrorSeverity)
        """
        error_message = str(error)
        
        # Check against known patterns
        for pattern in self.error_patterns:
            if pattern.matches(error_message):
                return pattern.category, pattern.severity
        
        # Default classification based on exception type
        if isinstance(error, Exception):
            error_type = type(error).__name__.lower()
            
            if "timeout" in error_type:
                return ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM
            elif "connection" in error_type or "network" in error_type:
                return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
            elif "auth" in error_type or "permission" in error_type:
                return ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH
            elif "json" in error_type or "parse" in error_type:
                return ErrorCategory.DATA_FORMAT, ErrorSeverity.MEDIUM
            elif "memory" in error_type or "resource" in error_type:
                return ErrorCategory.RESOURCE, ErrorSeverity.HIGH
        
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    def handle_error(self, error: Exception, strategy: Any, 
                    query: str, context: Dict[str, Any] = None) -> Any:
        """
        Handle error with appropriate recovery strategy
        
        Args:
            error: The exception that occurred
            strategy: The optimization strategy that failed
            query: The query being processed
            context: Additional context information
            
        Returns:
            OptimizationResult (potentially recovered or fallback)
        """
        # Create error instance
        error_instance = ErrorInstance.from_exception(error, strategy, query, context)
        
        # Record error
        self.record_error(error_instance)
        
        # Find applicable recovery actions
        recovery_actions = [
            action for action in self.recovery_actions.values()
            if action.can_execute(error_instance)
        ]
        
        # Sort by success rate (highest first)
        recovery_actions.sort(key=lambda a: a.success_rate, reverse=True)
        
        # Try recovery actions in order
        for action in recovery_actions:
            try:
                logger.info(f"Attempting recovery action: {action.name}")
                result = action.action(error_instance, strategy, query, context)
                
                if result and result.success:
                    # Update recovery action statistics
                    action.usage_count += 1
                    action.success_rate = (action.success_rate * (action.usage_count - 1) + 1.0) / action.usage_count
                    
                    result.warnings.append(f"Recovered using {action.description}")
                    logger.info(f"Successfully recovered using {action.name}")
                    return result
                
            except Exception as recovery_error:
                logger.warning(f"Recovery action {action.name} failed: {recovery_error}")
        
        # All recovery actions failed - return error result
        return self._create_error_result(error_instance, strategy, query)
    
    def record_error(self, error_instance: ErrorInstance):
        """Record an error instance for tracking and analysis"""
        self.error_history.append(error_instance)
        self.error_counts[error_instance.category] += 1
        
        # Keep only last 1000 errors to prevent memory growth
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        logger.error(f"Error recorded: {error_instance.category.value} - {error_instance.error_message}")
    
    def _retry_with_backoff(self, error_instance: ErrorInstance, strategy: Any, 
                           query: str, context: Dict[str, Any]) -> Any:
        """Retry with exponential backoff"""
        # This is a placeholder - actual retry would be handled by the calling optimizer
        # The optimizer would use this error handler to determine retry parameters
        pattern = self._find_matching_pattern(error_instance.error_message)
        
        if pattern and pattern.max_retries > 0:
            # Import at runtime to avoid circular imports
            from .interfaces import OptimizationResult
            
            # Return a result indicating retry should be attempted
            result = OptimizationResult(
                query=query,
                strategy=strategy,
                success=False,
                start_time=time.time()
            )
            result.warnings.append(f"Retry recommended with {pattern.base_delay}s delay")
            result.finalize()
            return result
        
        return None
    
    def _fallback_to_basic_optimizer(self, error_instance: ErrorInstance, strategy: Any,
                                   query: str, context: Dict[str, Any]) -> Any:
        """Fallback to basic adaptive optimizer"""
        # Check if already using basic strategy (string comparison)
        strategy_value = strategy.value if hasattr(strategy, 'value') else str(strategy)
        if strategy_value == 'basic_adaptive':
            return None  # Already using basic, can't fallback further
        
        # Import at runtime to avoid circular imports
        from .interfaces import OptimizationResult, OptimizationStrategy
        
        # This would trigger a fallback to basic optimizer
        # The actual implementation would be handled by the calling system
        result = OptimizationResult(
            query=query,
            strategy=OptimizationStrategy.BASIC_ADAPTIVE,
            success=False,
            start_time=time.time()
        )
        result.warnings.append("Fallback to basic optimizer recommended")
        result.finalize()
        return result
    
    def _reduce_query_complexity(self, error_instance: ErrorInstance, strategy: Any,
                                query: str, context: Dict[str, Any]) -> Any:
        """Reduce query complexity"""
        # Import at runtime to avoid circular imports
        from .interfaces import OptimizationResult
        
        # This would modify the query or parameters to reduce complexity
        result = OptimizationResult(
            query=query,
            strategy=strategy,
            success=False,
            start_time=time.time()
        )
        result.warnings.append("Query complexity reduction recommended")
        result.finalize()
        return result
    
    def _use_cache_only(self, error_instance: ErrorInstance, strategy: Any,
                       query: str, context: Dict[str, Any]) -> Any:
        """Use only cached results"""
        # Import at runtime to avoid circular imports
        from .interfaces import OptimizationResult
        
        result = OptimizationResult(
            query=query,
            strategy=strategy,
            success=False,
            start_time=time.time()
        )
        result.warnings.append("Cache-only mode recommended")
        result.finalize()
        return result
    
    def _refresh_authentication(self, error_instance: ErrorInstance, strategy: Any,
                              query: str, context: Dict[str, Any]) -> Any:
        """Refresh authentication"""
        # Import at runtime to avoid circular imports
        from .interfaces import OptimizationResult
        
        result = OptimizationResult(
            query=query,
            strategy=strategy,
            success=False,
            start_time=time.time()
        )
        result.warnings.append("Authentication refresh recommended")
        result.finalize()
        return result
    
    def _find_matching_pattern(self, error_message: str) -> Optional[ErrorPattern]:
        """Find the error pattern that matches the given error message"""
        for pattern in self.error_patterns:
            if pattern.matches(error_message):
                return pattern
        return None
    
    def _create_error_result(self, error_instance: ErrorInstance, strategy: Any,
                           query: str) -> Any:
        """Create an error result when recovery fails"""
        # Import at runtime to avoid circular imports
        from .interfaces import OptimizationResult
        
        result = OptimizationResult(
            query=query,
            strategy=strategy,
            success=False,
            start_time=time.time()
        )
        
        result.errors.append(f"{error_instance.error_type}: {error_instance.error_message}")
        result.warnings.append(f"Error category: {error_instance.category.value}")
        result.warnings.append(f"Error severity: {error_instance.severity.value}")
        
        result.finalize()
        return result
    
    def get_error_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for analysis"""
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
        
        if not recent_errors:
            return {"total_errors": 0, "time_window_hours": time_window_hours}
        
        # Category breakdown
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        strategy_counts = defaultdict(int)
        
        for error in recent_errors:
            category_counts[error.category.value] += 1
            severity_counts[error.severity.value] += 1
            strategy_counts[error.strategy.value] += 1
        
        # Recovery action statistics
        recovery_stats = {
            name: {
                "usage_count": action.usage_count,
                "success_rate": action.success_rate,
                "avg_recovery_time": action.avg_recovery_time
            }
            for name, action in self.recovery_actions.items()
        }
        
        return {
            "total_errors": len(recent_errors),
            "time_window_hours": time_window_hours,
            "category_breakdown": dict(category_counts),
            "severity_breakdown": dict(severity_counts),
            "strategy_breakdown": dict(strategy_counts),
            "recovery_actions": recovery_stats,
            "error_rate_per_hour": len(recent_errors) / time_window_hours if time_window_hours > 0 else 0
        }
    
    def get_recommendations(self) -> List[str]:
        """Get error handling recommendations"""
        recommendations = []
        
        stats = self.get_error_statistics(24)  # Last 24 hours
        
        if stats["total_errors"] == 0:
            return ["No errors detected in the last 24 hours - system is performing well."]
        
        # High error rate
        if stats["error_rate_per_hour"] > 5:
            recommendations.append(
                f"High error rate detected: {stats['error_rate_per_hour']:.1f} errors/hour. "
                "Consider investigating root causes."
            )
        
        # Category-specific recommendations
        category_breakdown = stats["category_breakdown"]
        
        if category_breakdown.get("network", 0) > 10:
            recommendations.append(
                "High number of network errors detected. "
                "Consider checking network connectivity and adding more retry logic."
            )
        
        if category_breakdown.get("api_limit", 0) > 5:
            recommendations.append(
                "API rate limit errors detected. "
                "Consider implementing request throttling or increasing API quotas."
            )
        
        if category_breakdown.get("timeout", 0) > 5:
            recommendations.append(
                "Timeout errors detected. "
                "Consider increasing timeout values or optimizing query complexity."
            )
        
        # Recovery action effectiveness
        recovery_stats = stats["recovery_actions"]
        low_success_actions = [
            name for name, stats in recovery_stats.items()
            if stats["usage_count"] > 0 and stats["success_rate"] < 0.5
        ]
        
        if low_success_actions:
            recommendations.append(
                f"Low success rate recovery actions: {', '.join(low_success_actions)}. "
                "Consider reviewing or improving these recovery strategies."
            )
        
        return recommendations


# Decorator for automatic error handling
def handle_errors(strategy: Any = None, 
                 context: Dict[str, Any] = None):
    """
    Decorator to automatically handle errors in optimizer methods
    
    Args:
        strategy: The optimization strategy being used
        context: Additional context for error handling
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                query = kwargs.get('query', args[1] if len(args) > 1 else 'Unknown')
                return error_handler.handle_error(e, strategy, query, context)
        return wrapper
    return decorator


# Singleton instance
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler