"""
Unified Configuration System

This module provides a comprehensive configuration system that consolidates
all configuration options from different optimizers and strategies into a 
single, coherent configuration interface.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import os


logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Available caching backends"""
    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"
    DISABLED = "disabled"


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class PerformanceConfig:
    """Performance-related configuration settings"""
    # Parallel execution settings
    enable_parallel_execution: bool = True
    max_concurrent_calls: int = 5
    max_concurrent_predicates: int = 3
    max_concurrent_queries: int = 5
    max_concurrent_api_calls: int = 10
    enable_async_processing: bool = True
    max_worker_threads: int = 4
    max_worker_processes: int = 2
    
    # Timeout settings
    query_timeout_seconds: int = 300
    api_timeout_seconds: int = 60
    subquery_timeout_seconds: int = 120
    
    # Resource limits
    memory_limit_mb: int = 2048
    max_results_per_query: int = 100
    max_entities_per_query: int = 50
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0


@dataclass 
class QualityConfig:
    """Quality and confidence thresholds"""
    # Confidence thresholds
    confidence_threshold: float = 0.7
    quality_threshold: float = 0.8
    entity_confidence_threshold: float = 0.5
    
    # Iteration limits
    max_iterations: int = 10
    max_subqueries: int = 20
    max_refinement_iterations: int = 3
    
    # Quality assessment
    enable_quality_assessment: bool = True
    quality_improvement_threshold: float = 0.1
    minimum_evidence_score: float = 0.3
    
    # Evidence scoring weights
    evidence_weight: float = 0.3
    predicate_weight: float = 0.25
    source_weight: float = 0.2
    multiplicity_weight: float = 0.15
    study_weight: float = 0.1
    
    # Result thresholds
    min_results_threshold: int = 10
    fallback_threshold: float = 0.3


@dataclass
class CachingConfig:
    """Caching configuration settings"""
    enable_caching: bool = True
    backend: CacheBackend = CacheBackend.MEMORY
    cache_ttl: int = 3600  # 1 hour
    max_cache_size_mb: int = 512
    
    # Redis-specific settings (if backend is REDIS)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # File cache settings (if backend is FILE)
    cache_directory: str = "/tmp/agentic_bte_cache"
    
    # Cache keys configuration
    cache_query_results: bool = True
    cache_entity_extractions: bool = True
    cache_trapi_queries: bool = True
    cache_bte_responses: bool = True


@dataclass
class DomainConfig:
    """Biomedical domain-specific settings"""
    # Entity processing
    enable_entity_extraction: bool = True
    enable_entity_linking: bool = True
    enable_entity_resolution: bool = True
    enable_generic_entity_mapping: bool = True
    
    # Knowledge graph features
    enable_evidence_scoring: bool = True
    enable_rdf_accumulation: bool = True
    enable_predicate_selection: bool = True
    enable_meta_kg_filtering: bool = True
    
    # Domain expertise
    enable_domain_expertise: bool = True
    enable_pharmaceutical_expertise: bool = True
    enable_mechanistic_reasoning: bool = True
    enable_clinical_expertise: bool = True
    
    # Entity type preferences
    preferred_entity_sources: List[str] = field(default_factory=lambda: [
        "CHEBI", "DRUGBANK", "HGNC", "ENSEMBL", "MONDO", "DOID"
    ])
    
    # Additional predicate configuration
    max_predicates_per_query: int = 5
    min_results_threshold: int = 10
    fallback_threshold: float = 0.3
    
    # Predicate preferences by query intent
    therapeutic_predicates: List[str] = field(default_factory=lambda: [
        "biolink:treats", "biolink:related_to", "biolink:associated_with"
    ])
    genetic_predicates: List[str] = field(default_factory=lambda: [
        "biolink:gene_associated_with_condition", "biolink:related_to"
    ])
    mechanism_predicates: List[str] = field(default_factory=lambda: [
        "biolink:affects", "biolink:interacts_with", "biolink:related_to"
    ])


@dataclass
class IntegrationConfig:
    """External service integration settings"""
    # BTE settings
    bte_url: str = "http://localhost:3000"
    enable_local_bte: bool = True
    enable_remote_bte: bool = False
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 2000
    
    # MCP settings
    enable_mcp_integration: bool = True
    mcp_timeout: int = 60
    mcp_max_retries: int = 3
    
    # Name resolution services
    enable_name_resolution: bool = True
    name_resolver_url: str = "https://name-lookup.ci.transltr.io"
    
    # External knowledge bases
    enable_external_kb: bool = False
    external_kb_urls: List[str] = field(default_factory=list)


@dataclass
class DebugConfig:
    """Debugging and logging configuration"""
    log_level: LogLevel = LogLevel.INFO
    enable_debug_mode: bool = False
    enable_query_debugging: bool = True
    enable_performance_profiling: bool = False
    
    # Output settings
    save_intermediate_results: bool = False
    save_trapi_queries: bool = False
    save_execution_traces: bool = False
    
    # Logging destinations
    log_to_console: bool = True
    log_to_file: bool = False
    log_file_path: str = "agentic_bte.log"
    
    # Debug output formatting
    show_detailed_errors: bool = True
    show_timing_info: bool = True
    show_memory_usage: bool = False


@dataclass
class UnifiedConfig:
    """
    Comprehensive unified configuration for all agentic-bte components
    
    This configuration system consolidates settings from all different
    optimizers and strategies into a single, coherent interface.
    """
    
    # Strategy selection
    strategy: str = 'got_framework'
    fallback_strategies: list[str] = field(default_factory=lambda: [
        'simple',
        'langgraph',
        'hybrid'
    ])
    
    # Component configurations
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    
    # Global settings
    environment: str = "development"
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Post-initialization validation and environment variable loading"""
        self._load_from_environment()
        self._validate_configuration()
        self._setup_logging()
    
    def _load_from_environment(self):
        """Load configuration values from environment variables"""
        # OpenAI API key
        if not self.integration.openai_api_key:
            self.integration.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # BTE URL
        bte_url = os.getenv("BTE_URL")
        if bte_url:
            self.integration.bte_url = bte_url
        
        # Environment
        env = os.getenv("AGENTIC_BTE_ENV")
        if env:
            self.environment = env
        
        # Debug mode
        debug_mode = os.getenv("AGENTIC_BTE_DEBUG", "").lower()
        if debug_mode in ["true", "1", "yes"]:
            self.debug.enable_debug_mode = True
            self.debug.log_level = LogLevel.DEBUG
        
        # Redis settings (if using Redis cache)
        if self.caching.backend == CacheBackend.REDIS:
            redis_host = os.getenv("REDIS_HOST")
            if redis_host:
                self.caching.redis_host = redis_host
            
            redis_port = os.getenv("REDIS_PORT")
            if redis_port:
                self.caching.redis_port = int(redis_port)
            
            redis_password = os.getenv("REDIS_PASSWORD")
            if redis_password:
                self.caching.redis_password = redis_password
    
    def _validate_configuration(self):
        """Validate configuration settings"""
        # Validate timeouts
        if self.performance.query_timeout_seconds <= 0:
            raise ValueError("query_timeout_seconds must be positive")
        
        if self.performance.api_timeout_seconds <= 0:
            raise ValueError("api_timeout_seconds must be positive")
        
        # Validate thresholds
        if not (0.0 <= self.quality.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.quality.quality_threshold <= 1.0):
            raise ValueError("quality_threshold must be between 0.0 and 1.0")
        
        # Validate OpenAI configuration
        if not self.integration.openai_api_key:
            logger.warning("OpenAI API key not configured. Some features may not work.")
        
        # Validate concurrent limits
        if self.performance.max_concurrent_calls <= 0:
            raise ValueError("max_concurrent_calls must be positive")
        
        # Validate cache settings
        if self.caching.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")
    
    def _setup_logging(self):
        """Configure logging based on debug settings"""
        log_level = getattr(logging, self.debug.log_level.value)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set specific logger levels
        if self.debug.enable_debug_mode:
            logging.getLogger("agentic_bte").setLevel(logging.DEBUG)
        else:
            logging.getLogger("agentic_bte").setLevel(log_level)
    
    def get_strategy_config(self, strategy: str) -> Dict[str, Any]:
        """Get strategy-specific configuration"""
        base_config = {
            "performance": self.performance,
            "quality": self.quality,
            "caching": self.caching,
            "domain": self.domain,
            "integration": self.integration,
            "debug": self.debug
        }
        
        # Strategy-specific overrides
        strategy_configs = {
            "basic_adaptive": {
                "enable_parallel_execution": False,
                "max_iterations": 1,
                "enable_rdf_accumulation": False
            },
            "meta_kg_aware": {
                "enable_parallel_execution": True,
                "max_iterations": 5,
                "enable_quality_assessment": True
            },
            "placeholder_enhanced": {
                "enable_rdf_accumulation": True,
                "enable_domain_expertise": True,
                "max_iterations": 10
            },
            "parallel_execution": {
                "enable_parallel_execution": True,
                "enable_evidence_scoring": True,
                "max_concurrent_predicates": 4
            }
        }
        
        # Apply strategy-specific overrides
        if strategy in strategy_configs:
            overrides = strategy_configs[strategy]
            for key, value in overrides.items():
                if hasattr(self.performance, key):
                    setattr(self.performance, key, value)
                elif hasattr(self.quality, key):
                    setattr(self.quality, key, value)
                elif hasattr(self.domain, key):
                    setattr(self.domain, key, value)
        
        return base_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "strategy": self.strategy,
            "fallback_strategies": self.fallback_strategies,
            "performance": {
                "enable_parallel_execution": self.performance.enable_parallel_execution,
                "max_concurrent_calls": self.performance.max_concurrent_calls,
                "query_timeout_seconds": self.performance.query_timeout_seconds,
                "memory_limit_mb": self.performance.memory_limit_mb,
                "max_results_per_query": self.performance.max_results_per_query,
                "max_retries": self.performance.max_retries
            },
            "quality": {
                "confidence_threshold": self.quality.confidence_threshold,
                "quality_threshold": self.quality.quality_threshold,
                "max_iterations": self.quality.max_iterations,
                "max_subqueries": self.quality.max_subqueries
            },
            "caching": {
                "enable_caching": self.caching.enable_caching,
                "backend": self.caching.backend.value,
                "cache_ttl": self.caching.cache_ttl
            },
            "domain": {
                "enable_evidence_scoring": self.domain.enable_evidence_scoring,
                "enable_rdf_accumulation": self.domain.enable_rdf_accumulation,
                "enable_domain_expertise": self.domain.enable_domain_expertise
            },
            "integration": {
                "bte_url": self.integration.bte_url,
                "openai_model": self.integration.openai_model,
                "enable_mcp_integration": self.integration.enable_mcp_integration
            },
            "environment": self.environment,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UnifiedConfig':
        """Create configuration from dictionary"""
        config = cls()
        
        # Update strategy
        if "strategy" in config_dict:
            config.strategy = config_dict["strategy"]
        
        if "fallback_strategies" in config_dict:
            config.fallback_strategies = config_dict["fallback_strategies"]
        
        # Update component configs
        if "performance" in config_dict:
            perf = config_dict["performance"]
            for key, value in perf.items():
                if hasattr(config.performance, key):
                    setattr(config.performance, key, value)
        
        if "quality" in config_dict:
            qual = config_dict["quality"]
            for key, value in qual.items():
                if hasattr(config.quality, key):
                    setattr(config.quality, key, value)
        
        # Continue for other components...
        
        return config
    
    def copy(self) -> 'UnifiedConfig':
        """Create a copy of the configuration"""
        import copy
        return copy.deepcopy(self)
    
    def merge(self, other: 'UnifiedConfig') -> 'UnifiedConfig':
        """Merge with another configuration, with other taking precedence"""
        merged = self.copy()
        
        # Merge strategy settings
        merged.strategy = other.strategy
        merged.fallback_strategies = other.fallback_strategies
        
        # Merge component configurations
        # This would need detailed implementation for each component
        # For now, just replace the components entirely
        merged.performance = other.performance
        merged.quality = other.quality
        merged.caching = other.caching
        merged.domain = other.domain
        merged.integration = other.integration
        merged.debug = other.debug
        
        return merged


# Convenience function for creating common configurations
def create_development_config() -> UnifiedConfig:
    """Create configuration optimized for development"""
    config = UnifiedConfig()
    config.environment = "development"
    config.debug.enable_debug_mode = True
    config.debug.log_level = LogLevel.DEBUG
    config.debug.save_intermediate_results = True
    config.performance.query_timeout_seconds = 120
    config.caching.enable_caching = False  # Disable caching for development
    return config


def create_production_config() -> UnifiedConfig:
    """Create configuration optimized for production"""
    config = UnifiedConfig()
    config.environment = "production"
    config.strategy = "parallel_execution"
    config.debug.enable_debug_mode = False
    config.debug.log_level = LogLevel.INFO
    config.performance.max_concurrent_calls = 10
    config.caching.enable_caching = True
    config.caching.backend = CacheBackend.REDIS
    return config


def create_testing_config() -> UnifiedConfig:
    """Create configuration optimized for testing"""
    config = UnifiedConfig()
    config.environment = "testing"
    config.debug.enable_debug_mode = True
    config.performance.query_timeout_seconds = 30
    config.performance.max_concurrent_calls = 2
    config.caching.enable_caching = False
    config.quality.max_iterations = 3
    return config