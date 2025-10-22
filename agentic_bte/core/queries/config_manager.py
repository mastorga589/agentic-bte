"""
Unified Configuration Management for Query Optimizers

This module provides a centralized configuration system for all optimizer types
with validation, defaults, and environment-aware settings.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

from ...config.settings import get_settings

logger = logging.getLogger(__name__)


class ConfigScope(Enum):
    """Configuration scope levels"""
    GLOBAL = "global"           # System-wide settings
    OPTIMIZER = "optimizer"     # Per optimizer type settings
    SESSION = "session"         # Per session settings
    QUERY = "query"             # Per query settings


@dataclass
class OptimizerConfig:
    """Base configuration for all optimizer types with validation"""
    # Results and filtering
    max_results: int = 50          # Maximum total results to return
    k: int = 5                     # Maximum results per entity
    confidence_threshold: float = 0.7  # Minimum confidence threshold for results
    
    # Execution controls
    max_iterations: int = 5        # Maximum iterations for adaptive planning
    max_retries: int = 2           # Maximum retry attempts
    timeout: int = 300             # Maximum execution time in seconds
    
    # Optimization settings
    enable_caching: bool = True    # Whether to use result caching
    cache_ttl: int = 3600          # Cache TTL in seconds (1 hour default)
    max_concurrent_batches: int = 3  # Maximum concurrent batches
    
    # Advanced settings
    fallback_on_error: bool = True  # Whether to use fallbacks on error
    detailed_metrics: bool = True   # Whether to track detailed metrics
    debug_mode: bool = False        # Enable additional debug output
    
    # Custom parameters (for specific optimizers)
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self):
        """Validate configuration values and adjust if needed"""
        # Ensure valid ranges for numeric values
        self.max_results = max(1, min(1000, self.max_results))
        self.k = max(1, min(50, self.k))
        self.confidence_threshold = max(0.0, min(1.0, self.confidence_threshold))
        self.max_iterations = max(1, min(20, self.max_iterations))
        self.max_retries = max(0, min(5, self.max_retries))
        self.timeout = max(30, min(1800, self.timeout))  # 30s to 30m
        self.max_concurrent_batches = max(1, min(6, self.max_concurrent_batches))
        self.cache_ttl = max(60, min(86400, self.cache_ttl))  # 1m to 24h
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def update(self, updates: Dict[str, Any]) -> 'OptimizerConfig':
        """Update configuration with new values"""
        # Apply updates that match our fields
        for key, value in updates.items():
            if hasattr(self, key) and key != 'custom_params':
                setattr(self, key, value)
        
        # Handle custom_params separately (merge instead of replace)
        if 'custom_params' in updates and isinstance(updates['custom_params'], dict):
            self.custom_params.update(updates['custom_params'])
        
        # Re-validate after updates
        self.validate()
        return self
    
    def with_overrides(self, **kwargs) -> 'OptimizerConfig':
        """Create a new config with specific overrides"""
        new_config = OptimizerConfig(**self.to_dict())
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
        new_config.validate()
        return new_config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OptimizerConfig':
        """Create configuration from dictionary"""
        # Filter out unknown keys to prevent __init__ errors
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    @classmethod
    def from_env(cls) -> 'OptimizerConfig':
        """Load configuration from environment variables"""
        settings = get_settings()
        
        # Load from configured settings
        config = cls(
            max_results=getattr(settings, "bte_max_results", 50),
            k=getattr(settings, "bte_k", 5),
            confidence_threshold=getattr(settings, "bte_confidence_threshold", 0.7),
            max_iterations=getattr(settings, "optimizer_max_iterations", 5),
            max_retries=getattr(settings, "optimizer_max_retries", 2),
            timeout=getattr(settings, "optimizer_timeout", 300),
            enable_caching=getattr(settings, "enable_result_caching", True),
            cache_ttl=getattr(settings, "cache_ttl", 3600),
            max_concurrent_batches=getattr(settings, "max_concurrent_batches", 3),
            fallback_on_error=getattr(settings, "optimizer_fallback_on_error", True),
            detailed_metrics=getattr(settings, "detailed_metrics", True),
            debug_mode=getattr(settings, "debug_mode", False)
        )
        
        # Override with direct environment variables if present
        env_prefix = "OPTIMIZER_"
        for key in dir(config):
            if key.startswith("_") or callable(getattr(config, key)):
                continue
            
            env_var = f"{env_prefix}{key.upper()}"
            if env_var in os.environ:
                # Convert to appropriate type
                env_value = os.environ[env_var]
                current_value = getattr(config, key)
                
                if isinstance(current_value, bool):
                    setattr(config, key, env_value.lower() in ('true', '1', 'yes'))
                elif isinstance(current_value, int):
                    setattr(config, key, int(env_value))
                elif isinstance(current_value, float):
                    setattr(config, key, float(env_value))
                else:
                    setattr(config, key, env_value)
        
        # Custom parameters from environment
        custom_params_env = os.environ.get(f"{env_prefix}CUSTOM_PARAMS")
        if custom_params_env:
            try:
                custom_params = json.loads(custom_params_env)
                config.custom_params.update(custom_params)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse {env_prefix}CUSTOM_PARAMS as JSON")
        
        return config


class ConfigManager:
    """
    Centralized configuration manager for query optimizers
    
    This manager handles configuration across different scopes and optimizer types,
    with support for defaults, overrides, and validation.
    """
    
    def __init__(self):
        """Initialize the configuration manager"""
        # Load base configurations
        self.global_config = OptimizerConfig.from_env()
        
        # Optimizer-specific configurations
        self.optimizer_configs: Dict[str, OptimizerConfig] = {}
        
        # Session and query configurations
        self.session_configs: Dict[str, OptimizerConfig] = {}
        self.query_configs: Dict[str, Dict[str, OptimizerConfig]] = {}
        
        # Load optimizer-specific configurations
        self._load_optimizer_configs()
        
        logger.info("Configuration manager initialized")
    
    def _load_optimizer_configs(self):
        """Load optimizer-specific configurations"""
        settings = get_settings()
        optimizer_configs = getattr(settings, "optimizer_configs", {})
        
        for optimizer_type, config_dict in optimizer_configs.items():
            try:
                # Start with global config as base
                optimizer_config = OptimizerConfig(**self.global_config.to_dict())
                # Apply optimizer-specific overrides
                optimizer_config.update(config_dict)
                self.optimizer_configs[optimizer_type] = optimizer_config
                logger.debug(f"Loaded configuration for optimizer: {optimizer_type}")
            except Exception as e:
                logger.error(f"Error loading config for optimizer {optimizer_type}: {e}")
    
    def get_config(self, 
                  optimizer_type: str, 
                  session_id: Optional[str] = None,
                  query_id: Optional[str] = None) -> OptimizerConfig:
        """
        Get configuration for a specific optimizer, session, and query
        
        Args:
            optimizer_type: The type of optimizer
            session_id: Optional session identifier
            query_id: Optional query identifier
            
        Returns:
            Combined configuration with appropriate overrides
        """
        # Start with global config
        config = OptimizerConfig(**self.global_config.to_dict())
        
        # Apply optimizer-specific config if exists
        if optimizer_type in self.optimizer_configs:
            config.update(self.optimizer_configs[optimizer_type].to_dict())
        
        # Apply session-specific config if exists
        if session_id and session_id in self.session_configs:
            config.update(self.session_configs[session_id].to_dict())
        
        # Apply query-specific config if exists
        if session_id and query_id and session_id in self.query_configs and \
           query_id in self.query_configs[session_id]:
            config.update(self.query_configs[session_id][query_id].to_dict())
        
        return config
    
    def set_config(self, 
                  config_dict: Dict[str, Any],
                  scope: ConfigScope,
                  optimizer_type: Optional[str] = None,
                  session_id: Optional[str] = None,
                  query_id: Optional[str] = None) -> OptimizerConfig:
        """
        Set configuration at a specific scope
        
        Args:
            config_dict: Configuration dictionary to apply
            scope: Configuration scope level
            optimizer_type: Optional optimizer type for OPTIMIZER scope
            session_id: Optional session ID for SESSION or QUERY scope
            query_id: Optional query ID for QUERY scope
            
        Returns:
            Updated configuration
        """
        if scope == ConfigScope.GLOBAL:
            self.global_config.update(config_dict)
            logger.info("Updated global configuration")
            return self.global_config
        
        elif scope == ConfigScope.OPTIMIZER:
            if not optimizer_type:
                raise ValueError("Optimizer type required for OPTIMIZER scope")
            
            if optimizer_type not in self.optimizer_configs:
                # Create new config based on global defaults
                self.optimizer_configs[optimizer_type] = OptimizerConfig(**self.global_config.to_dict())
            
            self.optimizer_configs[optimizer_type].update(config_dict)
            logger.info(f"Updated configuration for optimizer: {optimizer_type}")
            return self.optimizer_configs[optimizer_type]
        
        elif scope == ConfigScope.SESSION:
            if not session_id:
                raise ValueError("Session ID required for SESSION scope")
            
            if session_id not in self.session_configs:
                # Create new config based on global defaults
                self.session_configs[session_id] = OptimizerConfig(**self.global_config.to_dict())
            
            self.session_configs[session_id].update(config_dict)
            logger.info(f"Updated configuration for session: {session_id}")
            return self.session_configs[session_id]
        
        elif scope == ConfigScope.QUERY:
            if not session_id or not query_id:
                raise ValueError("Session ID and Query ID required for QUERY scope")
            
            if session_id not in self.query_configs:
                self.query_configs[session_id] = {}
            
            if query_id not in self.query_configs[session_id]:
                # Create new config based on session defaults if they exist, otherwise global
                base_config = self.session_configs.get(session_id, self.global_config)
                self.query_configs[session_id][query_id] = OptimizerConfig(**base_config.to_dict())
            
            self.query_configs[session_id][query_id].update(config_dict)
            logger.info(f"Updated configuration for query: {query_id} in session: {session_id}")
            return self.query_configs[session_id][query_id]
        
        raise ValueError(f"Unsupported configuration scope: {scope}")
    
    def reset_config(self, 
                    scope: ConfigScope,
                    optimizer_type: Optional[str] = None,
                    session_id: Optional[str] = None,
                    query_id: Optional[str] = None):
        """
        Reset configuration to defaults at a specific scope
        
        Args:
            scope: Configuration scope level
            optimizer_type: Optional optimizer type for OPTIMIZER scope
            session_id: Optional session ID for SESSION or QUERY scope
            query_id: Optional query ID for QUERY scope
        """
        if scope == ConfigScope.GLOBAL:
            self.global_config = OptimizerConfig.from_env()
            logger.info("Reset global configuration to defaults")
        
        elif scope == ConfigScope.OPTIMIZER:
            if not optimizer_type:
                raise ValueError("Optimizer type required for OPTIMIZER scope")
            
            if optimizer_type in self.optimizer_configs:
                del self.optimizer_configs[optimizer_type]
                logger.info(f"Reset configuration for optimizer: {optimizer_type}")
        
        elif scope == ConfigScope.SESSION:
            if not session_id:
                raise ValueError("Session ID required for SESSION scope")
            
            if session_id in self.session_configs:
                del self.session_configs[session_id]
                # Also remove any query configs for this session
                if session_id in self.query_configs:
                    del self.query_configs[session_id]
                logger.info(f"Reset configuration for session: {session_id}")
        
        elif scope == ConfigScope.QUERY:
            if not session_id or not query_id:
                raise ValueError("Session ID and Query ID required for QUERY scope")
            
            if session_id in self.query_configs and query_id in self.query_configs[session_id]:
                del self.query_configs[session_id][query_id]
                logger.info(f"Reset configuration for query: {query_id} in session: {session_id}")
        
        else:
            raise ValueError(f"Unsupported configuration scope: {scope}")
    
    def get_optimizer_specific_param(self, 
                                   optimizer_type: str,
                                   param_name: str, 
                                   default_value: Any = None,
                                   session_id: Optional[str] = None,
                                   query_id: Optional[str] = None) -> Any:
        """
        Get an optimizer-specific parameter with fallback to default
        
        Args:
            optimizer_type: Type of optimizer
            param_name: Parameter name to retrieve
            default_value: Default value if parameter not found
            session_id: Optional session identifier
            query_id: Optional query identifier
            
        Returns:
            Parameter value or default
        """
        # Get the appropriate config
        config = self.get_config(optimizer_type, session_id, query_id)
        
        # Check if parameter exists in custom_params
        if param_name in config.custom_params:
            return config.custom_params[param_name]
        
        # Check if parameter exists directly in config
        if hasattr(config, param_name):
            return getattr(config, param_name)
        
        return default_value


# Singleton instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager