"""
Configuration Settings for Agentic BTE

This module provides centralized configuration management for all components
of the Agentic BTE system, including API keys, model settings, and feature flags.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class AgenticBTESettings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Keys
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for LLM operations"
    )
    
    # Model Settings
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use for LLM operations"
    )
    
    openai_temperature: float = Field(
        default=0.1,
        description="Temperature setting for OpenAI models",
        ge=0.0,
        le=2.0
    )
    
    # SpaCy Model Settings
    scispacy_large_model: str = Field(
        default="en_core_sci_lg",
        description="Large scientific spaCy model"
    )
    
    scispacy_drug_disease_model: str = Field(
        default="en_ner_bc5cdr_md",
        description="Drug/disease NER spaCy model"
    )
    
    # BTE API Settings
    bte_api_base_url: str = Field(
        default="https://bte.transltr.io/v1",
        description="Base URL for BTE API"
    )
    
    bte_meta_kg_endpoint: str = Field(
        default="meta_knowledge_graph",
        description="Meta knowledge graph endpoint"
    )
    
    # BTE async behavior
    bte_prefer_async: bool = Field(
        default=False,
        description="Prefer using asyncquery endpoints; if false, use /query directly"
    )
    bte_async_poll_seconds: int = Field(
        default=300,
        description="Max seconds to poll async result before fallback",
        ge=1,
        le=1800
    )
    bte_async_poll_interval: float = Field(
        default=2.0,
        description="Polling interval seconds for async result",
        ge=0.1,
        le=30.0
    )
    
    # SRI Name Resolver Settings
    sri_name_resolver_url: str = Field(
        default="https://name-lookup.ci.transltr.io/lookup",
        description="SRI Name Resolver API URL"
    )
    
    # Query Processing Settings
    max_subqueries: int = Field(
        default=10,
        description="Maximum number of subqueries for decomposition",
        ge=1,
        le=20
    )
    
    max_results_per_query: int = Field(
        default=50,
        description="Maximum results to retrieve per query",
        ge=1,
        le=1000
    )
    
    # TRAPI batching
    trapi_batch_limit: int = Field(
        default=10,
        description="Maximum entity IDs per TRAPI query (batch size)",
        ge=1,
        le=500
    )
    
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence threshold for results",
        ge=0.0,
        le=1.0
    )
    
    # Excluded predicates for BTE queries
    excluded_predicates: List[str] = Field(
        default=[
            "biolink:related_to",
            "biolink:associated_with", 
            "biolink:correlated_with",
            "biolink:coexists_with"
        ],
        description="Predicates to exclude from TRAPI queries for better specificity"
    )
    
    # Caching Settings
    enable_caching: bool = Field(
        default=True,
        description="Enable result caching"
    )
    
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache time-to-live in seconds",
        ge=60
    )
    
    # Retry Settings
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for API calls",
        ge=1,
        le=10
    )
    
    retry_delay_seconds: int = Field(
        default=5,
        description="Delay between retries in seconds",
        ge=1,
        le=60
    )
    
    # Logging Settings
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Feature Flags
    enable_semantic_classification: bool = Field(
        default=True,
        description="Enable LLM-based semantic query classification"
    )

    # Query shaping policy (flexible by default)
    enforce_two_node: bool = Field(
        default=False,
        description="If true, force TRAPI to a single-edge, two-node graph; if false, allow LLM-shaped graphs"
    )
    bp_prefilter_mode: str = Field(
        default="off",
        description="Biological process gene prefilter mode: off | suggest | enforce"
    )
    
    enable_entity_name_resolution: bool = Field(
        default=True,
        description="Enable entity ID to name resolution"
    )
    
    enable_query_optimization: bool = Field(
        default=True,
        description="Enable query optimization strategies"
    )
    
    # Graph of Thoughts (GoT) Settings
    got_max_iterations: int = Field(
        default=5,
        description="Maximum iterations for GoT framework",
        ge=1,
        le=15
    )
    
    got_quality_threshold: float = Field(
        default=0.1,
        description="Quality threshold for GoT iterative refinement",
        ge=0.01,
        le=1.0
    )
    
    got_enable_refinement: bool = Field(
        default=True,
        description="Enable GoT iterative result refinement"
    )
    
    got_enable_parallel_execution: bool = Field(
        default=True,
        description="Enable parallel execution in GoT framework"
    )
    
    got_max_concurrent: int = Field(
        default=3,
        description="Maximum concurrent operations for GoT",
        ge=1,
        le=6
    )
    
    got_default_output_format: str = Field(
        default="comprehensive",
        description="Default output format for GoT MCP tool"
    )
    
    got_enable_graph_visualization: bool = Field(
        default=False,
        description="Enable graph visualization in GoT (disabled for MCP by default)"
    )
    
    got_save_results_default: bool = Field(
        default=False,
        description="Default setting for saving GoT results to files"
    )
    
    # Development Settings
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode for development"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Environment variable prefixes
        env_prefix = "AGENTIC_BTE_"


# Global settings instance
settings = AgenticBTESettings()


def get_settings() -> AgenticBTESettings:
    """Get the application settings instance"""
    return settings


def reload_settings() -> AgenticBTESettings:
    """Reload settings from environment variables"""
    global settings
    settings = AgenticBTESettings()
    return settings