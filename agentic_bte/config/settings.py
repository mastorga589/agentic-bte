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
    
    enable_entity_name_resolution: bool = Field(
        default=True,
        description="Enable entity ID to name resolution"
    )
    
    enable_query_optimization: bool = Field(
        default=True,
        description="Enable query optimization strategies"
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