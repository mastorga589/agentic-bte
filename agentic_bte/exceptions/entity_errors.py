"""
Entity-Related Exception Classes

This module provides specialized exception classes for entity processing operations.
"""

from typing import Optional, Any, Dict, List
from .base import AgenticBTEError, ValidationError


class EntityError(AgenticBTEError):
    """Base class for entity-related errors"""
    pass


class EntityRecognitionError(EntityError):
    """Raised when entity recognition fails"""
    
    def __init__(
        self, 
        message: str,
        input_text: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize entity recognition error
        
        Args:
            message: Error message
            input_text: The text that failed to be processed
            details: Optional additional error details
            original_error: Optional original exception
        """
        super().__init__(message, details, original_error)
        self.input_text = input_text


class EntityLinkingError(EntityError):
    """Raised when entity linking to knowledge bases fails"""
    
    def __init__(
        self, 
        message: str,
        entity_text: Optional[str] = None,
        knowledge_base: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize entity linking error
        
        Args:
            message: Error message
            entity_text: The entity text that failed to link
            knowledge_base: The knowledge base that was being queried
            details: Optional additional error details
            original_error: Optional original exception
        """
        super().__init__(message, details, original_error)
        self.entity_text = entity_text
        self.knowledge_base = knowledge_base


class EntityClassificationError(EntityError):
    """Raised when entity type classification fails"""
    
    def __init__(
        self, 
        message: str,
        entity_text: Optional[str] = None,
        available_types: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize entity classification error
        
        Args:
            message: Error message
            entity_text: The entity text that failed classification
            available_types: List of available entity types
            details: Optional additional error details
            original_error: Optional original exception
        """
        super().__init__(message, details, original_error)
        self.entity_text = entity_text
        self.available_types = available_types or []


class EntityResolutionError(EntityError):
    """Raised when entity name resolution fails"""
    
    def __init__(
        self, 
        message: str,
        entity_id: Optional[str] = None,
        resolver_service: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize entity resolution error
        
        Args:
            message: Error message
            entity_id: The entity ID that failed resolution
            resolver_service: The resolver service that was used
            details: Optional additional error details
            original_error: Optional original exception
        """
        super().__init__(message, details, original_error)
        self.entity_id = entity_id
        self.resolver_service = resolver_service


class InvalidEntityError(ValidationError):
    """Raised when entity data is invalid or malformed"""
    
    def __init__(
        self, 
        message: str,
        entity_data: Optional[Dict[str, Any]] = None,
        validation_errors: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize invalid entity error
        
        Args:
            message: Error message
            entity_data: The invalid entity data
            validation_errors: List of specific validation errors
            details: Optional additional error details
        """
        super().__init__(message, details)
        self.entity_data = entity_data
        self.validation_errors = validation_errors or []


class EntityCacheError(EntityError):
    """Raised when entity caching operations fail"""
    pass