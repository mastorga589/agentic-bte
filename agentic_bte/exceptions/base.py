"""
Base Exception Classes for Agentic BTE

This module provides the base exception hierarchy for all Agentic BTE components.
"""

from typing import Optional, Any, Dict


class AgenticBTEError(Exception):
    """Base exception class for all Agentic BTE errors"""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize base exception
        
        Args:
            message: Error message
            details: Optional additional error details
            original_error: Optional original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error
    
    def __str__(self) -> str:
        """String representation of the error"""
        result = self.message
        if self.details:
            result += f" (Details: {self.details})"
        if self.original_error:
            result += f" (Caused by: {self.original_error})"
        return result


class ConfigurationError(AgenticBTEError):
    """Raised when there are configuration-related errors"""
    pass


class ValidationError(AgenticBTEError):
    """Raised when input validation fails"""
    pass


class ExternalServiceError(AgenticBTEError):
    """Raised when external service calls fail"""
    
    def __init__(
        self, 
        message: str,
        service_name: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize external service error
        
        Args:
            message: Error message
            service_name: Name of the external service
            status_code: HTTP status code if applicable
            details: Optional additional error details
            original_error: Optional original exception
        """
        super().__init__(message, details, original_error)
        self.service_name = service_name
        self.status_code = status_code


class ProcessingTimeoutError(AgenticBTEError):
    """Raised when processing operations timeout"""
    
    def __init__(
        self, 
        message: str,
        timeout_seconds: float,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize timeout error
        
        Args:
            message: Error message
            timeout_seconds: The timeout value that was exceeded
            details: Optional additional error details
            original_error: Optional original exception
        """
        super().__init__(message, details, original_error)
        self.timeout_seconds = timeout_seconds


class KnowledgeRetrievalError(AgenticBTEError):
    """Raised when knowledge retrieval operations fail"""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        service: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize knowledge retrieval error
        
        Args:
            message: Error message
            query: The query that failed (if applicable)
            service: The service that failed (if applicable)
            details: Optional additional error details
            original_error: Optional original exception
        """
        super().__init__(message, details, original_error)
        self.query = query
        self.service = service
