# Comprehensive Error Handling Module
# Provides centralized error handling, logging, and recovery mechanisms

import logging
import traceback
from typing import Dict, Any, Optional, Union
from datetime import datetime
from functools import wraps
import json
from enum import Enum

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better organization"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_PROCESSING = "data_processing"
    MODEL_EXECUTION = "model_execution"
    FILE_OPERATION = "file_operation"
    NETWORK = "network"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    SYSTEM = "system"

class NexusError(Exception):
    """Base exception class for Nexus application"""
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.user_message = user_message or self._get_user_friendly_message()
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(self.message)
    
    def _get_user_friendly_message(self) -> str:
        """Generate user-friendly error message"""
        messages = {
            ErrorCategory.VALIDATION: "Invalid input provided. Please check your data and try again.",
            ErrorCategory.AUTHENTICATION: "Authentication failed. Please check your credentials.",
            ErrorCategory.AUTHORIZATION: "You don't have permission to perform this action.",
            ErrorCategory.DATA_PROCESSING: "An error occurred while processing your data. Please try again.",
            ErrorCategory.MODEL_EXECUTION: "The AI model encountered an error. Please try again or use a different query.",
            ErrorCategory.FILE_OPERATION: "File operation failed. Please check the file and try again.",
            ErrorCategory.NETWORK: "Network connection error. Please check your connection.",
            ErrorCategory.DATABASE: "Database operation failed. Please try again later.",
            ErrorCategory.CONFIGURATION: "Configuration error. Please check your settings.",
            ErrorCategory.SYSTEM: "A system error occurred. Please try again later."
        }
        return messages.get(self.category, "An unexpected error occurred.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            "error": self.user_message,
            "error_code": f"{self.category.value}_{self.severity.value}",
            "timestamp": self.timestamp,
            "details": self.details if logging.getLogger().level <= logging.DEBUG else {}
        }

# Specific error classes
class ValidationError(NexusError):
    """Input validation errors"""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        details = {"field": field} if field else {}
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            details=details,
            **kwargs
        )

class DataProcessingError(NexusError):
    """Data processing errors"""
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        details = {"operation": operation} if operation else {}
        super().__init__(
            message,
            category=ErrorCategory.DATA_PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )

class ModelExecutionError(NexusError):
    """Model execution errors"""
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        details = {"model": model_name} if model_name else {}
        super().__init__(
            message,
            category=ErrorCategory.MODEL_EXECUTION,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )

class FileOperationError(NexusError):
    """File operation errors"""
    def __init__(self, message: str, filename: Optional[str] = None, **kwargs):
        details = {"filename": filename} if filename else {}
        super().__init__(
            message,
            category=ErrorCategory.FILE_OPERATION,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )

class ErrorHandler:
    """Central error handler with logging and recovery mechanisms"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.last_errors: list = []
        self.max_error_history = 100
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        raise_error: bool = True
    ) -> Dict[str, Any]:
        """
        Handle an error with proper logging and formatting
        
        Args:
            error: The exception to handle
            context: Additional context about the error
            raise_error: Whether to re-raise the error after handling
            
        Returns:
            Dictionary with error information for API responses
        """
        # Create error response
        if isinstance(error, NexusError):
            error_response = error.to_dict()
            severity = error.severity
        else:
            # Convert standard exceptions to NexusError
            nexus_error = NexusError(
                message=str(error),
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                details={"original_type": type(error).__name__}
            )
            error_response = nexus_error.to_dict()
            severity = ErrorSeverity.MEDIUM
        
        # Add context if provided
        if context:
            error_response["context"] = context
        
        # Log the error
        self._log_error(error, error_response, severity)
        
        # Track error statistics
        self._track_error(error)
        
        # Store in error history
        self._store_error(error_response)
        
        if raise_error:
            raise error
        
        return error_response
    
    def _log_error(self, error: Exception, error_response: Dict[str, Any], severity: ErrorSeverity):
        """Log error with appropriate level"""
        log_message = {
            "error_type": type(error).__name__,
            "message": str(error),
            "response": error_response,
            "traceback": traceback.format_exc() if self.logger.level <= logging.DEBUG else None
        }
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(json.dumps(log_message))
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(json.dumps(log_message))
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(json.dumps(log_message))
        else:
            self.logger.info(json.dumps(log_message))
    
    def _track_error(self, error: Exception):
        """Track error statistics"""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def _store_error(self, error_response: Dict[str, Any]):
        """Store error in history"""
        self.last_errors.append(error_response)
        if len(self.last_errors) > self.max_error_history:
            self.last_errors.pop(0)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "error_counts": self.error_counts,
            "total_errors": sum(self.error_counts.values()),
            "recent_errors": self.last_errors[-10:]
        }
    
    def clear_statistics(self):
        """Clear error statistics"""
        self.error_counts.clear()
        self.last_errors.clear()

# Decorator for automatic error handling
def handle_errors(
    default_return: Any = None,
    log_errors: bool = True,
    raise_errors: bool = False
):
    """
    Decorator for automatic error handling in functions
    
    Args:
        default_return: Default value to return on error
        log_errors: Whether to log errors
        raise_errors: Whether to re-raise errors after handling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                
                if raise_errors:
                    raise
                
                return default_return
        return wrapper
    return decorator

# Global error handler instance
error_handler = ErrorHandler()

# Utility functions for common error scenarios
def validate_required_fields(data: Dict[str, Any], required_fields: list) -> None:
    """
    Validate that required fields are present in data
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        
    Raises:
        ValidationError: If any required field is missing
    """
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            field=missing_fields[0],
            details={"missing_fields": missing_fields}
        )

def validate_file_extension(filename: str, allowed_extensions: list) -> None:
    """
    Validate file extension
    
    Args:
        filename: Name of the file to validate
        allowed_extensions: List of allowed extensions
        
    Raises:
        ValidationError: If file extension is not allowed
    """
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in allowed_extensions:
        raise ValidationError(
            f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}",
            field="filename",
            details={"extension": ext, "allowed": allowed_extensions}
        )

def safe_json_parse(json_str: str, default: Any = None) -> Union[Dict, list, Any]:
    """
    Safely parse JSON string
    
    Args:
        json_str: JSON string to parse
        default: Default value to return on error
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def create_error_response(
    message: str,
    status_code: int = 400,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        message: Error message
        status_code: HTTP status code
        details: Additional error details
        
    Returns:
        Standardized error response dictionary
    """
    return {
        "error": message,
        "status": "error",
        "status_code": status_code,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details or {}
    }
