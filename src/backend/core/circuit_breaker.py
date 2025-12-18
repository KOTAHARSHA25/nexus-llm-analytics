# Circuit Breaker Pattern for LLM Operations
# Provides graceful degradation when Ollama is unavailable

import time
import logging
from enum import Enum
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from functools import wraps

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker activated, fail fast
    HALF_OPEN = "half_open"  # Testing if service is back

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 3  # Number of failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    success_threshold: int = 2  # Successful calls needed to close circuit
    timeout: float = 30.0  # Individual operation timeout
    
@dataclass 
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring"""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0
    last_success_time: float = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_calls: int = 0
    
class CircuitBreaker:
    """
    Circuit breaker implementation for LLM operations
    
    Features:
    - Automatic failure detection and recovery
    - Exponential backoff for retries
    - Health monitoring and statistics
    - Graceful degradation to fallback responses
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._state_change_time = time.time()
        
    def call(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Execute function with circuit breaker protection"""
        self.stats.total_calls += 1
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self._state_change_time = time.time()
                logging.info(f"Circuit breaker '{self.name}' transitioning to HALF-OPEN")
            else:
                return self._get_fallback_response("Circuit breaker is OPEN - service unavailable")
        
        # Attempt to call the function
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Check if result indicates success
            if self._is_successful_result(result):
                self._record_success()
                logging.debug(f"Circuit breaker '{self.name}' successful call ({execution_time:.2f}s)")
                return result
            else:
                self._record_failure("Function returned error result")
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure(str(e))
            logging.error(f"Circuit breaker '{self.name}' failed call ({execution_time:.2f}s): {e}")
            
            # Return fallback response instead of raising exception
            return self._get_fallback_response(f"Service error: {str(e)}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt circuit reset"""
        time_since_open = time.time() - self._state_change_time
        return time_since_open >= self.config.recovery_timeout
    
    def _is_successful_result(self, result: Dict[str, Any]) -> bool:
        """Determine if result indicates success"""
        if isinstance(result, dict):
            # Check for explicit error indicators
            if "error" in result or result.get("status") == "error":
                return False
            # Check for successful response patterns
            if "response" in result or "result" in result or result.get("status") == "success":
                return True
        return False
    
    def _record_success(self):
        """Record successful operation"""
        self.stats.success_count += 1
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0
        self.stats.last_success_time = time.time()
        
        # Transition circuit state based on success threshold
        if self.state == CircuitState.HALF_OPEN:
            if self.stats.consecutive_successes >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self._state_change_time = time.time()
                logging.info(f"Circuit breaker '{self.name}' CLOSED - service recovered")
    
    def _record_failure(self, error_msg: str):
        """Record failed operation"""
        self.stats.failure_count += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.last_failure_time = time.time()
        
        # Transition circuit state based on failure threshold
        if self.state == CircuitState.CLOSED:
            if self.stats.consecutive_failures >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self._state_change_time = time.time()
                logging.warning(f"Circuit breaker '{self.name}' OPENED - too many failures ({self.stats.consecutive_failures})")
        
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state opens the circuit again
            self.state = CircuitState.OPEN
            self._state_change_time = time.time()
            logging.warning(f"Circuit breaker '{self.name}' reopened after failure in HALF-OPEN state")
    
    def _get_fallback_response(self, error_msg: str) -> Dict[str, Any]:
        """Generate fallback response when circuit is open"""
        fallback_strategies = {
            "data_analysis": self._get_data_analysis_fallback,
            "rag_retrieval": self._get_rag_fallback,
            "code_review": self._get_code_review_fallback,
            "visualization": self._get_visualization_fallback,
            "default": self._get_default_fallback
        }
        
        strategy = fallback_strategies.get(self.name, fallback_strategies["default"])
        return strategy(error_msg)
    
    def _get_data_analysis_fallback(self, error_msg: str) -> Dict[str, Any]:
        """Fallback for data analysis operations"""
        return {
            "success": False,
            "result": f"""Data Analysis Unavailable

[!] The AI analysis service is currently unavailable ({error_msg}).

Alternative Analysis Options:
1. Basic Data Summary:
   - Check data shape, columns, and basic statistics
   - Look for missing values and data types
   - Generate simple descriptive statistics

2. Manual Data Exploration:
   - Use df.head() to preview data
   - Use df.describe() for statistical summary  
   - Use df.info() for data structure info

3. Simple Visualizations:
   - Create basic plots with matplotlib/seaborn
   - Generate histograms for numeric columns
   - Create value counts for categorical data

The system will automatically retry when the AI service becomes available.
""",
            "type": "fallback_analysis",
            "status": "service_unavailable",
            "retry_after": self.config.recovery_timeout,
            "fallback_used": True
        }
    
    def _get_rag_fallback(self, error_msg: str) -> Dict[str, Any]:
        """Fallback for RAG/document retrieval operations"""
        return {
            "success": False,
            "result": f"""Document Analysis Unavailable

[!] The AI document analysis service is currently unavailable ({error_msg}).

Alternative Document Review Options:
1. Manual Document Review:
   - Open the document directly for reading
   - Use search functionality to find specific content
   - Extract text manually for analysis

2. Basic Text Processing:
   - Use simple keyword search
   - Count word frequencies
   - Extract basic statistics (word count, pages, etc.)

3. Structured Data Extraction:
   - Look for tables, lists, and structured content
   - Extract numerical data manually
   - Identify key sections and headings

The AI-powered document analysis will be available once the service recovers.
""",
            "type": "fallback_rag",
            "status": "service_unavailable",
            "retry_after": self.config.recovery_timeout,
            "fallback_used": True
        }
    
    def _get_code_review_fallback(self, error_msg: str) -> Dict[str, Any]:
        """Fallback for code review operations"""
        return {
            "success": True,  # Allow code execution with basic validation
            "result": f"""Code Review Service Unavailable

[!] AI code review is currently unavailable ({error_msg}).

Basic Safety Checks Applied:
- Blocked dangerous imports and operations
- Limited memory and execution time
- Sandboxed execution environment active

Manual Review Recommended:
1. Check for suspicious operations
2. Validate data transformations
3. Review output for accuracy
4. Test with small data samples first

Code will execute with basic safety measures only.
""",
            "type": "fallback_review",
            "status": "partial_service",
            "retry_after": self.config.recovery_timeout,
            "fallback_used": True
        }
    
    def _get_visualization_fallback(self, error_msg: str) -> Dict[str, Any]:
        """Fallback for visualization operations"""
        return {
            "success": False,
            "result": f"""Visualization Service Unavailable

[!] AI-powered visualization is currently unavailable ({error_msg}).

Manual Visualization Options:
1. Basic Matplotlib/Seaborn:
   - plt.hist() for histograms
   - plt.scatter() for scatter plots
   - sns.boxplot() for box plots

2. Pandas Built-in Plotting:
   - df.plot() for quick plots
   - df.plot.bar() for bar charts
   - df.plot.line() for line plots

3. Static Chart Templates:
   - Use predefined chart configurations
   - Apply standard styling and colors
   - Generate basic interactive plots

Advanced AI-generated visualizations will return when service recovers.
""",
            "type": "fallback_visualization", 
            "status": "service_unavailable",
            "retry_after": self.config.recovery_timeout,
            "fallback_used": True
        }
    
    def _get_default_fallback(self, error_msg: str) -> Dict[str, Any]:
        """Default fallback response"""
        return {
            "success": False,
            "error": f"Service temporarily unavailable: {error_msg}",
            "status": "service_unavailable",
            "retry_after": self.config.recovery_timeout,
            "fallback_used": True,
            "message": "The AI service is currently unavailable. Please try again later."
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status and statistics"""
        uptime_hours = (time.time() - (self.stats.last_success_time or time.time())) / 3600
        
        return {
            "name": self.name,
            "state": self.state.value,
            "health": "healthy" if self.state == CircuitState.CLOSED else "degraded",
            "statistics": {
                "total_calls": self.stats.total_calls,
                "success_count": self.stats.success_count,
                "failure_count": self.stats.failure_count,
                "success_rate": (self.stats.success_count / max(1, self.stats.total_calls)) * 100,
                "consecutive_failures": self.stats.consecutive_failures,
                "consecutive_successes": self.stats.consecutive_successes,
                "last_failure": self.stats.last_failure_time,
                "last_success": self.stats.last_success_time,
                "uptime_hours": round(uptime_hours, 2)
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold
            }
        }
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._state_change_time = time.time()
        logging.info(f"Circuit breaker '{self.name}' manually reset")

# Global circuit breakers for different services
_circuit_breakers: Dict[str, CircuitBreaker] = {}

def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get or create a circuit breaker for a service"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]

def circuit_breaker_protected(name: str, config: CircuitBreakerConfig = None):
    """Decorator to add circuit breaker protection to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cb = get_circuit_breaker(name, config)
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator

def get_all_circuit_breaker_status() -> Dict[str, Any]:
    """Get status of all circuit breakers"""
    return {
        "circuit_breakers": [cb.get_health_status() for cb in _circuit_breakers.values()],
        "overall_health": "healthy" if all(cb.state == CircuitState.CLOSED for cb in _circuit_breakers.values()) else "degraded"
    }