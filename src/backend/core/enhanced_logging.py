"""
Enhanced Logging Module
=======================
Beautiful, informative logging with emojis, colors, and structured output.

Features:
- Emoji-prefixed log levels
- Colored console output
- Clean, readable format
- Request tracing
- Performance metrics
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional
import json


# ANSI color codes for terminal
class Colors:
    """ANSI color codes for terminal output"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    
    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


# Emoji prefixes for log levels
LEVEL_EMOJI = {
    'DEBUG': '🔍',
    'INFO': '✨',
    'WARNING': '⚠️',
    'ERROR': '❌',
    'CRITICAL': '🚨',
}

# Color mapping for log levels
LEVEL_COLORS = {
    'DEBUG': Colors.DIM + Colors.CYAN,
    'INFO': Colors.GREEN,
    'WARNING': Colors.YELLOW,
    'ERROR': Colors.RED,
    'CRITICAL': Colors.BOLD + Colors.BRIGHT_RED,
}


class EmojiFormatter(logging.Formatter):
    """Custom formatter with emojis and colors"""
    
    def __init__(self, use_colors: bool = True, use_emojis: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
        self.use_emojis = use_emojis
    
    def format(self, record: logging.LogRecord) -> str:
        # Get emoji and color for level
        level_name = record.levelname
        emoji = LEVEL_EMOJI.get(level_name, '📝') if self.use_emojis else ''
        color = LEVEL_COLORS.get(level_name, '') if self.use_colors else ''
        reset = Colors.RESET if self.use_colors else ''
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Format the message - TRUNCATE if too long to prevent data dumps
        message = record.getMessage()
        
        # Prevent logging of large data previews (likely accidental data dumps)
        MAX_LOG_LENGTH = 500  # Reasonable limit for log messages
        if len(message) > MAX_LOG_LENGTH:
            message = message[:MAX_LOG_LENGTH] + f"... [truncated {len(message)-MAX_LOG_LENGTH} chars]"
        
        # Clean format for console
        if self.use_colors:
            formatted = f"{Colors.DIM}{timestamp}{reset} {emoji} {color}{message}{reset}"
        else:
            formatted = f"{timestamp} {emoji} {message}"
        
        return formatted


class FileFormatter(logging.Formatter):
    """JSON formatter for file logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_enhanced_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_colors: bool = True,
    use_emojis: bool = True,
) -> None:
    """
    Configure enhanced logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        use_colors: Enable colored output
        use_emojis: Enable emoji prefixes
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler with emoji formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(EmojiFormatter(use_colors=use_colors, use_emojis=use_emojis))
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(console_handler)
    
    # File handler with JSON formatter (if specified)
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(FileFormatter())
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    noisy_loggers = [
        'httpx',
        'httpcore',
        'urllib3',
        'requests',
        'chromadb',
        'chromadb.telemetry',
        'chromadb.config',
        'openai',
        'litellm',
        'litellm.utils',
        'litellm.llms',
        'watchfiles',
        'uvicorn.access',
        'uvicorn.error',
        'PIL',
        'matplotlib',
        'asyncio',
        'parso',
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Special handling for uvicorn to keep it clean
    logging.getLogger('uvicorn').setLevel(logging.INFO)


def log_startup_banner(app_name: str = "Nexus LLM Analytics", version: str = "2.0"):
    """Print a beautiful startup banner"""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║  🚀 {app_name}                                    ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║  📊 AI-Powered Data Analytics Platform                       ║
║  🔒 Privacy-First • Local LLMs • Multi-Agent                 ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def log_system_info(memory_gb: float, available_gb: float, models: list):
    """Log system information in a nice table format"""
    info = f"""
┌─────────────────────────────────────────────────────────────┐
│  💻 System Status                                           │
├─────────────────────────────────────────────────────────────┤
│  🧠 Memory: {available_gb:.1f}GB available / {memory_gb:.1f}GB total          │
│  🤖 Models: {', '.join(models[:3]) if models else 'Loading...'}           │
└─────────────────────────────────────────────────────────────┘
"""
    print(info)


def log_request(method: str, path: str, status: int, duration_ms: float):
    """Log an HTTP request in a clean format"""
    status_emoji = "✅" if 200 <= status < 300 else "⚠️" if 300 <= status < 400 else "❌"
    logging.info(f"{status_emoji} {method} {path} → {status} ({duration_ms:.0f}ms)")


def log_analysis_start(query: str, model: str):
    """Log the start of an analysis"""
    query_preview = query[:50] + "..." if len(query) > 50 else query
    logging.info(f"🔄 Analysis started: \"{query_preview}\" using {model}")


def log_analysis_complete(query: str, duration_s: float, success: bool):
    """Log analysis completion"""
    query_preview = query[:30] + "..." if len(query) > 30 else query
    emoji = "✅" if success else "❌"
    logging.info(f"{emoji} Analysis complete: \"{query_preview}\" ({duration_s:.1f}s)")


# =============================================================================
# Phase 3.9: Structured Logging Enhancement
# =============================================================================

class StructuredLogger:
    """
    Phase 3.9: Structured logging for production observability.
    
    Provides consistent, machine-parseable log output with:
    - Request/response correlation
    - Performance timing
    - Error context
    - Agent execution traces
    - LLM interaction logging
    
    Example:
        logger = StructuredLogger("query_processor")
        with logger.span("process_query", query_id="abc123"):
            result = process(query)
            logger.log_event("query_processed", tokens=100, latency_ms=250)
    """
    
    def __init__(self, component: str, log_level: int = logging.INFO):
        self.component = component
        self.logger = logging.getLogger(f"nexus.{component}")
        self.logger.setLevel(log_level)
        self._context: dict = {}
    
    def set_context(self, **kwargs):
        """Set persistent context that will be included in all logs"""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear the persistent context"""
        self._context.clear()
    
    def _format_structured(self, event: str, level: str, **data) -> dict:
        """Format a structured log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "component": self.component,
            "event": event,
            **self._context,
            **data
        }
        return entry
    
    def log_event(self, event: str, level: str = "INFO", **data):
        """
        Log a structured event.
        
        Args:
            event: Event name (e.g., "query_received", "agent_started")
            level: Log level
            **data: Additional structured data
        """
        entry = self._format_structured(event, level, **data)
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Log as JSON for structured parsing
        self.logger.log(log_level, json.dumps(entry))
    
    def info(self, event: str, **data):
        """Log an INFO level event"""
        self.log_event(event, "INFO", **data)
    
    def debug(self, event: str, **data):
        """Log a DEBUG level event"""
        self.log_event(event, "DEBUG", **data)
    
    def warning(self, event: str, **data):
        """Log a WARNING level event"""
        self.log_event(event, "WARNING", **data)
    
    def error(self, event: str, error: Exception = None, **data):
        """Log an ERROR level event with optional exception details"""
        if error:
            data["error_type"] = type(error).__name__
            data["error_message"] = str(error)
        self.log_event(event, "ERROR", **data)
    
    def log_request(
        self, 
        request_id: str,
        method: str,
        path: str,
        status: int,
        duration_ms: float,
        **extra
    ):
        """Log an HTTP request in structured format"""
        self.info(
            "http_request",
            request_id=request_id,
            method=method,
            path=path,
            status_code=status,
            duration_ms=round(duration_ms, 2),
            **extra
        )
    
    def log_agent_execution(
        self,
        agent_name: str,
        action: str,
        duration_ms: float,
        success: bool,
        **extra
    ):
        """Log agent execution in structured format"""
        self.info(
            "agent_execution",
            agent=agent_name,
            action=action,
            duration_ms=round(duration_ms, 2),
            success=success,
            **extra
        )
    
    def log_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        **extra
    ):
        """Log LLM API call in structured format"""
        self.info(
            "llm_call",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=round(latency_ms, 2),
            success=success,
            **extra
        )
    
    def log_cache_event(
        self,
        cache_type: str,
        hit: bool,
        key_preview: str = None,
        **extra
    ):
        """Log cache hit/miss in structured format"""
        self.debug(
            "cache_access",
            cache_type=cache_type,
            cache_hit=hit,
            key_preview=key_preview[:20] if key_preview else None,
            **extra
        )
    
    def log_rag_query(
        self,
        query_preview: str,
        num_results: int,
        search_type: str,
        latency_ms: float,
        **extra
    ):
        """Log RAG query in structured format"""
        self.info(
            "rag_query",
            query_preview=query_preview[:50] if query_preview else None,
            num_results=num_results,
            search_type=search_type,
            latency_ms=round(latency_ms, 2),
            **extra
        )


class SpanContext:
    """Context manager for tracing spans"""
    
    def __init__(self, logger: StructuredLogger, operation: str, **tags):
        self.logger = logger
        self.operation = operation
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(
            "span_start",
            operation=self.operation,
            **self.tags
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000
        
        if exc_type:
            self.logger.error(
                "span_end",
                operation=self.operation,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.tags
            )
        else:
            self.logger.debug(
                "span_end",
                operation=self.operation,
                duration_ms=round(duration_ms, 2),
                success=True,
                **self.tags
            )


def get_structured_logger(component: str) -> StructuredLogger:
    """
    Factory function to get a structured logger for a component.
    
    Args:
        component: Component name (e.g., "api", "agent", "rag")
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(component)


# Pre-configured loggers for common components
api_logger = StructuredLogger("api")
agent_logger = StructuredLogger("agent")
rag_logger = StructuredLogger("rag")
cache_logger = StructuredLogger("cache")
llm_logger = StructuredLogger("llm")
