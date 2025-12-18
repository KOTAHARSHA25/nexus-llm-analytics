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
        
        # Format the message
        message = record.getMessage()
        
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
