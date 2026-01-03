
import logging
import sys
import json
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler

# Custom formatter for JSON logs (Cloud/Production friendly)
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "path": record.pathname,
            "line": record.lineno
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

# Custom formatter for Colored Console logs (Dev friendly)
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[41m'  # Red background
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

def setup_enhanced_logging(level: str = "INFO", log_file: Optional[Path] = None, use_colors: bool = True):
    """
    Configure enterprise-grade logging with rotation and dual formatters.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    #Clear existing handlers
    root_logger.handlers.clear()

    # 1. Console Handler (Human Friendly)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ) if use_colors else logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 2. File Handler (Machine Friendly / Persistent)
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024, # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    logging.info(f"Enhanced logging initialized at level {level}")
