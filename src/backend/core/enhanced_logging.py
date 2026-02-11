"""
Enhanced Logging — Nexus LLM Analytics v2.0
============================================

Enterprise-grade logging configuration with JSON and coloured console
formatters, rotating file handlers, and structured output.

Enterprise v2.0 Additions
-------------------------
* **CorrelationIdFilter** — Logging filter that injects a correlation
  (request) ID into every log record for distributed tracing.
* **SensitiveDataFilter** — Redacts PII patterns (emails, API keys)
  from log messages before they are emitted.
* ``configure_logging()`` — High-level convenience wrapper around
  ``setup_enhanced_logging()`` accepting a :class:`~backend.core.config.Settings`
  object.

Backward Compatibility
----------------------
``setup_enhanced_logging()``, ``JsonFormatter``, and
``ColoredFormatter`` are unchanged.

.. versionchanged:: 2.0
   Added correlation-ID and PII-redaction filters.

Author: Nexus Analytics Research Team
Date: February 2026
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Custom formatter for JSON logs (Cloud/Production friendly)
class JsonFormatter(logging.Formatter):
    """Logging formatter that serialises log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            JSON-encoded representation of the log record.
        """
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
    """Logging formatter that adds ANSI colour codes for terminal output."""

    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[41m'  # Red background
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """Wrap the log message in the appropriate ANSI colour escape.

        Args:
            record: The log record to format.

        Returns:
            Colour-wrapped formatted log string.
        """
        color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

def setup_enhanced_logging(level: str = "INFO", log_file: Optional[Path] = None, use_colors: bool = True) -> None:
    """Configure enterprise-grade logging with rotation and dual formatters.

    Args:
        level: Logging verbosity (e.g. ``"INFO"``, ``"DEBUG"``).
        log_file: Optional path for a rotating file handler.
        use_colors: Whether to use ANSI colours on the console handler.
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

    logger.info("Enhanced logging initialized at level %s", level)


# ============================================================================
# Enterprise v2.0 — Correlation-ID & PII Redaction Filters
# ============================================================================

# Thread-local storage for correlation IDs
_correlation_local = threading.local()


def set_correlation_id(cid: str) -> None:
    """Store a correlation / request ID for the current thread.

    Args:
        cid: The correlation identifier (usually a UUID from the
             incoming HTTP request).

    .. versionadded:: 2.0
    """
    _correlation_local.correlation_id = cid


def get_correlation_id() -> str:
    """Return the correlation ID for the current thread, or ``'-'``.

    .. versionadded:: 2.0
    """
    return getattr(_correlation_local, "correlation_id", "-")


class CorrelationIdFilter(logging.Filter):
    """Inject ``correlation_id`` into every log record.

    Attach this filter to a handler or logger to make the correlation
    ID available in format strings via ``%(correlation_id)s``.

    Example::

        handler.addFilter(CorrelationIdFilter())
        fmt = logging.Formatter('%(asctime)s [%(correlation_id)s] %(message)s')
        handler.setFormatter(fmt)

    .. versionadded:: 2.0
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        """Add ``correlation_id`` attribute to *record*.

        Args:
            record: The log record being processed.

        Returns:
            Always ``True`` (never drops records).
        """
        record.correlation_id = get_correlation_id()  # type: ignore[attr-defined]
        return True


class SensitiveDataFilter(logging.Filter):
    """Redact common PII patterns from log messages.

    Patterns redacted:
    * Email addresses → ``[REDACTED_EMAIL]``
    * Bearer / API tokens → ``[REDACTED_TOKEN]``
    * Sequences of 16+ hex digits → ``[REDACTED_KEY]``

    .. versionadded:: 2.0
    """

    _EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    _TOKEN_RE = re.compile(r"(?i)(bearer|token|api[_-]?key)\s*[:=]\s*\S+")
    _HEX_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        """Scrub PII from *record.msg*.

        Args:
            record: The log record to sanitise.

        Returns:
            Always ``True``.
        """
        if isinstance(record.msg, str):
            record.msg = self._EMAIL_RE.sub("[REDACTED_EMAIL]", record.msg)
            record.msg = self._TOKEN_RE.sub("[REDACTED_TOKEN]", record.msg)
            record.msg = self._HEX_RE.sub("[REDACTED_KEY]", record.msg)
        return True


def configure_logging(settings: Optional[object] = None) -> None:
    """High-level convenience wrapper.

    Calls :func:`setup_enhanced_logging` with parameters extracted from
    a :class:`~backend.core.config.Settings` object (or sensible
    defaults).

    Args:
        settings: Optional settings object with ``log_level``,
                  ``log_file``, and ``debug`` attributes.

    .. versionadded:: 2.0
    """
    level = "INFO"
    log_file: Optional[Path] = None
    use_colors = True

    if settings is not None:
        level = getattr(settings, "log_level", "INFO")
        lf = getattr(settings, "log_file", None)
        log_file = Path(lf) if lf else None
        use_colors = getattr(settings, "debug", True)

    setup_enhanced_logging(level=level, log_file=log_file, use_colors=use_colors)
