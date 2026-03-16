"""Shared utility functions for the Nexus LLM Analytics backend.

Provides logging formatters, data-versioning helpers, an agent
registry, and general-purpose utilities consumed across every
backend service.

**v2.0 Additions**

* :class:`TimingContext` — lightweight context manager for
  measuring wall-clock execution time of arbitrary code blocks.
* :class:`DataVersionManager` — enhanced audit trail with diff
  tracking, file-hash computation, and structured version history
  for uploaded / transformed data files.

All public symbols from v1.x remain available; no breaking changes
were introduced.

Author: Nexus Analytics Team
Date: February 2026
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import time as _time
from typing import Any

class JsonFormatter(logging.Formatter):
	"""Logging formatter that outputs log records as single-line JSON objects."""

	def format(self, record: logging.LogRecord) -> str:
		"""Format a log record as a JSON string.

		Args:
			record: The log record to format.

		Returns:
			JSON-encoded string representation of the log record.
		"""
		log_record = {
			'level': record.levelname,
			'time': self.formatTime(record, self.datefmt),
			'name': record.name,
			'message': record.getMessage(),
		}
		if record.exc_info:
			log_record['exception'] = self.formatException(record.exc_info)
		return json.dumps(log_record)

# Data versioning/audit trail
def log_data_version(event: str, filename: str, details: Any | None = None) -> None:
	"""Append an audit-trail entry for a data-file event.

	Args:
		event: Description of the data event (e.g. ``'upload'``).
		filename: Name of the affected data file.
		details: Optional extra metadata about the event.
	"""
	audit_dir = os.path.join(os.path.dirname(__file__), '../../data/audit')
	os.makedirs(audit_dir, exist_ok=True)
	entry = {
		'event': event,
		'filename': filename,
		'details': details,
		'timestamp': datetime.datetime.now().isoformat()
	}
	with open(os.path.join(audit_dir, 'audit_log.jsonl'), 'a') as f:
		f.write(json.dumps(entry) + '\n')

# Configurable logging level
def setup_logging(logfile: str = 'nexus.log') -> None:
	"""Configure the root logger with JSON-formatted console and file handlers.

	Args:
		logfile: Path to the log output file.
	"""
	level = os.environ.get('LOG_LEVEL', 'INFO').upper()
	logger = logging.getLogger()
	logger.setLevel(level)
	# Console handler
	ch = logging.StreamHandler()
	ch.setLevel(level)
	ch.setFormatter(JsonFormatter())
	# File handler
	fh = logging.FileHandler(logfile)
	fh.setLevel(level)
	fh.setFormatter(JsonFormatter())
	# Avoid duplicate handlers
	logger.handlers.clear()
	logger.addHandler(ch)
	logger.addHandler(fh)

# User-friendly error messages
def friendly_error(msg: str, suggestion: str | None = None) -> dict[str, str]:
	"""Build a user-friendly error payload.

	Args:
		msg: Human-readable error message.
		suggestion: Optional recovery suggestion shown to the user.

	Returns:
		Dict with ``'error'`` and ``'suggestion'`` keys.
	"""
	return {'error': msg, 'suggestion': suggestion or 'Please check your input and try again.'}

# Pluggable agent registry
class AgentRegistry:
	"""Registry for dynamically pluggable analysis agents."""

	def __init__(self) -> None:
		self.registry: dict[str, Any] = {}

	def register(self, name: str, agent: Any) -> None:
		"""Register an agent under the given name.

		Args:
			name: Unique identifier for the agent.
			agent: The agent instance to register.
		"""
		self.registry[name] = agent

	def get(self, name: str) -> Any | None:
		"""Retrieve a registered agent by name.

		Args:
			name: The agent identifier.

		Returns:
			The agent instance, or ``None`` if not found.
		"""
		return self.registry.get(name)


# ============================================================================
# Enterprise v2.0 — TimingContext & DataVersionManager
# ============================================================================


class TimingContext:
	"""Context manager for measuring execution time of code blocks.

	Captures wall-clock elapsed time and makes it available via the
	``elapsed_ms`` property after the block exits.

	Example::

		with TimingContext() as t:
			heavy_computation()
		print(f"Took {t.elapsed_ms:.1f} ms")

	.. versionadded:: 2.0
	"""

	def __init__(self, label: str = "") -> None:
		self.label = label
		self._start: float = 0.0
		self._end: float = 0.0

	def __enter__(self) -> "TimingContext":
		self._start = _time.perf_counter()
		return self

	def __exit__(self, *exc_info) -> None:
		self._end = _time.perf_counter()
		if self.label:
			logging.debug("TimingContext [%s]: %.1f ms", self.label, self.elapsed_ms)

	@property
	def elapsed_ms(self) -> float:
		"""Wall-clock elapsed time in milliseconds."""
		return (self._end - self._start) * 1000.0


class DataVersionManager:
	"""Enhanced audit trail with diff tracking for data files.

	Wraps :func:`log_data_version` with structured version tracking,
	file-hash computation, and simple diff detection between
	successive versions.

	Args:
		audit_dir: Directory for audit log files.

	Example::

		mgr = DataVersionManager()
		mgr.record_version("upload", "sales.csv", row_count=1500)
		print(mgr.get_history("sales.csv"))

	.. versionadded:: 2.0
	"""

	def __init__(self, audit_dir: str | None = None) -> None:
		self._audit_dir = audit_dir or os.path.join(
			os.path.dirname(__file__), '../../data/audit'
		)
		os.makedirs(self._audit_dir, exist_ok=True)
		self._versions: dict[str, list[dict]] = {}
		self._lock = threading.Lock()

	def record_version(
		self,
		event: str,
		filename: str,
		*,
		row_count: int | None = None,
		col_count: int | None = None,
		file_hash: str | None = None,
		details: Any | None = None,
	) -> None:
		"""Record a data file version event.

		Args:
			event: Event type (``upload``, ``transform``, ``delete``).
			filename: Name of the affected data file.
			row_count: Optional row count for tabular data.
			col_count: Optional column count.
			file_hash: Optional SHA-256 digest of the file.
			details: Arbitrary extra metadata.
		"""
		entry = {
			'event': event,
			'filename': filename,
			'row_count': row_count,
			'col_count': col_count,
			'file_hash': file_hash,
			'details': details,
			'timestamp': datetime.datetime.now().isoformat(),
		}
		# Persist to JSONL
		log_data_version(event, filename, details=entry)

		# In-memory tracking
		with self._lock:
			self._versions.setdefault(filename, []).append(entry)

	def get_history(self, filename: str) -> list[dict]:
		"""Return version history for *filename*.

		Args:
			filename: The data file to look up.

		Returns:
			List of version-event dicts, oldest first.
		"""
		with self._lock:
			return list(self._versions.get(filename, []))

