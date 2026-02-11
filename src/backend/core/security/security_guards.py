from __future__ import annotations

"""
Enterprise Security Guards for RestrictedPython
================================================
Production-grade security guards with audit logging, rate limiting,
threat detection, and configurable security policies.

.. versionadded:: 2.0.0
   Added :class:`AuditLogger`, :class:`RateLimiter`,
   :class:`ThreatDetector`, and :class:`SecurityPolicy`.

Backward Compatibility
----------------------
All v1.x public APIs (:class:`SecurityGuards`, :class:`ResourceManager`,
:class:`CodeValidator`) remain unchanged.  New enterprise classes are
additive and opt-in.
"""

import ast
import hashlib
import json
import logging
import operator
import signal
import sys
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

__all__ = [
    # v1.x (backward compatible)
    "SecurityGuards",
    "ResourceManager",
    "CodeValidator",
    # v2.0 Enterprise additions
    "AuditLogger",
    "AuditEvent",
    "AuditSeverity",
    "RateLimiter",
    "RateLimitExceeded",
    "ThreatDetector",
    "ThreatLevel",
    "ThreatEvent",
    "SecurityPolicy",
    "SecurityPolicyBuilder",
    "get_audit_logger",
    "get_rate_limiter",
    "get_threat_detector",
]

# Handle Windows compatibility for resource module
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    # Windows doesn't have resource module
    HAS_RESOURCE = False
    resource = None

class SecurityGuards:
    """Enhanced security guards for RestrictedPython execution.

    Provides attribute-access guards, item-access guards, guarded import
    controls, write sanitisation, and a restricted builtins factory.
    All guard methods are ``@staticmethod`` so they can be injected
    directly into the sandbox's globals dict.

    Class Attributes:
        DANGEROUS_MODULES: Module names that must never be imported.
        DANGEROUS_BUILTINS: Built-in names that must never be exposed.
        DANGEROUS_ATTRIBUTES: Dunder / introspection attributes that
            must never be accessed.
    """

    # Dangerous modules and functions that should never be accessible
    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'importlib', 'imp', 'marshal', 
        'pickle', 'shelve', 'dill', 'joblib', 'cloudpickle',
        'ctypes', 'multiprocessing', 'threading', 'asyncio',
        'socket', 'urllib', 'requests', 'http', 'ftplib',
        'smtplib', 'telnetlib', 'xmlrpc', 'sqlite3', 'dbm'
    }
    
    DANGEROUS_BUILTINS = {
        'eval', 'exec', 'compile', '__import__', 'open', 'file',
        'input', 'raw_input', 'reload', 'vars', 'globals', 'locals',
        'dir', 'getattr', 'setattr', 'delattr', 'hasattr'
    }
    
    DANGEROUS_ATTRIBUTES = {
        '__class__', '__bases__', '__subclasses__', '__mro__',
        '__globals__', '__code__', '__func__', '__self__',
        'func_globals', 'func_code', 'im_class', 'im_func', 'im_self'
    }
    
    @staticmethod
    def safer_getattr(obj: Any, name: str, default: Any = None, getattr: Any = getattr) -> Any:
        """Guard attribute access, blocking dangerous or private names.

        Args:
            obj: Target object.
            name: Attribute name to access.
            default: Fallback value if attribute is missing.
            getattr: Callable used for the actual lookup.

        Returns:
            The attribute value.

        Raises:
            AttributeError: If the attribute is forbidden.
        """
        if name in SecurityGuards.DANGEROUS_ATTRIBUTES:
            raise AttributeError(f"Access to '{name}' is not allowed")
        
        if name.startswith('_'):
            raise AttributeError(f"Access to private attribute '{name}' is not allowed")
        
        # Additional checks for dangerous patterns
        # Only block truly dangerous operations that can access/modify system internals
        # Allow pandas operations since df is already loaded and isolated in sandbox
        dangerous_names = {
            'subclasses', 'bases', 'mro', 'globals', 'code', 'func', 'self',
            'import', 'builtins', 'file', 'open',
            'to_pickle', 'read_pickle',  # Pickle is dangerous (arbitrary code execution)
            'eval', 'exec', 'compile'     # Code execution
        }
        
        # Block file writing to_* methods ONLY when accessing file system, not df operations
        # These are handled by allowing df operations but blocking FileIO in restricted builtins
        
        if any(dangerous in name.lower() for dangerous in dangerous_names):
            raise AttributeError(f"Access to potentially dangerous attribute '{name}' is not allowed")
        
        return getattr(obj, name, default)
    
    @staticmethod
    def safe_getitem(obj: Any, key: Any) -> Any:
        """Prevent item access to dangerous keys.

        Args:
            obj: Container to index into.
            key: Key or index.

        Returns:
            The item value.

        Raises:
            KeyError: If the key is forbidden.
        """
        # Prevent access to dangerous keys
        if isinstance(key, str):
            if key in SecurityGuards.DANGEROUS_ATTRIBUTES:
                raise KeyError(f"Access to dangerous key '{key}' is not allowed")
            if key.startswith('_'):
                raise KeyError(f"Access to private key '{key}' is not allowed")
        
        return operator.getitem(obj, key)
    
    @staticmethod
    def guarded_write(s: Any) -> str:
        """Sanitise and log sandbox write operations.

        Args:
            s: Value being written.

        Returns:
            The (possibly truncated) string representation.
        """
        # Convert to string safely
        if hasattr(s, '__str__'):
            output = str(s)
        else:
            output = repr(s)
        
        # Limit output size to prevent DoS
        if len(output) > 1000:
            output = output[:1000] + "...(truncated)"
        
        # Log instead of direct output to prevent information leakage
        logger.info("Sandbox output: %s", output)
        return output
    
    @staticmethod
    def safe_print(*args: Any, **kwargs: Any) -> None:
        """Log output instead of printing directly.

        Args:
            *args: Positional arguments identical to built-in ``print``.
            **kwargs: Keyword arguments (ignored).
        """
        # Convert all arguments to safe strings
        safe_args: list[str] = []
        for arg in args:
            try:
                arg_str = str(arg)
                if len(arg_str) > 500:  # Limit individual argument size
                    arg_str = arg_str[:500] + "...(truncated)"
                safe_args.append(arg_str)
            except Exception:
                safe_args.append("<unprintable>")
        
        output = " ".join(safe_args)
        logger.info("Sandbox print: %s", output)
        return None  # print returns None
    
    @staticmethod
    def safer_setattr(obj: Any, name: str, value: Any) -> None:
        """Guard attribute assignment, blocking dangerous or private names.

        Args:
            obj: Target object.
            name: Attribute name.
            value: Value to set.

        Raises:
            AttributeError: If the attribute is forbidden.
        """
        if name in SecurityGuards.DANGEROUS_ATTRIBUTES or name.startswith('_'):
            raise AttributeError(f"Setting attribute '{name}' is not allowed")
        
        return setattr(obj, name, value)
    
    @staticmethod
    def safer_delattr(obj: Any, name: str) -> None:
        """Guard attribute deletion, blocking dangerous or private names.

        Args:
            obj: Target object.
            name: Attribute name.

        Raises:
            AttributeError: If the attribute is forbidden.
        """
        if name in SecurityGuards.DANGEROUS_ATTRIBUTES or name.startswith('_'):
            raise AttributeError(f"Deleting attribute '{name}' is not allowed")
        
        return delattr(obj, name)
    
    @staticmethod
    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """Allow imports only from an explicit safe-list.

        Args:
            name: Fully-qualified module name.
            globals: Global namespace (unused).
            locals: Local namespace (unused).
            fromlist: Names to import from *name*.
            level: Relative import level.

        Returns:
            The imported module.

        Raises:
            ImportError: If the module is not in the allow-list.
        """
        
        # Check if the base module is in dangerous list
        base_module = name.split('.')[0]
        if base_module in SecurityGuards.DANGEROUS_MODULES:
            raise ImportError(f"Import of '{name}' is not allowed")
        
        # Only allow specific safe modules
        allowed_modules = {
            'pandas', 'numpy', 'matplotlib', 'plotly', 'seaborn',
            'scipy', 'sklearn', 'statsmodels', 'math', 'datetime',
            'json', 'csv', 'collections', 'itertools', 'functools',
            're', 'string', 'textwrap', 'unicodedata'
        }
        
        if base_module not in allowed_modules:
            raise ImportError(f"Import of '{name}' is not allowed. Allowed modules: {allowed_modules}")
        
        return __import__(name, globals, locals, fromlist, level)
    
    @staticmethod
    def create_safe_builtins() -> dict[str, Any]:
        """Build a restricted builtins dict for sandbox execution.

        Returns:
            Mapping of allowed built-in names to their implementations.
        """
        # Start with empty dict - DO NOT inherit from RestrictedPython's safe_builtins
        # as it may contain vulnerabilities
        safe_dict = {}
        
        # Only add explicitly vetted safe operations
        safe_dict.update({
            # Basic type constructors - safe
            'bool': bool,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            'frozenset': frozenset,
            
            # Safe numeric operations
            'len': len,
            'range': range,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'pow': pow,
            'divmod': divmod,
            
            # Safe iteration
            'enumerate': enumerate,
            'zip': zip,
            'reversed': reversed,
            'sorted': sorted,
            'iter': iter,
            'next': next,
            
            # Safe higher-order functions
            'map': map,
            'filter': filter,
            'all': all,
            'any': any,
            
            # Safe type checking
            'type': type,
            'isinstance': isinstance,
            'issubclass': issubclass,
            
            # Safe string/numeric conversion
            'chr': chr,
            'ord': ord,
            'bin': bin,
            'oct': oct,
            'hex': hex,
            
            # Restricted Python operators (required for basic operations)
            '_getitem_': SecurityGuards.safe_getitem,
            '_getattr_': SecurityGuards.safer_getattr,
            '_getiter_': iter,
            '_write_': SecurityGuards.guarded_write,
            '__import__': SecurityGuards.guarded_import,
            
            # Safe print function
            'print': SecurityGuards.safe_print,
        })
        
        # Remove dangerous builtins
        for dangerous in SecurityGuards.DANGEROUS_BUILTINS:
            # We want to remove standard __import__ but keep our guarded one if we set it.
            # However, since we defined it in the dict above, popping it removes it.
            # So we let it pop, then re-add the guarded version explicitly.
            safe_dict.pop(dangerous, None)
            
        # Re-add guarded restrictions that override dangerous builtins
        safe_dict['__import__'] = SecurityGuards.guarded_import
        
        return safe_dict

class ResourceManager:
    """Enforce memory and CPU limits during sandboxed execution.

    On Unix-like systems uses the ``resource`` module and ``SIGALRM``
    for hard limits.  On Windows falls back to a daemon ``Timer``
    thread that injects a ``TimeoutError`` via ``ctypes``.
    """

    @staticmethod
    @contextmanager
    def limit_resources(
        max_memory_mb: int = 256, max_cpu_seconds: int = 30,
    ) -> Generator[None, None, None]:
        """Limit memory and CPU usage (Windows compatible).

        Args:
            max_memory_mb: Maximum memory in megabytes.
            max_cpu_seconds: Maximum wall-clock seconds.

        Yields:
            Control to the caller while resource limits are active.
        """
        import threading
        import time
        import ctypes
        
        # Set resource limits (Unix-like systems only)
        old_limits = {}
        timeout_thread = None
        timed_out = threading.Event()
        
        try:
            # Unix-like resource limits
            if HAS_RESOURCE and resource:
                if hasattr(resource, 'RLIMIT_AS'):
                    # Memory limit
                    old_limits['memory'] = resource.getrlimit(resource.RLIMIT_AS)
                    resource.setrlimit(resource.RLIMIT_AS, (max_memory_mb * 1024 * 1024, -1))
                
                if hasattr(resource, 'RLIMIT_CPU'):
                    # CPU time limit
                    old_limits['cpu'] = resource.getrlimit(resource.RLIMIT_CPU)
                    resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_seconds, max_cpu_seconds + 5))
                
                # Set timeout signal (Unix only)
                if hasattr(signal, 'SIGALRM'):
                    def signal_timeout_handler(signum, frame):
                        raise TimeoutError(f"Code execution exceeded {max_cpu_seconds} seconds")
                    old_handler = signal.signal(signal.SIGALRM, signal_timeout_handler)
                    signal.alarm(max_cpu_seconds)
            else:
                # Windows fallback: use ctypes to raise exception in the main thread
                main_thread_id = threading.current_thread().ident
                
                def timeout_handler():
                    timed_out.set()
                    # Raise TimeoutError in the target thread via ctypes
                    if main_thread_id is not None:
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_ulong(main_thread_id),
                            ctypes.py_object(TimeoutError)
                        )
                
                timeout_thread = threading.Timer(max_cpu_seconds, timeout_handler)
                timeout_thread.daemon = True
                timeout_thread.start()
            
            yield
            
        except TimeoutError:
            raise TimeoutError(f"Code execution exceeded {max_cpu_seconds} seconds")
        except Exception as e:
            raise e
        finally:
            # Cleanup
            try:
                if timeout_thread and timeout_thread.is_alive():
                    timeout_thread.cancel()
                
                if HAS_RESOURCE and resource:
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)
                        if 'old_handler' in locals():
                            signal.signal(signal.SIGALRM, old_handler)
                    
                    for limit_type, old_limit in old_limits.items():
                        if limit_type == 'memory' and hasattr(resource, 'RLIMIT_AS'):
                            resource.setrlimit(resource.RLIMIT_AS, old_limit)
                        elif limit_type == 'cpu' and hasattr(resource, 'RLIMIT_CPU'):
                            resource.setrlimit(resource.RLIMIT_CPU, old_limit)
            except Exception:
                logger.debug("Resource limit cleanup failed (non-critical)")

class CodeValidator:
    """Pre-execution AST and pattern validation for sandboxed code.

    Provides two complementary layers of defence:

    1. **AST walk** — rejects dangerous function calls, attribute
       accesses, and imports at the syntax-tree level.
    2. **Pattern scan** — string-level check for known dangerous
       substrings that might bypass AST analysis.
    """

    @staticmethod
    def validate_ast(code: str) -> tuple[bool, str]:
        """Walk the AST to reject dangerous calls, attributes, and imports.

        Args:
            code: Python source code.

        Returns:
            ``(is_valid, message)`` tuple.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in SecurityGuards.DANGEROUS_BUILTINS:
                        return False, f"Dangerous function call: {node.func.id}"
            
            # Check for dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in SecurityGuards.DANGEROUS_ATTRIBUTES:
                    return False, f"Dangerous attribute access: {node.attr}"
            
            # Check for import statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base_module = alias.name.split('.')[0]
                    if base_module in SecurityGuards.DANGEROUS_MODULES:
                        return False, f"Dangerous import: {alias.name}"
            
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    base_module = node.module.split('.')[0]
                    if base_module in SecurityGuards.DANGEROUS_MODULES:
                        return False, f"Dangerous import from: {node.module}"
        
        return True, "Code validation passed"
    
    @staticmethod
    def validate_code_patterns(code: str) -> tuple[bool, str]:
        """Scan raw source text for known dangerous string patterns.

        Args:
            code: Python source code.

        Returns:
            ``(is_valid, message)`` tuple.
        """
        dangerous_patterns = [
            'exec(', 'eval(', '__import__', 'open(', 'file(',
            'os.system', 'subprocess', 'socket.', 'urllib.',
            'pickle.', 'marshal.', 'ctypes.', 'globals()',
            'locals()', 'vars()', 'dir()', '__class__',
            '__bases__', '__subclasses__', '__mro__'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return False, f"Dangerous pattern detected: {pattern}"
        
        return True, "Pattern validation passed"


# =============================================================================
# ENTERPRISE: AUDIT LOGGING (SOC2/HIPAA Compliance)
# =============================================================================

class AuditSeverity(Enum):
    """Severity levels for audit events, aligned with syslog/CEF standards."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Immutable record of a security-relevant action.

    Designed for compliance with SOC2 CC6.1 (logical access controls)
    and HIPAA 164.312(b) (audit controls).

    Attributes:
        event_id: Unique UUID for this event.
        timestamp: ISO-8601 timestamp when the event occurred.
        event_type: Category (e.g. ``'code_execution'``, ``'access_denied'``).
        severity: :class:`AuditSeverity` level.
        actor: Identifier of the user or system component.
        action: Specific action taken (e.g. ``'execute'``, ``'import_blocked'``).
        resource: Target resource (e.g. code hash, module name).
        outcome: ``'success'``, ``'failure'``, or ``'blocked'``.
        details: Free-form metadata dict.
        ip_address: Client IP when available.
        session_id: Session or request correlation identifier.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    event_type: str = ""
    severity: AuditSeverity = AuditSeverity.INFO
    actor: str = "system"
    action: str = ""
    resource: str = ""
    outcome: str = "success"
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "severity": self.severity.value,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "details": self.details,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
        }

    def to_cef(self) -> str:
        """Format as Common Event Format (CEF) for SIEM integration.

        Returns:
            CEF-formatted string suitable for Splunk, QRadar, etc.
        """
        severity_map = {
            AuditSeverity.DEBUG: 1,
            AuditSeverity.INFO: 3,
            AuditSeverity.WARNING: 5,
            AuditSeverity.ERROR: 7,
            AuditSeverity.CRITICAL: 10,
        }
        cef_severity = severity_map.get(self.severity, 3)
        return (
            f"CEF:0|NexusLLM|SecurityGuard|2.0|{self.event_type}|"
            f"{self.action}|{cef_severity}|src={self.ip_address} "
            f"act={self.action} outcome={self.outcome} "
            f"msg={self.resource}"
        )


class AuditLogger:
    """Enterprise-grade audit logger with file persistence and rotation.

    Thread-safe. Supports structured JSON logging, CEF format for SIEM
    integration, log rotation by size, and configurable retention policies.

    .. code-block:: python

        audit = get_audit_logger()
        audit.log_execution("sandbox", "exec_code", code_hash, "success")
        audit.log_access_denied("user_123", "os.system", "blocked_import")

    Args:
        log_dir: Directory for audit log files.
        max_file_size_mb: Rotate when a log file exceeds this size.
        max_files: Maximum number of rotated log files to keep.
        enable_cef: When ``True``, also write CEF-formatted events.
    """

    def __init__(
        self,
        log_dir: str = "data/audit",
        max_file_size_mb: int = 50,
        max_files: int = 10,
        enable_cef: bool = False,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._max_file_size = max_file_size_mb * 1024 * 1024
        self._max_files = max_files
        self._enable_cef = enable_cef
        self._lock = threading.Lock()
        self._events: List[AuditEvent] = []
        self._event_count = 0
        self._current_file = self._log_dir / "security_audit.jsonl"
        logger.info("AuditLogger initialized: dir=%s, cef=%s", log_dir, enable_cef)

    def log(self, event: AuditEvent) -> None:
        """Record an audit event (thread-safe).

        Args:
            event: The :class:`AuditEvent` to persist.
        """
        with self._lock:
            self._events.append(event)
            self._event_count += 1
            self._persist_event(event)

        if event.severity in (AuditSeverity.ERROR, AuditSeverity.CRITICAL):
            logger.warning(
                "AUDIT[%s]: %s/%s → %s (%s)",
                event.severity.value, event.actor, event.action,
                event.outcome, event.resource,
            )

    def log_execution(
        self, actor: str, action: str, resource: str, outcome: str,
        **details: Any,
    ) -> AuditEvent:
        """Convenience: log a code-execution event.

        Args:
            actor: Identity of the executor.
            action: Execution action (e.g. ``'sandbox_exec'``).
            resource: Code hash or identifier.
            outcome: ``'success'`` | ``'failure'`` | ``'blocked'``.
            **details: Extra metadata.

        Returns:
            The created :class:`AuditEvent`.
        """
        event = AuditEvent(
            event_type="code_execution",
            severity=AuditSeverity.INFO if outcome == "success" else AuditSeverity.WARNING,
            actor=actor, action=action, resource=resource,
            outcome=outcome, details=details,
        )
        self.log(event)
        return event

    def log_access_denied(
        self, actor: str, resource: str, reason: str, **details: Any,
    ) -> AuditEvent:
        """Convenience: log a blocked-access event.

        Args:
            actor: Identity of the requester.
            resource: Resource that was denied.
            reason: Human-readable denial reason.
            **details: Extra metadata.

        Returns:
            The created :class:`AuditEvent`.
        """
        event = AuditEvent(
            event_type="access_denied",
            severity=AuditSeverity.WARNING,
            actor=actor, action="access_denied", resource=resource,
            outcome="blocked", details={"reason": reason, **details},
        )
        self.log(event)
        return event

    def log_threat_detected(
        self, threat_type: str, source: str, details: Dict[str, Any],
    ) -> AuditEvent:
        """Convenience: log a threat detection event.

        Args:
            threat_type: Category of the detected threat.
            source: Origin of the threat.
            details: Threat metadata.

        Returns:
            The created :class:`AuditEvent`.
        """
        event = AuditEvent(
            event_type="threat_detected",
            severity=AuditSeverity.CRITICAL,
            actor=source, action="threat_detected",
            resource=threat_type, outcome="blocked", details=details,
        )
        self.log(event)
        return event

    def _persist_event(self, event: AuditEvent) -> None:
        """Write event to disk as JSONL (and optionally CEF)."""
        try:
            self._maybe_rotate()
            with open(self._current_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
            if self._enable_cef:
                cef_path = self._log_dir / "security_audit.cef"
                with open(cef_path, "a", encoding="utf-8") as f:
                    f.write(event.to_cef() + "\n")
        except Exception as e:
            logger.error("Audit persistence failed: %s", e)

    def _maybe_rotate(self) -> None:
        """Rotate log file if it exceeds the configured size limit."""
        if self._current_file.exists() and self._current_file.stat().st_size > self._max_file_size:
            rotated = self._log_dir / f"security_audit_{int(time.time())}.jsonl"
            self._current_file.rename(rotated)
            # Prune old files
            log_files = sorted(self._log_dir.glob("security_audit_*.jsonl"))
            while len(log_files) > self._max_files:
                log_files.pop(0).unlink(missing_ok=True)

    def get_events(
        self, since: Optional[float] = None, event_type: Optional[str] = None,
        severity: Optional[AuditSeverity] = None, limit: int = 100,
    ) -> List[AuditEvent]:
        """Query recorded events with optional filters.

        Args:
            since: Unix timestamp; return events after this time.
            event_type: Filter by event type string.
            severity: Filter by minimum severity.
            limit: Maximum events to return.

        Returns:
            Filtered list of :class:`AuditEvent` instances.
        """
        with self._lock:
            filtered = self._events
            if event_type:
                filtered = [e for e in filtered if e.event_type == event_type]
            if severity:
                sev_order = list(AuditSeverity)
                min_idx = sev_order.index(severity)
                filtered = [e for e in filtered if sev_order.index(e.severity) >= min_idx]
            return filtered[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate audit statistics.

        Returns:
            Dict with total counts, per-type breakdown, per-severity
            breakdown, and per-outcome breakdown.
        """
        with self._lock:
            by_type: Dict[str, int] = defaultdict(int)
            by_severity: Dict[str, int] = defaultdict(int)
            by_outcome: Dict[str, int] = defaultdict(int)
            for e in self._events:
                by_type[e.event_type] += 1
                by_severity[e.severity.value] += 1
                by_outcome[e.outcome] += 1
            return {
                "total_events": self._event_count,
                "by_type": dict(by_type),
                "by_severity": dict(by_severity),
                "by_outcome": dict(by_outcome),
            }


# =============================================================================
# ENTERPRISE: RATE LIMITER (DoS Protection)
# =============================================================================

class RateLimitExceeded(Exception):
    """Raised when a rate limit has been exceeded.

    Attributes:
        identifier: The rate-limited identity.
        limit: The configured maximum requests per window.
        window_seconds: The time window in seconds.
        retry_after: Seconds until the limit resets.
    """

    def __init__(self, identifier: str, limit: int, window_seconds: int, retry_after: float) -> None:
        self.identifier = identifier
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for '{identifier}': "
            f"{limit} requests per {window_seconds}s. "
            f"Retry after {retry_after:.1f}s."
        )


class RateLimiter:
    """Sliding-window rate limiter for sandbox execution and API endpoints.

    Thread-safe, per-identity rate limiting with configurable windows,
    burst allowances, and automatic cleanup of expired entries.

    .. code-block:: python

        limiter = get_rate_limiter()
        limiter.check("user_123")  # Raises RateLimitExceeded if over limit

    Args:
        default_limit: Default maximum requests per window.
        default_window_seconds: Default sliding window duration.
        burst_multiplier: Factor by which the limit is multiplied for
            short bursts (first 10% of window).
        cleanup_interval: Seconds between expired-entry cleanup passes.
    """

    def __init__(
        self,
        default_limit: int = 60,
        default_window_seconds: int = 60,
        burst_multiplier: float = 1.5,
        cleanup_interval: float = 300,
    ) -> None:
        self._default_limit = default_limit
        self._default_window = default_window_seconds
        self._burst_multiplier = burst_multiplier
        self._cleanup_interval = cleanup_interval
        self._lock = threading.Lock()
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._custom_limits: Dict[str, tuple] = {}  # id → (limit, window)
        self._last_cleanup = time.time()
        self._blocked_count = 0
        self._total_checks = 0
        logger.info(
            "RateLimiter initialized: limit=%d/%ds, burst=%.1fx",
            default_limit, default_window_seconds, burst_multiplier,
        )

    def set_limit(self, identifier: str, limit: int, window_seconds: int) -> None:
        """Set a custom rate limit for a specific identifier.

        Args:
            identifier: The identity to configure (user ID, IP, etc.).
            limit: Maximum requests per window.
            window_seconds: Sliding window duration.
        """
        with self._lock:
            self._custom_limits[identifier] = (limit, window_seconds)

    def check(self, identifier: str = "default") -> bool:
        """Check and record a request against the rate limit.

        Args:
            identifier: The identity making the request.

        Returns:
            ``True`` if the request is allowed.

        Raises:
            RateLimitExceeded: When the limit has been exceeded.
        """
        now = time.time()
        with self._lock:
            self._total_checks += 1
            limit, window = self._custom_limits.get(
                identifier, (self._default_limit, self._default_window)
            )
            # Purge expired timestamps
            cutoff = now - window
            self._requests[identifier] = [
                t for t in self._requests[identifier] if t > cutoff
            ]
            # Burst check (first 10% of window allows burst_multiplier × limit)
            current_count = len(self._requests[identifier])
            effective_limit = int(limit * self._burst_multiplier) if current_count < limit * 0.1 else limit

            if current_count >= effective_limit:
                self._blocked_count += 1
                oldest = self._requests[identifier][0] if self._requests[identifier] else now
                retry_after = oldest + window - now
                raise RateLimitExceeded(identifier, limit, window, max(0, retry_after))

            self._requests[identifier].append(now)
            self._maybe_cleanup(now)
            return True

    def _maybe_cleanup(self, now: float) -> None:
        """Remove expired entries periodically."""
        if now - self._last_cleanup > self._cleanup_interval:
            expired_ids = [
                k for k, v in self._requests.items() if not v
            ]
            for k in expired_ids:
                del self._requests[k]
            self._last_cleanup = now

    def get_usage(self, identifier: str) -> Dict[str, Any]:
        """Get current rate-limit usage for an identifier.

        Args:
            identifier: The identity to query.

        Returns:
            Dict with ``current``, ``limit``, ``window_seconds``,
            ``remaining``, and ``reset_in`` fields.
        """
        with self._lock:
            limit, window = self._custom_limits.get(
                identifier, (self._default_limit, self._default_window)
            )
            now = time.time()
            cutoff = now - window
            active = [t for t in self._requests.get(identifier, []) if t > cutoff]
            return {
                "current": len(active),
                "limit": limit,
                "window_seconds": window,
                "remaining": max(0, limit - len(active)),
                "reset_in": round(active[0] + window - now, 1) if active else 0,
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate rate-limiter statistics.

        Returns:
            Dict with total checks, blocks, block rate, active identities,
            and per-identity usage summaries.
        """
        with self._lock:
            return {
                "total_checks": self._total_checks,
                "total_blocked": self._blocked_count,
                "block_rate": round(
                    self._blocked_count / max(self._total_checks, 1) * 100, 2
                ),
                "active_identities": len(self._requests),
                "custom_limits": len(self._custom_limits),
            }


# =============================================================================
# ENTERPRISE: THREAT DETECTOR (Advanced Pattern Analysis)
# =============================================================================

class ThreatLevel(Enum):
    """Categorised threat severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ThreatEvent:
    """Record of a detected security threat.

    Attributes:
        threat_id: Unique identifier.
        timestamp: ISO-8601 time of detection.
        threat_type: Category (e.g. ``'injection'``, ``'sandbox_escape'``).
        level: :class:`ThreatLevel` severity.
        source: Origin or actor responsible.
        description: Human-readable explanation.
        code_snippet: Excerpt of offending code.
        indicators: List of specific matched indicators.
        mitigated: Whether the threat was automatically blocked.
    """
    threat_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    threat_type: str = ""
    level: ThreatLevel = ThreatLevel.NONE
    source: str = ""
    description: str = ""
    code_snippet: str = ""
    indicators: List[str] = field(default_factory=list)
    mitigated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "threat_id": self.threat_id,
            "timestamp": self.timestamp,
            "threat_type": self.threat_type,
            "level": self.level.value,
            "source": self.source,
            "description": self.description,
            "code_snippet": self.code_snippet[:200],
            "indicators": self.indicators,
            "mitigated": self.mitigated,
        }


class ThreatDetector:
    """Multi-layer threat detection engine for sandbox code analysis.

    Performs layered detection combining:
    1. **Pattern matching** — regex-based signatures for known attack vectors.
    2. **Behavioural analysis** — entropy, obfuscation, and structural anomalies.
    3. **AST introspection** — deep analysis of code structure for hidden threats.

    All detections are logged via :class:`AuditLogger` when an instance
    is provided.

    Args:
        audit_logger: Optional audit logger for automatic threat logging.
        custom_signatures: Additional regex patterns to include in scans.
    """

    # ---- Signature database ----
    _INJECTION_PATTERNS: List[tuple] = [
        (r'__import__\s*\(\s*["\']os["\']\s*\)', "os import injection", ThreatLevel.CRITICAL),
        (r'getattr\s*\(\s*__builtins__', "builtins attribute bypass", ThreatLevel.CRITICAL),
        (r'eval\s*\(\s*compile\s*\(', "eval-compile chain", ThreatLevel.CRITICAL),
        (r'\(\).__class__.__bases__', "type hierarchy traversal", ThreatLevel.CRITICAL),
        (r'exec\s*\(\s*bytes\s*\(', "bytes-exec injection", ThreatLevel.CRITICAL),
        (r'__subclasses__\s*\(\s*\)', "subclass enumeration", ThreatLevel.HIGH),
        (r'os\.environ', "environment variable access", ThreatLevel.HIGH),
        (r'subprocess\.(?:run|Popen|call)', "subprocess execution", ThreatLevel.CRITICAL),
        (r'socket\.socket', "raw socket creation", ThreatLevel.HIGH),
        (r'ctypes\.(?:cdll|windll|CFUNCTYPE)', "ctypes FFI bypass", ThreatLevel.CRITICAL),
        (r'pickle\.loads?\s*\(', "deserialization attack", ThreatLevel.HIGH),
        (r'marshal\.loads?\s*\(', "marshal deserialization", ThreatLevel.HIGH),
        (r'__code__\s*=', "code object manipulation", ThreatLevel.CRITICAL),
        (r'sys\._getframe', "stack frame access", ThreatLevel.HIGH),
        (r'importlib\.import_module', "dynamic import bypass", ThreatLevel.HIGH),
    ]

    _OBFUSCATION_PATTERNS: List[tuple] = [
        (r'chr\s*\(\s*\d+\s*\)\s*\+\s*chr', "character-by-character obfuscation", ThreatLevel.MEDIUM),
        (r'\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}', "hex-encoded payload", ThreatLevel.MEDIUM),
        (r'base64\.b64decode', "base64 payload decode", ThreatLevel.MEDIUM),
        (r'codecs\.decode\s*\(.*rot_?13', "ROT13 obfuscation", ThreatLevel.MEDIUM),
        (r'bytes\.fromhex', "hex bytes construction", ThreatLevel.MEDIUM),
    ]

    _EXFILTRATION_PATTERNS: List[tuple] = [
        (r'requests\.(?:get|post|put)', "HTTP request attempt", ThreatLevel.HIGH),
        (r'urllib\.request\.urlopen', "URL access attempt", ThreatLevel.HIGH),
        (r'http\.client\.HTTP', "HTTP client creation", ThreatLevel.HIGH),
        (r'smtplib\.SMTP', "SMTP mail attempt", ThreatLevel.HIGH),
        (r'ftplib\.FTP', "FTP connection attempt", ThreatLevel.HIGH),
    ]

    def __init__(
        self,
        audit_logger: Optional[AuditLogger] = None,
        custom_signatures: Optional[List[tuple]] = None,
    ) -> None:
        self._audit = audit_logger
        self._lock = threading.Lock()
        self._threats: List[ThreatEvent] = []
        self._scan_count = 0
        self._threat_count = 0

        import re as _re
        self._re = _re

        # Compile all signature patterns
        self._signatures: List[tuple] = []
        for category in (self._INJECTION_PATTERNS, self._OBFUSCATION_PATTERNS, self._EXFILTRATION_PATTERNS):
            self._signatures.extend(category)
        if custom_signatures:
            self._signatures.extend(custom_signatures)

        self._compiled = [(self._re.compile(p, self._re.IGNORECASE), desc, lvl) for p, desc, lvl in self._signatures]
        logger.info("ThreatDetector initialized: %d signatures loaded", len(self._compiled))

    def scan(self, code: str, source: str = "unknown") -> List[ThreatEvent]:
        """Perform a full multi-layer security scan on code.

        Args:
            code: Python source code to analyse.
            source: Identifier for the code source (user, session, etc.).

        Returns:
            List of detected :class:`ThreatEvent` instances.
        """
        with self._lock:
            self._scan_count += 1

        events: List[ThreatEvent] = []

        # Layer 1: Signature-based detection
        events.extend(self._scan_signatures(code, source))

        # Layer 2: Behavioural analysis
        events.extend(self._scan_behavioural(code, source))

        # Layer 3: AST deep inspection
        events.extend(self._scan_ast(code, source))

        # Record and audit
        with self._lock:
            self._threats.extend(events)
            self._threat_count += len(events)

        if events and self._audit:
            worst = max(events, key=lambda e: list(ThreatLevel).index(e.level))
            self._audit.log_threat_detected(
                threat_type=worst.threat_type,
                source=source,
                details={"total_threats": len(events), "highest_level": worst.level.value},
            )

        return events

    def _scan_signatures(self, code: str, source: str) -> List[ThreatEvent]:
        """Layer 1: Pattern-based signature scanning."""
        events = []
        for compiled_re, desc, level in self._compiled:
            matches = compiled_re.findall(code)
            if matches:
                events.append(ThreatEvent(
                    threat_type="signature_match",
                    level=level,
                    source=source,
                    description=desc,
                    code_snippet=matches[0][:100] if matches else "",
                    indicators=[desc],
                    mitigated=True,
                ))
        return events

    def _scan_behavioural(self, code: str, source: str) -> List[ThreatEvent]:
        """Layer 2: Behavioural anomaly detection."""
        events = []

        # High entropy detection (obfuscated code tends to have high entropy)
        entropy = self._calculate_entropy(code)
        if entropy > 5.5 and len(code) > 100:
            events.append(ThreatEvent(
                threat_type="behavioural_anomaly",
                level=ThreatLevel.MEDIUM,
                source=source,
                description=f"High code entropy ({entropy:.2f}) suggests obfuscation",
                indicators=["high_entropy"],
                mitigated=False,
            ))

        # Excessive nesting depth
        max_depth = self._calculate_nesting_depth(code)
        if max_depth > 10:
            events.append(ThreatEvent(
                threat_type="behavioural_anomaly",
                level=ThreatLevel.LOW,
                source=source,
                description=f"Excessive nesting depth ({max_depth}) may indicate obfuscation",
                indicators=["deep_nesting"],
                mitigated=False,
            ))

        # Unusually long single lines (possible payload embedding)
        for i, line in enumerate(code.split("\n"), 1):
            if len(line) > 1000:
                events.append(ThreatEvent(
                    threat_type="behavioural_anomaly",
                    level=ThreatLevel.LOW,
                    source=source,
                    description=f"Line {i} is {len(line)} chars (possible embedded payload)",
                    indicators=["long_line"],
                    mitigated=False,
                ))
                break  # Report only the first occurrence

        return events

    def _scan_ast(self, code: str, source: str) -> List[ThreatEvent]:
        """Layer 3: AST deep inspection for structural threats."""
        events = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return events  # Let CodeValidator handle syntax errors

        # Check for excessive dynamic attribute access chains
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                chain_len = self._attribute_chain_length(node)
                if chain_len > 5:
                    events.append(ThreatEvent(
                        threat_type="ast_anomaly",
                        level=ThreatLevel.MEDIUM,
                        source=source,
                        description=f"Deep attribute chain (depth={chain_len}) may be sandbox escape",
                        indicators=["deep_attribute_chain"],
                        mitigated=False,
                    ))

            # Detect lambda-based code construction
            if isinstance(node, ast.Lambda):
                body_source = ast.dump(node.body)
                if "Call" in body_source and "Attribute" in body_source:
                    events.append(ThreatEvent(
                        threat_type="ast_anomaly",
                        level=ThreatLevel.MEDIUM,
                        source=source,
                        description="Lambda with method call chain (potential sandbox escape)",
                        indicators=["lambda_method_chain"],
                        mitigated=False,
                    ))

        return events

    @staticmethod
    def _attribute_chain_length(node: ast.Attribute) -> int:
        """Count the depth of a chained attribute access."""
        depth = 1
        current = node.value
        while isinstance(current, ast.Attribute):
            depth += 1
            current = current.value
        return depth

    @staticmethod
    def _calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy of a string (in bits)."""
        import math
        if not text:
            return 0.0
        freq: Dict[str, int] = defaultdict(int)
        for ch in text:
            freq[ch] += 1
        total = len(text)
        return -sum((c / total) * math.log2(c / total) for c in freq.values())

    @staticmethod
    def _calculate_nesting_depth(code: str) -> int:
        """Estimate maximum indentation depth."""
        max_depth = 0
        for line in code.split("\n"):
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                depth = indent // 4  # Assume 4-space indentation
                max_depth = max(max_depth, depth)
        return max_depth

    def get_statistics(self) -> Dict[str, Any]:
        """Return threat detection statistics.

        Returns:
            Dict with scan/threat counts, per-level breakdown, and
            per-type breakdown.
        """
        with self._lock:
            by_level: Dict[str, int] = defaultdict(int)
            by_type: Dict[str, int] = defaultdict(int)
            for t in self._threats:
                by_level[t.level.value] += 1
                by_type[t.threat_type] += 1
            return {
                "total_scans": self._scan_count,
                "total_threats": self._threat_count,
                "by_level": dict(by_level),
                "by_type": dict(by_type),
                "threat_rate": round(
                    self._threat_count / max(self._scan_count, 1) * 100, 2
                ),
            }


# =============================================================================
# ENTERPRISE: SECURITY POLICY ENGINE
# =============================================================================

class SecurityPolicy:
    """Configurable security policy governing sandbox behaviour.

    Encapsulates all tuneable security parameters in a single object so
    that different deployment environments (development, staging,
    production, air-gapped) can use pre-built policy profiles.

    Use :class:`SecurityPolicyBuilder` for a fluent construction API.

    Attributes:
        name: Human-readable policy name.
        max_code_length: Maximum allowed code length in characters.
        max_line_count: Maximum number of lines permitted.
        max_execution_time_seconds: Hard wall-clock timeout.
        max_memory_mb: Maximum memory for sandbox execution.
        allowed_modules: Set of importable module base names.
        blocked_modules: Set of explicitly blocked module base names.
        allowed_builtins: Set of permitted built-in names.
        blocked_builtins: Set of explicitly blocked built-in names.
        blocked_attributes: Set of blocked dunder / private attributes.
        enable_file_io: Whether file I/O operations are permitted.
        enable_network: Whether network operations are permitted.
        enable_subprocess: Whether subprocess execution is permitted.
        enable_threading: Whether threading is permitted.
        rate_limit: Maximum executions per minute per identity.
        require_audit: Whether all executions must be audit-logged.
        threat_scan_required: Whether threat scanning is mandatory.
    """

    def __init__(
        self,
        name: str = "default",
        max_code_length: int = 10_000,
        max_line_count: int = 500,
        max_execution_time_seconds: int = 120,
        max_memory_mb: int = 512,
        allowed_modules: Optional[Set[str]] = None,
        blocked_modules: Optional[Set[str]] = None,
        allowed_builtins: Optional[Set[str]] = None,
        blocked_builtins: Optional[Set[str]] = None,
        blocked_attributes: Optional[Set[str]] = None,
        enable_file_io: bool = False,
        enable_network: bool = False,
        enable_subprocess: bool = False,
        enable_threading: bool = False,
        rate_limit: int = 60,
        require_audit: bool = True,
        threat_scan_required: bool = True,
    ) -> None:
        self.name = name
        self.max_code_length = max_code_length
        self.max_line_count = max_line_count
        self.max_execution_time_seconds = max_execution_time_seconds
        self.max_memory_mb = max_memory_mb
        self.allowed_modules = allowed_modules or SecurityGuards.guarded_import.__func__.__code__.co_consts  # noqa — fallback
        self.allowed_modules = allowed_modules if allowed_modules is not None else {
            "pandas", "numpy", "matplotlib", "plotly", "seaborn",
            "scipy", "sklearn", "statsmodels", "math", "datetime",
            "json", "csv", "collections", "itertools", "functools",
            "re", "string", "textwrap", "unicodedata",
        }
        self.blocked_modules = blocked_modules if blocked_modules is not None else set(SecurityGuards.DANGEROUS_MODULES)
        self.allowed_builtins = allowed_builtins or set()
        self.blocked_builtins = blocked_builtins if blocked_builtins is not None else set(SecurityGuards.DANGEROUS_BUILTINS)
        self.blocked_attributes = blocked_attributes if blocked_attributes is not None else set(SecurityGuards.DANGEROUS_ATTRIBUTES)
        self.enable_file_io = enable_file_io
        self.enable_network = enable_network
        self.enable_subprocess = enable_subprocess
        self.enable_threading = enable_threading
        self.rate_limit = rate_limit
        self.require_audit = require_audit
        self.threat_scan_required = threat_scan_required

    def validate_code(self, code: str) -> tuple:
        """Validate code against this policy.

        Args:
            code: Python source code.

        Returns:
            ``(is_valid, message)`` tuple.
        """
        if len(code) > self.max_code_length:
            return False, f"Code exceeds policy limit ({len(code)} > {self.max_code_length})"
        if code.count("\n") > self.max_line_count:
            return False, f"Too many lines ({code.count(chr(10))} > {self.max_line_count})"
        return True, "Policy validation passed"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy to a dictionary."""
        return {
            "name": self.name,
            "max_code_length": self.max_code_length,
            "max_line_count": self.max_line_count,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "max_memory_mb": self.max_memory_mb,
            "allowed_modules": sorted(self.allowed_modules),
            "blocked_modules": sorted(self.blocked_modules),
            "enable_file_io": self.enable_file_io,
            "enable_network": self.enable_network,
            "rate_limit": self.rate_limit,
            "require_audit": self.require_audit,
            "threat_scan_required": self.threat_scan_required,
        }

    # -- Pre-built profiles --
    @classmethod
    def development(cls) -> "SecurityPolicy":
        """Relaxed policy for local development."""
        return cls(
            name="development",
            max_code_length=50_000,
            max_line_count=2_000,
            max_execution_time_seconds=300,
            max_memory_mb=1024,
            rate_limit=1000,
            require_audit=False,
            threat_scan_required=False,
        )

    @classmethod
    def production(cls) -> "SecurityPolicy":
        """Strict policy for production deployments."""
        return cls(
            name="production",
            max_code_length=10_000,
            max_line_count=500,
            max_execution_time_seconds=60,
            max_memory_mb=512,
            rate_limit=30,
            require_audit=True,
            threat_scan_required=True,
        )

    @classmethod
    def air_gapped(cls) -> "SecurityPolicy":
        """Maximum-security policy for air-gapped environments."""
        return cls(
            name="air_gapped",
            max_code_length=5_000,
            max_line_count=200,
            max_execution_time_seconds=30,
            max_memory_mb=256,
            rate_limit=10,
            require_audit=True,
            threat_scan_required=True,
            enable_file_io=False,
            enable_network=False,
            enable_subprocess=False,
            enable_threading=False,
        )


class SecurityPolicyBuilder:
    """Fluent builder for constructing custom :class:`SecurityPolicy` instances.

    .. code-block:: python

        policy = (
            SecurityPolicyBuilder("custom_policy")
            .max_code_length(20_000)
            .max_memory_mb(1024)
            .allow_module("polars")
            .require_audit(True)
            .build()
        )
    """

    def __init__(self, name: str = "custom") -> None:
        self._kwargs: Dict[str, Any] = {"name": name}
        self._extra_modules: Set[str] = set()
        self._extra_blocked: Set[str] = set()

    def max_code_length(self, value: int) -> "SecurityPolicyBuilder":
        self._kwargs["max_code_length"] = value
        return self

    def max_line_count(self, value: int) -> "SecurityPolicyBuilder":
        self._kwargs["max_line_count"] = value
        return self

    def max_execution_time(self, seconds: int) -> "SecurityPolicyBuilder":
        self._kwargs["max_execution_time_seconds"] = seconds
        return self

    def max_memory_mb(self, value: int) -> "SecurityPolicyBuilder":
        self._kwargs["max_memory_mb"] = value
        return self

    def allow_module(self, module: str) -> "SecurityPolicyBuilder":
        self._extra_modules.add(module)
        return self

    def block_module(self, module: str) -> "SecurityPolicyBuilder":
        self._extra_blocked.add(module)
        return self

    def rate_limit(self, value: int) -> "SecurityPolicyBuilder":
        self._kwargs["rate_limit"] = value
        return self

    def require_audit(self, value: bool) -> "SecurityPolicyBuilder":
        self._kwargs["require_audit"] = value
        return self

    def threat_scan_required(self, value: bool) -> "SecurityPolicyBuilder":
        self._kwargs["threat_scan_required"] = value
        return self

    def build(self) -> SecurityPolicy:
        """Construct the :class:`SecurityPolicy` from accumulated settings.

        Returns:
            Configured ``SecurityPolicy`` instance.
        """
        policy = SecurityPolicy(**self._kwargs)
        policy.allowed_modules |= self._extra_modules
        policy.blocked_modules |= self._extra_blocked
        return policy


# =============================================================================
# SINGLETONS (Enterprise services)
# =============================================================================

_audit_logger: Optional[AuditLogger] = None
_rate_limiter: Optional[RateLimiter] = None
_threat_detector: Optional[ThreatDetector] = None
_enterprise_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """Get or create the singleton :class:`AuditLogger` (thread-safe).

    Returns:
        Shared audit logger instance.
    """
    global _audit_logger
    if _audit_logger is None:
        with _enterprise_lock:
            if _audit_logger is None:
                _audit_logger = AuditLogger()
    return _audit_logger


def get_rate_limiter() -> RateLimiter:
    """Get or create the singleton :class:`RateLimiter` (thread-safe).

    Returns:
        Shared rate limiter instance.
    """
    global _rate_limiter
    if _rate_limiter is None:
        with _enterprise_lock:
            if _rate_limiter is None:
                _rate_limiter = RateLimiter()
    return _rate_limiter


def get_threat_detector() -> ThreatDetector:
    """Get or create the singleton :class:`ThreatDetector` (thread-safe).

    Returns:
        Shared threat detector instance (configured with the singleton audit logger).
    """
    global _threat_detector
    if _threat_detector is None:
        with _enterprise_lock:
            if _threat_detector is None:
                _threat_detector = ThreatDetector(audit_logger=get_audit_logger())
    return _threat_detector