from __future__ import annotations

"""
Security Components
===================
Enterprise-grade sandbox and security guard implementations for safe
LLM-generated code execution.

.. versionadded:: 2.0.0
   Enterprise security components: audit logging, rate limiting,
   threat detection, security policies, circuit breaker, execution
   quotas, and compliance logging.

Submodules
----------
sandbox
    RestrictedPython-based secure execution environment with resource
    limits, AST validation, restricted module access, circuit breaker,
    execution quotas, and compliance audit trail.
security_guards
    Attribute guards, import controls, resource management,
    pre-execution code validation, audit logging, rate limiting,
    threat detection, and configurable security policies.

Backward Compatibility
----------------------
All v1.x public names remain available at the same import paths.
"""

__all__ = [
    # v1.x — sandbox (backward compatible)
    "EnhancedSandbox",
    "Sandbox",
    # v1.x — security_guards (backward compatible)
    "SecurityGuards",
    "ResourceManager",
    "CodeValidator",
    # v2.0 — enterprise sandbox additions
    "EnterpriseSandbox",
    "ExecutionQuota",
    "SandboxCircuitBreaker",
    "CircuitState",
    "ComplianceLogger",
    "ExecutionMetrics",
    # v2.0 — enterprise security_guards additions
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

from .sandbox import (  # noqa: E402
    EnhancedSandbox,
    Sandbox,
    EnterpriseSandbox,
    ExecutionQuota,
    SandboxCircuitBreaker,
    CircuitState,
    ComplianceLogger,
    ExecutionMetrics,
)
from .security_guards import (  # noqa: E402
    CodeValidator,
    ResourceManager,
    SecurityGuards,
    AuditLogger,
    AuditEvent,
    AuditSeverity,
    RateLimiter,
    RateLimitExceeded,
    ThreatDetector,
    ThreatLevel,
    ThreatEvent,
    SecurityPolicy,
    SecurityPolicyBuilder,
    get_audit_logger,
    get_rate_limiter,
    get_threat_detector,
)
