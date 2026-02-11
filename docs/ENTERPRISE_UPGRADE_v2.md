# Nexus LLM Analytics — Enterprise Upgrade v2.0

## Overview

Version 2.0 adds **enterprise-grade** capabilities to the security and engine
modules while maintaining **full backward compatibility** with all v1.x APIs.

Every existing class, function, and import path works exactly as before.
Enterprise features are **additive** — new classes appended after existing code,
new exports added to `__all__` and `__init__.py`.

---

## Upgrade Summary

| Module | File | v2.0 Additions |
|--------|------|----------------|
| **Security** | `security_guards.py` | `AuditLogger`, `RateLimiter`, `ThreatDetector`, `SecurityPolicy`, `SecurityPolicyBuilder` |
| **Security** | `sandbox.py` | `EnterpriseSandbox`, `ExecutionQuota`, `SandboxCircuitBreaker`, `ComplianceLogger`, `ExecutionMetrics` |
| **Engine** | `automated_validation.py` | `ValidationPipeline`, `ValidatorRegistry`, `BaseValidator`, `ConfidenceScorer`, `batch_validate` |
| **Engine** | `intelligent_router.py` | `EnterpriseRouter`, `RouterCircuitBreaker`, `RoutingPolicy`, `ABTestManager`, `RouterHealthCheck` |
| **Engine** | `model_selector.py` | `ModelHealthChecker`, `ModelPool`, `ConnectionManager` |
| **Engine** | `cot_parser.py` | `StreamingCoTParser`, `MultiFormatParser`, `ParseMetrics` |
| **Engine** | `self_correction_engine.py` | `CorrectionStrategy`, `CorrectionObserver`, `CorrectionMetrics` |
| **Engine** | `smart_fallback.py` | `EnterpriseFallbackManager`, `FallbackCircuitBreaker`, `FallbackAnalytics`, `FallbackPolicy` |
| **Engine** | `user_preferences.py` | `PreferenceProfile`, `PreferenceMigrator`, `MultiTenantPreferences`, `PreferenceVersion` |
| **Engine** | `paper_metrics.py` | `PrometheusExporter`, `MetricsDashboard`, `RealTimeTelemetry` |
| **Engine** | `query_orchestrator.py` | `OrchestratorMiddleware`, `ExecutionHook`, `OrchestratorMetrics` |

---

## Backward Compatibility

### Preserved v1.x APIs

All original classes, functions, and their signatures are **unchanged**:

```python
# These all work exactly as before
from backend.core.security import SecurityGuards, EnhancedSandbox, Sandbox
from backend.core.engine import (
    IntelligentRouter, QueryOrchestrator, SelfCorrectionEngine,
    ModelSelector, SmartFallbackManager, CoTParser, CriticParser,
    RuntimeEvaluator, PaperMetricsCollector, UserPreferencesManager,
)
```

### New v2.0 imports

```python
# Enterprise features available alongside v1.x
from backend.core.security import (
    AuditLogger, RateLimiter, ThreatDetector,
    SecurityPolicy, EnterpriseSandbox,
)
from backend.core.engine import (
    EnterpriseRouter, ValidationPipeline, ModelHealthChecker,
    StreamingCoTParser, PrometheusExporter, MultiTenantPreferences,
)
```

---

## Enterprise Features Deep Dive

### 1. Security — Audit Logging (SOC2/HIPAA)

```python
from backend.core.security import get_audit_logger, AuditSeverity

audit = get_audit_logger()
audit.log_event(
    event_type="code_execution",
    severity=AuditSeverity.INFO,
    details={"user": "analyst_1", "query": "show sales trends"},
)

# Exports JSONL + CEF formats with automatic rotation
```

### 2. Security — Rate Limiting

```python
from backend.core.security import get_rate_limiter

limiter = get_rate_limiter()
limiter.check_rate("user_123")  # Raises RateLimitExceeded if over limit
```

### 3. Security — Threat Detection

```python
from backend.core.security import get_threat_detector

detector = get_threat_detector()
threats = detector.analyze("import os; os.system('rm -rf /')")
# Returns ThreatEvent with level=CRITICAL
```

### 4. Routing — Enterprise Router with Circuit Breakers

```python
from backend.core.engine import (
    EnterpriseRouter, RoutingPolicy, RoutingPolicyBuilder,
    get_enterprise_router,
)

policy = (RoutingPolicyBuilder("production")
    .fast_threshold(0.25)
    .balanced_threshold(0.60)
    .max_cost(5.0)
    .latency_sla(3000)
    .block_model("llama3.1:70b")
    .build())

router = get_enterprise_router(policy)
decision = router.route("show total revenue by quarter")

# Record outcomes for circuit breaker learning
router.record_outcome(decision.selected_tier, latency_ms=450, success=True)

# Get enterprise statistics
stats = router.get_enterprise_statistics()
# Includes circuit breaker states, health report, A/B test results
```

### 5. Validation — Enterprise Pipeline

```python
from backend.core.engine import (
    ValidationPipeline, ValidatorRegistry, BaseValidator,
    ConfidenceScorer, batch_validate, get_validator_registry,
)

# Custom validator
class ProfanityValidator(BaseValidator):
    name = "profanity_check"
    priority = 5  # Runs first

    def validate(self, ctx):
        if "bad_word" in ctx.output:
            ctx.issues.append(ValidationIssue(...))
        return ctx

# Register and use
registry = get_validator_registry()
registry.register(ProfanityValidator())

pipeline = ValidationPipeline(registry)
result = pipeline.run(query="...", reasoning="...", output="...", data_context={})
score = ConfidenceScorer().score(result, reasoning="...", output="...")
```

### 6. Model Management — Health Checks & Connection Pooling

```python
from backend.core.engine import (
    get_model_health_checker, get_connection_manager, ModelPool,
)

# Health checks
hc = get_model_health_checker()
hc.check_all()
report = hc.get_report()

# Connection pooling with retries
conn = get_connection_manager()
result = conn.generate("llama3.1:8b", "What is 2+2?", timeout=30)

# Model warm-up pool
pool = ModelPool(max_loaded=3)
pool.warm_up("llama3.1:8b")
```

### 7. Streaming Parser

```python
from backend.core.engine import StreamingCoTParser

parser = StreamingCoTParser()
for token in llm_stream:
    result = parser.feed(token)
    if result.reasoning:
        update_ui("reasoning", result.reasoning)

final = parser.finalize()
```

### 8. Prometheus Metrics Export

```python
from backend.core.engine import get_prometheus_exporter

exporter = get_prometheus_exporter()
prometheus_text = exporter.export()
# Returns standard Prometheus text exposition format
# Ready for /metrics endpoint scraping
```

### 9. Multi-Tenant Preferences

```python
from backend.core.engine import get_multi_tenant_preferences

mt = get_multi_tenant_preferences()
manager = mt.get_manager("tenant_abc")
prefs = manager.load_preferences()
```

### 10. Self-Correction Strategies

```python
from backend.core.engine import CorrectionStrategy, CorrectionObserver

class AggressiveCorrectionStrategy(CorrectionStrategy):
    name = "aggressive"
    def should_apply(self, feedback):
        return "HIGH" in str(feedback)
    def transform_prompt(self, original, feedback, context):
        return f"CRITICAL FIX NEEDED:\n{feedback}\n\n{original}"

class SlackNotificationObserver(CorrectionObserver):
    def on_loop_failure(self, error, iterations_used, **kwargs):
        send_slack_alert(f"Correction failed: {error}")
```

---

## Architecture Principles

1. **Additive Only**: Enterprise classes are appended after existing code.
   No v1.x function signatures were modified.

2. **Thread-Safe Singletons**: All new singletons use double-check locking
   pattern consistent with existing codebase.

3. **Plugin Architecture**: `BaseValidator`, `CorrectionStrategy`,
   `OrchestratorMiddleware` provide extension points without modifying core.

4. **Circuit Breaker Pattern**: Applied to routing, sandbox execution, and
   fallback chains for automatic failure recovery.

5. **Observability**: Every subsystem exposes `get_statistics()` methods.
   `PrometheusExporter` provides standardised metrics export.

6. **Enterprise Policies**: `SecurityPolicy`, `RoutingPolicy`, `FallbackPolicy`
   provide declarative configuration with builder patterns.

---

## File Structure

```
src/backend/core/
├── security/
│   ├── __init__.py          # Updated: 30+ exports (v1.x + v2.0)
│   ├── security_guards.py   # +AuditLogger, RateLimiter, ThreatDetector, SecurityPolicy
│   └── sandbox.py           # +EnterpriseSandbox, ExecutionQuota, CircuitBreaker
└── engine/
    ├── __init__.py           # Updated: 120+ exports (v1.x + v2.0)
    ├── automated_validation.py   # +ValidationPipeline, ValidatorRegistry, ConfidenceScorer
    ├── intelligent_router.py     # +EnterpriseRouter, CircuitBreaker, ABTestManager
    ├── model_selector.py         # +ModelHealthChecker, ModelPool, ConnectionManager
    ├── cot_parser.py             # +StreamingCoTParser, MultiFormatParser, ParseMetrics
    ├── self_correction_engine.py # +CorrectionStrategy, Observer, Metrics
    ├── smart_fallback.py         # +EnterpriseFallbackManager, CircuitBreaker, Analytics
    ├── user_preferences.py       # +PreferenceProfile, Migrator, MultiTenant
    ├── paper_metrics.py          # +PrometheusExporter, Dashboard, RealTimeTelemetry
    └── query_orchestrator.py     # +Middleware, ExecutionHook, OrchestratorMetrics
```

---

## Migration Guide

### From v1.x to v2.0

**Zero changes required.** All existing code continues to work.

To optionally adopt enterprise features:

1. Replace `IntelligentRouter` → `EnterpriseRouter` for circuit breakers
2. Replace `SmartFallbackManager` → `EnterpriseFallbackManager` for analytics
3. Replace `EnhancedSandbox` → `EnterpriseSandbox` for compliance logging
4. Add `ValidationPipeline` alongside existing `RuntimeEvaluator`
5. Add `PrometheusExporter` for monitoring integration
6. Add `MultiTenantPreferences` for multi-user deployments
