"""
Engine Components
=================
Core orchestration and routing components for Nexus LLM Analytics.

This package contains the central decision-making pipeline:

.. code-block:: text

    Query → IntelligentRouter (complexity) → QueryOrchestrator (plan)
          → SelfCorrectionEngine (validate) → Response

.. versionadded:: 2.0.0
   Enterprise-grade extensions added to every submodule while
   preserving full backward compatibility with v1.x import paths.

Submodules
----------
intelligent_router
    Complexity-based query routing to model tiers.
    **v2.0**: CircuitBreaker, RoutingPolicy, ABTestManager, EnterpriseRouter.
query_orchestrator
    Master decision maker: complexity → model, method, review.
self_correction_engine
    Generator → Critic → Feedback self-correction loop.
    **v2.0**: CorrectionStrategy, CorrectionObserver, CorrectionMetrics.
model_selector
    Unified model discovery, RAM-aware selection, smart fallback.
    **v2.0**: ModelHealthChecker, ModelPool, ConnectionManager.
smart_fallback
    Centralized fallback chain management.
    **v2.0**: FallbackCircuitBreaker, FallbackAnalytics, EnterpriseFallbackManager.
cot_parser
    Chain-of-Thought and Critic response parsers.
    **v2.0**: StreamingCoTParser, MultiFormatParser, ParseMetrics.
automated_validation
    Runtime hallucination / statistical-claim validation.
    **v2.0**: ValidationPipeline, ValidatorRegistry, ConfidenceScorer.
paper_metrics
    Research-paper metrics collector (ICMLAS tables).
    **v2.0**: PrometheusExporter, MetricsDashboard, RealTimeTelemetry.
user_preferences
    Persistent user preference management.
    **v2.0**: PreferenceProfile, PreferenceMigrator, MultiTenantPreferences.
"""

from __future__ import annotations

__all__ = [
    # ── intelligent_router (v1.x) ──
    "IntelligentRouter",
    "ModelTier",
    "RoutingDecision",
    "get_intelligent_router",
    # ── intelligent_router (v2.0) ──
    "RouterCircuitBreaker",
    "RouterCircuitState",
    "RoutingPolicy",
    "RoutingPolicyBuilder",
    "RouterHealthCheck",
    "ABTestManager",
    "EnterpriseRouter",
    "get_enterprise_router",
    # ── query_orchestrator (v1.x) ──
    "QueryOrchestrator",
    "ExecutionMethod",
    "ExecutionPlan",
    "ReviewLevel",
    "get_query_orchestrator",
    # ── query_orchestrator (v2.0) ──
    "OrchestratorMiddleware",
    "ExecutionHook",
    "OrchestratorMetrics",
    "get_orchestrator_metrics",
    # ── self_correction_engine (v1.x) ──
    "SelfCorrectionEngine",
    "CorrectionIteration",
    "CorrectionResult",
    # ── self_correction_engine (v2.0) ──
    "CorrectionStrategy",
    "CorrectionObserver",
    "CorrectionMetrics",
    "get_correction_metrics",
    # ── model_selector (v1.x) ──
    "ModelSelector",
    "DynamicModelDiscovery",
    "RAMAwareSelector",
    "ModelCapability",
    "ModelInfo",
    "MemoryPressureLevel",
    "MemorySnapshot",
    "ModelSelectionResult",
    "get_model_discovery",
    "get_ram_selector",
    # ── model_selector (v2.0) ──
    "ModelHealthChecker",
    "ModelHealthStatus",
    "ModelPool",
    "ConnectionManager",
    "get_model_health_checker",
    "get_connection_manager",
    # ── smart_fallback (v1.x) ──
    "SmartFallbackManager",
    "FallbackReason",
    "FallbackEvent",
    "FallbackChain",
    "GracefulDegradation",
    "get_fallback_manager",
    # ── smart_fallback (v2.0) ──
    "FallbackCircuitBreaker",
    "FallbackAnalytics",
    "FallbackPolicy",
    "EnterpriseFallbackManager",
    "get_enterprise_fallback_manager",
    # ── cot_parser (v1.x) ──
    "CoTParser",
    "CriticParser",
    "ParsedCoT",
    "CriticFeedback",
    "CriticIssue",
    # ── cot_parser (v2.0) ──
    "StreamingCoTParser",
    "MultiFormatParser",
    "ParseMetrics",
    "get_parse_metrics",
    # ── automated_validation (v1.x) ──
    "RuntimeEvaluator",
    "ValidationResult",
    "ValidationIssue",
    # ── automated_validation (v2.0) ──
    "ValidationPipeline",
    "ValidatorRegistry",
    "BaseValidator",
    "ConfidenceScorer",
    "ValidationSeverity",
    "ValidationContext",
    "batch_validate",
    "get_validator_registry",
    # ── paper_metrics (v1.x) ──
    "PaperMetricsCollector",
    "get_paper_metrics",
    # ── paper_metrics (v2.0) ──
    "PrometheusExporter",
    "MetricsDashboard",
    "RealTimeTelemetry",
    "get_prometheus_exporter",
    # ── user_preferences (v1.x) ──
    "UserPreferences",
    "UserPreferencesManager",
    "get_preferences_manager",
    # ── user_preferences (v2.0) ──
    "PreferenceVersion",
    "PreferenceMigrator",
    "PreferenceProfile",
    "MultiTenantPreferences",
    "get_multi_tenant_preferences",
]

# ---------------------------------------------------------------------------
# Lazy re-exports — mirrors __all__ so ``from backend.core.engine import X``
# works for every public name.
# ---------------------------------------------------------------------------

from .intelligent_router import (  # noqa: E402
    IntelligentRouter,
    ModelTier,
    RoutingDecision,
    get_intelligent_router,
    # v2.0
    RouterCircuitBreaker,
    RouterCircuitState,
    RoutingPolicy,
    RoutingPolicyBuilder,
    RouterHealthCheck,
    ABTestManager,
    EnterpriseRouter,
    get_enterprise_router,
)
from .query_orchestrator import (  # noqa: E402
    ExecutionMethod,
    ExecutionPlan,
    QueryOrchestrator,
    ReviewLevel,
    get_query_orchestrator,
    # v2.0
    OrchestratorMiddleware,
    ExecutionHook,
    OrchestratorMetrics,
    get_orchestrator_metrics,
)
from .self_correction_engine import (  # noqa: E402
    CorrectionIteration,
    CorrectionResult,
    SelfCorrectionEngine,
    # v2.0
    CorrectionStrategy,
    CorrectionObserver,
    CorrectionMetrics,
    get_correction_metrics,
)
from .model_selector import (  # noqa: E402
    DynamicModelDiscovery,
    MemoryPressureLevel,
    MemorySnapshot,
    ModelCapability,
    ModelInfo,
    ModelSelectionResult,
    ModelSelector,
    RAMAwareSelector,
    get_model_discovery,
    get_ram_selector,
    # v2.0
    ModelHealthChecker,
    ModelHealthStatus,
    ModelPool,
    ConnectionManager,
    get_model_health_checker,
    get_connection_manager,
)
from .smart_fallback import (  # noqa: E402
    FallbackChain,
    FallbackEvent,
    FallbackReason,
    GracefulDegradation,
    SmartFallbackManager,
    get_fallback_manager,
    # v2.0
    FallbackCircuitBreaker,
    FallbackAnalytics,
    FallbackPolicy,
    EnterpriseFallbackManager,
    get_enterprise_fallback_manager,
)
from .cot_parser import (  # noqa: E402
    CoTParser,
    CriticFeedback,
    CriticIssue,
    CriticParser,
    ParsedCoT,
    # v2.0
    StreamingCoTParser,
    MultiFormatParser,
    ParseMetrics,
    get_parse_metrics,
)
from .automated_validation import (  # noqa: E402
    RuntimeEvaluator,
    ValidationIssue,
    ValidationResult,
    # v2.0
    ValidationPipeline,
    ValidatorRegistry,
    BaseValidator,
    ConfidenceScorer,
    ValidationSeverity,
    ValidationContext,
    batch_validate,
    get_validator_registry,
)
from .paper_metrics import (  # noqa: E402
    PaperMetricsCollector,
    get_paper_metrics,
    # v2.0
    PrometheusExporter,
    MetricsDashboard,
    RealTimeTelemetry,
    get_prometheus_exporter,
)
from .user_preferences import (  # noqa: E402
    UserPreferences,
    UserPreferencesManager,
    get_preferences_manager,
    # v2.0
    PreferenceVersion,
    PreferenceMigrator,
    PreferenceProfile,
    MultiTenantPreferences,
    get_multi_tenant_preferences,
)
