"""Prometheus-Compatible Metrics Facade — Nexus LLM Analytics v2.0
=================================================================

Provides a thin abstraction over ``prometheus_client`` so that the
``/metrics`` and ``/metrics/json`` endpoints work regardless of whether
the Prometheus client library is installed.

Exports
-------
``METRICS``
    Module-level metrics registry instance.
``generate_metrics_output()``
    Render all collected metrics in Prometheus text exposition format.
``get_metrics_content_type()``
    Return the MIME type expected by Prometheus scrapers.

When ``prometheus_client`` is available, real Counter / Histogram /
Gauge objects are used.  Otherwise a lightweight :class:`FallbackMetrics`
stand-in accumulates counters in-process for the ``/metrics/json``
endpoint while returning an empty string for text-format scraping.

Backward Compatibility
----------------------
All three public symbols (``METRICS``, ``generate_metrics_output``,
``get_metrics_content_type``) are stable and must not be removed.

Author: Nexus Analytics Research Team
Since: v1.5 (refactored v2.0 — February 2026)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import Prometheus client; fall back gracefully
# ---------------------------------------------------------------------------
_HAS_PROMETHEUS = False
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _HAS_PROMETHEUS = True
except ImportError:
    logger.info("prometheus_client not installed — using fallback metrics")


# ---------------------------------------------------------------------------
# Fallback metrics implementation (no external deps)
# ---------------------------------------------------------------------------

class FallbackMetrics:
    """Lightweight in-process metrics when ``prometheus_client`` is absent.

    Tracks counters and gauges in plain dicts.  Sufficient for the
    ``/metrics/json`` endpoint but produces an empty string for Prometheus
    text scraping (since no scraper is configured anyway).

    Thread-safe via a reentrant lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = {
            "requests_total": 0,
            "analyses_total": 0,
            "analyses_success": 0,
            "analyses_failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "llm_requests_total": 0,
            "llm_errors_total": 0,
        }
        self._gauges: Dict[str, float] = {
            "active_analyses": 0,
            "uptime_seconds": 0,
        }
        self._start_time = time.monotonic()

    # -- Counter helpers -------------------------------------------------- #

    def inc(self, name: str, amount: float = 1) -> None:
        """Increment counter *name* by *amount* (default 1)."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def set_gauge(self, name: str, value: float) -> None:
        """Set gauge *name* to *value*."""
        with self._lock:
            self._gauges[name] = value

    # -- Reporting -------------------------------------------------------- #

    def get_fallback_stats(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of all counters and gauges."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": {
                    **self._gauges,
                    "uptime_seconds": round(time.monotonic() - self._start_time, 1),
                },
            }

    def generate_text(self) -> str:
        """Render metrics in Prometheus text exposition format.

        When ``prometheus_client`` is not installed this returns a minimal
        comment block so Prometheus scrapers see a valid (but empty) page.
        """
        lines = ["# Nexus LLM Analytics — fallback metrics (prometheus_client not installed)"]
        with self._lock:
            for name, value in self._counters.items():
                lines.append(f"# TYPE nexus_{name} counter")
                lines.append(f"nexus_{name} {value}")
            for name, value in self._gauges.items():
                lines.append(f"# TYPE nexus_{name} gauge")
                lines.append(f"nexus_{name} {value}")
            lines.append(
                f"# TYPE nexus_uptime_seconds gauge\n"
                f"nexus_uptime_seconds {round(time.monotonic() - self._start_time, 1)}"
            )
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Prometheus-backed metrics (when the library is available)
# ---------------------------------------------------------------------------

class PrometheusMetrics:
    """Thin wrapper around ``prometheus_client`` objects.

    Provides the same ``.inc()`` / ``.set_gauge()`` / ``.get_fallback_stats()``
    API as :class:`FallbackMetrics` so callers are library-agnostic.
    """

    def __init__(self) -> None:
        self.requests_total = Counter(
            "nexus_requests_total",
            "Total HTTP requests handled",
        )
        self.analyses_total = Counter(
            "nexus_analyses_total",
            "Total analysis jobs started",
        )
        self.analyses_success = Counter(
            "nexus_analyses_success",
            "Analyses that completed successfully",
        )
        self.analyses_failed = Counter(
            "nexus_analyses_failed",
            "Analyses that ended in error",
        )
        self.cache_hits = Counter("nexus_cache_hits", "Cache hit count")
        self.cache_misses = Counter("nexus_cache_misses", "Cache miss count")
        self.llm_requests_total = Counter(
            "nexus_llm_requests_total",
            "Total LLM API calls",
        )
        self.llm_errors_total = Counter(
            "nexus_llm_errors_total",
            "LLM API errors",
        )
        self.active_analyses = Gauge(
            "nexus_active_analyses",
            "Currently running analyses",
        )
        self.analysis_duration = Histogram(
            "nexus_analysis_duration_seconds",
            "Time spent per analysis",
            buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300),
        )

    # Convenience helpers matching FallbackMetrics API
    def inc(self, name: str, amount: float = 1) -> None:
        counter = getattr(self, name, None)
        if counter is not None and hasattr(counter, "inc"):
            counter.inc(amount)

    def set_gauge(self, name: str, value: float) -> None:
        gauge = getattr(self, name, None)
        if gauge is not None and hasattr(gauge, "set"):
            gauge.set(value)

    def get_fallback_stats(self) -> Dict[str, Any]:
        """Return a snapshot using prometheus_client's internal values."""
        return {
            "backend": "prometheus_client",
            "note": "Use /metrics endpoint for full Prometheus text format",
        }


# ---------------------------------------------------------------------------
# Module-level singleton and public helpers
# ---------------------------------------------------------------------------

METRICS: Any = PrometheusMetrics() if _HAS_PROMETHEUS else FallbackMetrics()
"""Module-level metrics singleton — use ``METRICS.inc('counter_name')``."""


def generate_metrics_output() -> str:
    """Return metrics in Prometheus text exposition format.

    Uses ``prometheus_client.generate_latest()`` when available;
    otherwise falls back to :meth:`FallbackMetrics.generate_text`.
    """
    if _HAS_PROMETHEUS:
        return generate_latest().decode("utf-8")
    return METRICS.generate_text()


def get_metrics_content_type() -> str:
    """Return the MIME type for Prometheus scraping.

    ``text/plain; version=0.0.4; charset=utf-8`` when the real library
    is installed, otherwise plain ``text/plain``.
    """
    if _HAS_PROMETHEUS:
        return CONTENT_TYPE_LATEST
    return "text/plain; charset=utf-8"
