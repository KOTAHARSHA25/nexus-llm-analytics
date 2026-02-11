"""
Centralized Metrics Collection System for Nexus LLM Analytics
==============================================================

Captures routing decisions, latencies, error correction outcomes,
model selections, and agent performance for research evaluation.

.. versionadded:: 2.0.0
   Added :class:`PrometheusExporter`, :class:`MetricsDashboard`,
   :class:`RealTimeTelemetry`, and :func:`get_prometheus_exporter`.

Addresses ICMLAS reviewer comments:
- R2: Impact of orchestration heuristics on latency and accuracy
- R3: Error correction mechanisms under diverse failure scenarios
- R4: Learning-based routing efficiency metrics
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    # v1.x (backward compatible)
    "PaperMetricsCollector",
    "RoutingDecision",
    "CorrectionOutcome",
    "ErrorRecoveryEvent",
    "AgentPerformance",
    "get_paper_metrics",
    # v2.0 Enterprise additions
    "PrometheusExporter",
    "MetricsDashboard",
    "RealTimeTelemetry",
    "get_prometheus_exporter",
]


@dataclass
class RoutingDecision:
    """Single routing decision capturing method, model, latency, and outcome."""

    timestamp: str
    query: str
    query_length: int
    complexity_score: float
    routing_method: str  # 'semantic' or 'heuristic_fallback'
    semantic_routing_success: bool
    selected_model: str
    execution_method: str  # 'code_generation', 'direct_llm'
    review_level: str  # 'none', 'optional', 'mandatory'
    routing_latency_ms: float  # Time to make routing decision
    total_latency_ms: float = 0.0  # Total query processing time
    agent_used: str = ""
    success: bool = False
    error_type: str = ""
    fallback_triggered: bool = False
    user_override: bool = False


@dataclass
class CorrectionOutcome:
    """Outcome of one self-correction loop including iterations, timing, and learning."""

    timestamp: str
    query: str
    iterations: int
    termination_reason: str  # 'validated', 'max_iterations', 'failure', 'parsing_failure'
    total_time_seconds: float
    automated_issues_found: int
    automated_issue_types: List[str]
    critic_invoked: bool
    critic_approved: bool
    safety_issues_found: int
    confidence_score: float
    learned_pattern_stored: bool
    learned_pattern_retrieved: bool
    correction_improved_output: bool


@dataclass
class ErrorRecoveryEvent:
    """Record of a single error-recovery activation with mechanism, outcome, and overhead."""

    timestamp: str
    mechanism: str  # 'circuit_breaker', 'model_fallback', 'timeout_adaptation', 'json_repair', 'resource_downgrade', 'automated_validation', 'self_correction', 'safety_validation'
    trigger: str  # What caused the recovery
    outcome: str  # 'recovered', 'degraded', 'failed'
    latency_overhead_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformance:
    """Accumulated performance counters and averages for a single specialist agent."""

    agent_name: str
    query_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    code_gen_count: int = 0
    code_exec_success: int = 0
    avg_confidence: float = 0.0


class PaperMetricsCollector:
    """
    Thread-safe centralized metrics collector.
    Stores all metrics in memory and persists to JSON for paper analysis.
    """
    
    def __init__(self, output_dir: str = "data/paper_metrics") -> None:
        """Initialize the metrics collector.

        Args:
            output_dir: Directory path for persisted JSON metric files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        
        # Core metrics stores
        self.routing_decisions: List[RoutingDecision] = []
        self.correction_outcomes: List[CorrectionOutcome] = []
        self.error_recovery_events: List[ErrorRecoveryEvent] = []
        self.agent_performance: Dict[str, AgentPerformance] = {}
        
        # Counters
        self._query_count = 0
        self._semantic_routing_count = 0
        self._heuristic_fallback_count = 0
        self._correction_loop_count = 0
        self._pattern_store_count = 0
        self._pattern_retrieve_count = 0
        
        # Session metadata
        self._session_start = datetime.now().isoformat()
        self._session_id = f"session_{int(time.time())}"
        
        logger.info("PaperMetricsCollector initialized (session: %s)", self._session_id)
    
    # ==========================================
    # ROUTING METRICS (Reviewer Comment 2 & 4) 
    # ==========================================
    
    def record_routing_decision(self, decision: RoutingDecision) -> None:
        """Record a routing decision with all metadata.

        Args:
            decision: Populated :class:`RoutingDecision` to store.
        """
        with self._lock:
            self.routing_decisions.append(decision)
            self._query_count += 1
            if decision.routing_method == 'semantic':
                self._semantic_routing_count += 1
            else:
                self._heuristic_fallback_count += 1
    
    def record_routing(self, query: str, complexity: float, method: str,
                       semantic_success: bool, model: str, exec_method: str,
                       review_level: str, routing_latency_ms: float,
                       **kwargs: Any) -> RoutingDecision:
        """Build and record a routing decision from individual parameters.

        Args:
            query: Raw user query text (truncated to 200 chars internally).
            complexity: Numeric complexity score assigned by the router.
            method: Routing method used (``'semantic'`` or ``'heuristic_fallback'``).
            semantic_success: Whether semantic routing resolved successfully.
            model: Name of the selected LLM model.
            exec_method: Execution strategy (``'code_generation'``, ``'direct_llm'``).
            review_level: Review tier (``'none'``, ``'optional'``, ``'mandatory'``).
            routing_latency_ms: Time in ms spent making the routing decision.
            **kwargs: Additional fields forwarded to :class:`RoutingDecision`.

        Returns:
            The newly created :class:`RoutingDecision` instance.
        """
        decision = RoutingDecision(
            timestamp=datetime.now().isoformat(),
            query=query[:200],
            query_length=len(query),
            complexity_score=complexity,
            routing_method=method,
            semantic_routing_success=semantic_success,
            selected_model=model,
            execution_method=exec_method,
            review_level=review_level,
            routing_latency_ms=routing_latency_ms,
            **kwargs
        )
        self.record_routing_decision(decision)
        return decision
    
    # ==========================================
    # ERROR CORRECTION METRICS (Reviewer Comment 3)
    # ==========================================
    
    def record_correction_outcome(self, outcome: CorrectionOutcome) -> None:
        """Record a self-correction loop outcome.

        Args:
            outcome: Populated :class:`CorrectionOutcome` to store.
        """
        with self._lock:
            self.correction_outcomes.append(outcome)
            self._correction_loop_count += 1
            if outcome.learned_pattern_stored:
                self._pattern_store_count += 1
            if outcome.learned_pattern_retrieved:
                self._pattern_retrieve_count += 1
    
    def record_error_recovery(self, mechanism: str, trigger: str,
                              outcome: str, latency_ms: float,
                              **details: Any) -> None:
        """Record any error recovery event.

        Args:
            mechanism: Recovery strategy (e.g. ``'circuit_breaker'``).
            trigger: Description of what caused the recovery.
            outcome: Result — ``'recovered'``, ``'degraded'``, or ``'failed'``.
            latency_ms: Overhead in milliseconds added by recovery.
            **details: Arbitrary extra context stored on the event.
        """
        event = ErrorRecoveryEvent(
            timestamp=datetime.now().isoformat(),
            mechanism=mechanism,
            trigger=trigger,
            outcome=outcome,
            latency_overhead_ms=latency_ms,
            details=details
        )
        with self._lock:
            self.error_recovery_events.append(event)
    
    # ==========================================
    # AGENT PERFORMANCE METRICS
    # ==========================================
    
    def record_agent_execution(self, agent_name: str, success: bool,
                               latency_ms: float, code_gen: bool = False,
                               code_exec_success: bool = False,
                               confidence: float = 0.0) -> None:
        """Record agent execution performance.

        Args:
            agent_name: Identifier of the specialist agent.
            success: Whether the agent completed successfully.
            latency_ms: Wall-clock execution time in milliseconds.
            code_gen: Whether code generation was used.
            code_exec_success: Whether generated code executed without error.
            confidence: Agent-reported confidence score (0-1).
        """
        with self._lock:
            if agent_name not in self.agent_performance:
                self.agent_performance[agent_name] = AgentPerformance(agent_name=agent_name)
            
            perf = self.agent_performance[agent_name]
            perf.query_count += 1
            if success:
                perf.success_count += 1
            else:
                perf.failure_count += 1
            perf.total_latency_ms += latency_ms
            if code_gen:
                perf.code_gen_count += 1
            if code_exec_success:
                perf.code_exec_success += 1
            # Running average for confidence
            if perf.query_count > 0:
                perf.avg_confidence = (
                    (perf.avg_confidence * (perf.query_count - 1) + confidence) / perf.query_count
                )
    
    # ==========================================
    # AGGREGATION & PAPER-READY STATISTICS
    # ==========================================
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get aggregated routing statistics for paper Table 1.

        Returns:
            Dict with method distribution, latencies, model/complexity
            stats, success rates, and fallback trigger rate.
        """
        with self._lock:
            if not self.routing_decisions:
                return {"error": "No routing data collected"}
            
            decisions = self.routing_decisions
            total = len(decisions)
            
            # Routing method distribution
            semantic_count = sum(1 for d in decisions if d.routing_method == 'semantic')
            heuristic_count = total - semantic_count
            
            # Semantic success rate
            semantic_attempts = [d for d in decisions if d.routing_method == 'semantic']
            semantic_success_rate = (
                sum(1 for d in semantic_attempts if d.semantic_routing_success) / len(semantic_attempts)
                if semantic_attempts else 0.0
            )
            
            # Latency statistics
            routing_latencies = [d.routing_latency_ms for d in decisions]
            total_latencies = [d.total_latency_ms for d in decisions if d.total_latency_ms > 0]
            
            # Model distribution
            model_counts = defaultdict(int)
            for d in decisions:
                model_counts[d.selected_model] += 1
            
            # Execution method distribution  
            method_counts = defaultdict(int)
            for d in decisions:
                method_counts[d.execution_method] += 1
            
            # Complexity distribution
            complexities = [d.complexity_score for d in decisions]
            
            # Success rate by routing method
            semantic_success = [d for d in decisions if d.routing_method == 'semantic' and d.success]
            heuristic_success = [d for d in decisions if d.routing_method != 'semantic' and d.success]
            
            return {
                "total_queries": total,
                "routing_method_distribution": {
                    "semantic": semantic_count,
                    "heuristic_fallback": heuristic_count,
                    "semantic_percentage": round(semantic_count / total * 100, 1) if total > 0 else 0
                },
                "semantic_routing_success_rate": round(semantic_success_rate * 100, 1),
                "routing_latency_ms": {
                    "mean": round(sum(routing_latencies) / len(routing_latencies), 2) if routing_latencies else 0,
                    "min": round(min(routing_latencies), 2) if routing_latencies else 0,
                    "max": round(max(routing_latencies), 2) if routing_latencies else 0,
                    "std": round(_std(routing_latencies), 2) if len(routing_latencies) > 1 else 0
                },
                "total_latency_ms": {
                    "mean": round(sum(total_latencies) / len(total_latencies), 2) if total_latencies else 0,
                    "min": round(min(total_latencies), 2) if total_latencies else 0,
                    "max": round(max(total_latencies), 2) if total_latencies else 0,
                    "std": round(_std(total_latencies), 2) if len(total_latencies) > 1 else 0
                },
                "model_distribution": dict(model_counts),
                "execution_method_distribution": dict(method_counts),
                "complexity_score": {
                    "mean": round(sum(complexities) / len(complexities), 3) if complexities else 0,
                    "min": round(min(complexities), 3) if complexities else 0,
                    "max": round(max(complexities), 3) if complexities else 0,
                },
                "overall_success_rate": round(
                    sum(1 for d in decisions if d.success) / total * 100, 1
                ) if total > 0 else 0,
                "success_by_routing_method": {
                    "semantic": round(
                        len(semantic_success) / semantic_count * 100, 1
                    ) if semantic_count > 0 else 0,
                    "heuristic": round(
                        len(heuristic_success) / heuristic_count * 100, 1
                    ) if heuristic_count > 0 else 0,
                },
                "fallback_trigger_rate": round(
                    sum(1 for d in decisions if d.fallback_triggered) / total * 100, 1
                ) if total > 0 else 0
            }
    
    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get aggregated correction statistics for paper Table 2.

        Returns:
            Dict with termination reasons, iteration/time stats,
            automated-validation counts, critic rates, and learning metrics.
        """
        with self._lock:
            if not self.correction_outcomes:
                return {"error": "No correction data collected"}
            
            outcomes = self.correction_outcomes
            total = len(outcomes)
            
            # Termination reasons
            reasons = defaultdict(int)
            for o in outcomes:
                reasons[o.termination_reason] += 1
            
            # Iteration statistics
            iterations = [o.iterations for o in outcomes]
            
            # Time statistics
            times = [o.total_time_seconds for o in outcomes]
            
            # Automated validation
            auto_issues = [o.automated_issues_found for o in outcomes]
            
            # Critic statistics
            critic_invoked = sum(1 for o in outcomes if o.critic_invoked)
            critic_approved = sum(1 for o in outcomes if o.critic_approved)
            
            # Learning statistics
            patterns_stored = sum(1 for o in outcomes if o.learned_pattern_stored)
            patterns_retrieved = sum(1 for o in outcomes if o.learned_pattern_retrieved)
            
            return {
                "total_correction_loops": total,
                "termination_reasons": dict(reasons),
                "validation_success_rate": round(
                    reasons.get('validated', 0) / total * 100, 1
                ) if total > 0 else 0,
                "iterations": {
                    "mean": round(sum(iterations) / len(iterations), 2) if iterations else 0,
                    "min": min(iterations) if iterations else 0,
                    "max": max(iterations) if iterations else 0
                },
                "time_seconds": {
                    "mean": round(sum(times) / len(times), 2) if times else 0,
                    "min": round(min(times), 2) if times else 0,
                    "max": round(max(times), 2) if times else 0
                },
                "automated_validation": {
                    "queries_with_issues": sum(1 for a in auto_issues if a > 0),
                    "total_issues_found": sum(auto_issues),
                    "mean_issues_per_query": round(sum(auto_issues) / len(auto_issues), 2) if auto_issues else 0
                },
                "critic_statistics": {
                    "invoked_count": critic_invoked,
                    "approved_count": critic_approved,
                    "approval_rate": round(
                        critic_approved / critic_invoked * 100, 1
                    ) if critic_invoked > 0 else 0
                },
                "self_learning": {
                    "patterns_stored": patterns_stored,
                    "patterns_retrieved": patterns_retrieved,
                    "retrieval_rate": round(
                        patterns_retrieved / total * 100, 1
                    ) if total > 0 else 0
                },
                "confidence_scores": {
                    "mean": round(
                        sum(o.confidence_score for o in outcomes) / total, 3
                    ) if total > 0 else 0
                }
            }
    
    def get_error_recovery_statistics(self) -> Dict[str, Any]:
        """Get aggregated error recovery statistics for paper Table 3.

        Returns:
            Dict with total events, overall recovery rate,
            per-mechanism breakdown, and cumulative latency overhead.
        """
        with self._lock:
            if not self.error_recovery_events:
                return {"error": "No error recovery events collected"}
            
            events = self.error_recovery_events
            total = len(events)
            
            # By mechanism
            by_mechanism = defaultdict(lambda: {"total": 0, "recovered": 0, "degraded": 0, "failed": 0, "latencies": []})
            for e in events:
                m = by_mechanism[e.mechanism]
                m["total"] += 1
                m[e.outcome] += 1
                m["latencies"].append(e.latency_overhead_ms)
            
            mechanism_stats = {}
            for mech, data in by_mechanism.items():
                mechanism_stats[mech] = {
                    "count": data["total"],
                    "recovery_rate": round(data["recovered"] / data["total"] * 100, 1) if data["total"] > 0 else 0,
                    "degraded_rate": round(data["degraded"] / data["total"] * 100, 1) if data["total"] > 0 else 0,
                    "failure_rate": round(data["failed"] / data["total"] * 100, 1) if data["total"] > 0 else 0,
                    "mean_latency_ms": round(sum(data["latencies"]) / len(data["latencies"]), 2) if data["latencies"] else 0
                }
            
            return {
                "total_recovery_events": total,
                "overall_recovery_rate": round(
                    sum(1 for e in events if e.outcome == 'recovered') / total * 100, 1
                ) if total > 0 else 0,
                "by_mechanism": mechanism_stats,
                "mechanisms_triggered": list(by_mechanism.keys()),
                "total_latency_overhead_ms": round(
                    sum(e.latency_overhead_ms for e in events), 2
                )
            }
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get per-agent performance statistics.

        Returns:
            Dict keyed by agent name with success rate, latency,
            code-gen rate, execution success rate, and confidence.
        """
        with self._lock:
            stats = {}
            for name, perf in self.agent_performance.items():
                stats[name] = {
                    "queries": perf.query_count,
                    "success_rate": round(
                        perf.success_count / perf.query_count * 100, 1
                    ) if perf.query_count > 0 else 0,
                    "avg_latency_ms": round(
                        perf.total_latency_ms / perf.query_count, 2
                    ) if perf.query_count > 0 else 0,
                    "code_gen_rate": round(
                        perf.code_gen_count / perf.query_count * 100, 1
                    ) if perf.query_count > 0 else 0,
                    "code_exec_success_rate": round(
                        perf.code_exec_success / perf.code_gen_count * 100, 1
                    ) if perf.code_gen_count > 0 else 0,
                    "avg_confidence": round(perf.avg_confidence, 3)
                }
            return stats
    
    # ==========================================
    # PERSISTENCE & EXPORT
    # ==========================================
    
    def save_to_file(self, filename: Optional[str] = None) -> str:
        """Persist all collected metrics to a JSON file.

        Args:
            filename: Target file name inside *output_dir*.  Defaults to
                ``paper_metrics_<session_id>.json``.

        Returns:
            Absolute path to the written JSON file.
        """
        if filename is None:
            filename = f"paper_metrics_{self._session_id}.json"
        
        filepath = self.output_dir / filename
        
        report = {
            "session_id": self._session_id,
            "session_start": self._session_start,
            "session_end": datetime.now().isoformat(),
            "summary": {
                "total_queries": self._query_count,
                "semantic_routing_calls": self._semantic_routing_count,
                "heuristic_fallback_calls": self._heuristic_fallback_count,
                "correction_loops_run": self._correction_loop_count,
                "patterns_stored": self._pattern_store_count,
                "patterns_retrieved": self._pattern_retrieve_count,
            },
            "routing_statistics": self.get_routing_statistics(),
            "correction_statistics": self.get_correction_statistics(),
            "error_recovery_statistics": self.get_error_recovery_statistics(),
            "agent_statistics": self.get_agent_statistics(),
            "raw_data": {
                "routing_decisions": [asdict(d) for d in self.routing_decisions],
                "correction_outcomes": [asdict(o) for o in self.correction_outcomes],
                "error_recovery_events": [asdict(e) for e in self.error_recovery_events],
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Paper metrics saved to %s", filepath)
        return str(filepath)
    
    def generate_latex_tables(self) -> str:
        """Generate LaTeX tables from collected metrics for IEEE paper.

        Returns:
            Multi-table LaTeX string (Tables 1-4) ready for inclusion.
        """
        routing = self.get_routing_statistics()
        correction = self.get_correction_statistics()
        recovery = self.get_error_recovery_statistics()
        agents = self.get_agent_statistics()
        
        latex = []
        
        # Table 1: Routing Performance
        latex.append(r"""
\begin{table}[htbp]
\centering
\caption{Query Routing Performance Analysis}
\label{tab:routing}
\begin{tabular}{lc}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline""")
        
        if isinstance(routing, dict) and "total_queries" in routing:
            latex.append(f"Total Queries Evaluated & {routing['total_queries']} \\\\")
            latex.append(f"Semantic Routing Rate & {routing['routing_method_distribution']['semantic_percentage']}\\% \\\\")
            latex.append(f"Semantic Success Rate & {routing['semantic_routing_success_rate']}\\% \\\\")
            latex.append(f"Mean Routing Latency (ms) & {routing['routing_latency_ms']['mean']} \\\\")
            latex.append(f"Mean Total Latency (ms) & {routing['total_latency_ms']['mean']} \\\\")
            latex.append(f"Overall Query Success Rate & {routing['overall_success_rate']}\\% \\\\")
            latex.append(f"Fallback Trigger Rate & {routing['fallback_trigger_rate']}\\% \\\\")
        
        latex.append(r"""\hline
\end{tabular}
\end{table}""")
        
        # Table 2: Error Correction Performance
        latex.append(r"""
\begin{table}[htbp]
\centering
\caption{Self-Correction Engine Performance}
\label{tab:correction}
\begin{tabular}{lc}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline""")
        
        if isinstance(correction, dict) and "total_correction_loops" in correction:
            latex.append(f"Total Correction Loops & {correction['total_correction_loops']} \\\\")
            latex.append(f"Validation Success Rate & {correction['validation_success_rate']}\\% \\\\")
            latex.append(f"Mean Iterations & {correction['iterations']['mean']} \\\\")
            latex.append(f"Mean Time (seconds) & {correction['time_seconds']['mean']} \\\\")
            latex.append(f"Automated Issues Found & {correction['automated_validation']['total_issues_found']} \\\\")
            latex.append(f"Critic Approval Rate & {correction['critic_statistics']['approval_rate']}\\% \\\\")
            latex.append(f"Patterns Stored & {correction['self_learning']['patterns_stored']} \\\\")
            latex.append(f"Pattern Retrieval Rate & {correction['self_learning']['retrieval_rate']}\\% \\\\")
        
        latex.append(r"""\hline
\end{tabular}
\end{table}""")
        
        # Table 3: Error Recovery Mechanisms
        latex.append(r"""
\begin{table}[htbp]
\centering
\caption{Error Recovery Mechanism Evaluation}
\label{tab:recovery}
\begin{tabular}{lccc}
\hline
\textbf{Mechanism} & \textbf{Count} & \textbf{Recovery Rate} & \textbf{Latency (ms)} \\
\hline""")
        
        if isinstance(recovery, dict) and "by_mechanism" in recovery:
            for mech, stats in recovery["by_mechanism"].items():
                name = mech.replace('_', ' ').title()
                latex.append(f"{name} & {stats['count']} & {stats['recovery_rate']}\\% & {stats['mean_latency_ms']} \\\\")
        
        latex.append(r"""\hline
\end{tabular}
\end{table}""")
        
        # Table 4: Agent Performance Comparison
        latex.append(r"""
\begin{table}[htbp]
\centering
\caption{Specialist Agent Performance Comparison}
\label{tab:agents}
\begin{tabular}{lcccc}
\hline
\textbf{Agent} & \textbf{Queries} & \textbf{Success} & \textbf{Latency (ms)} & \textbf{Code Gen} \\
\hline""")
        
        if isinstance(agents, dict):
            for name, stats in agents.items():
                latex.append(
                    f"{name} & {stats['queries']} & {stats['success_rate']}\\% "
                    f"& {stats['avg_latency_ms']} & {stats['code_gen_rate']}\\% \\\\"
                )
        
        latex.append(r"""\hline
\end{tabular}
\end{table}""")
        
        return "\n".join(latex)
    
    def reset(self) -> None:
        """Reset all metrics for a fresh benchmark run."""
        with self._lock:
            self.routing_decisions.clear()
            self.correction_outcomes.clear()
            self.error_recovery_events.clear()
            self.agent_performance.clear()
            self._query_count = 0
            self._semantic_routing_count = 0
            self._heuristic_fallback_count = 0
            self._correction_loop_count = 0
            self._pattern_store_count = 0
            self._pattern_retrieve_count = 0
            self._session_start = datetime.now().isoformat()
            self._session_id = f"session_{int(time.time())}"


def _std(values: List[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


# Thread-safe singleton
_metrics_instance = None
_metrics_lock = threading.Lock()

def get_paper_metrics() -> PaperMetricsCollector:
    """Get or create the singleton PaperMetricsCollector.

    Returns:
        The shared :class:`PaperMetricsCollector` singleton.
    """
    global _metrics_instance
    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = PaperMetricsCollector()
    return _metrics_instance


# =============================================================================
# ENTERPRISE: PROMETHEUS-COMPATIBLE EXPORTER
# =============================================================================

class PrometheusExporter:
    """Exports metrics in Prometheus text exposition format.

    Converts :class:`PaperMetricsCollector` data into the standard
    Prometheus text format for scraping by monitoring systems.

    .. code-block:: python

        exporter = get_prometheus_exporter()
        text = exporter.export()
        # Returns Prometheus-format metric lines
    """

    def __init__(self, collector: Optional[PaperMetricsCollector] = None) -> None:
        self._collector = collector or get_paper_metrics()

    def export(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Multi-line string with Prometheus metric definitions.
        """
        lines: List[str] = []
        metrics = self._collector.get_comprehensive_metrics()

        # Query metrics
        perf = metrics.get("performance_metrics", {})
        lines.append("# HELP nexus_queries_total Total queries processed")
        lines.append("# TYPE nexus_queries_total counter")
        lines.append(f'nexus_queries_total {perf.get("total_queries", 0)}')

        lines.append("# HELP nexus_query_latency_avg Average query latency in ms")
        lines.append("# TYPE nexus_query_latency_avg gauge")
        lines.append(f'nexus_query_latency_avg {perf.get("avg_latency_ms", 0)}')

        # Routing metrics
        routing = metrics.get("routing_metrics", {})
        method_dist = routing.get("method_distribution", {})
        lines.append("# HELP nexus_routing_method_total Queries by routing method")
        lines.append("# TYPE nexus_routing_method_total counter")
        for method, count in method_dist.items():
            lines.append(f'nexus_routing_method_total{{method="{method}"}} {count}')

        # Correction metrics
        correction = metrics.get("correction_metrics", {})
        lines.append("# HELP nexus_corrections_total Total correction loops")
        lines.append("# TYPE nexus_corrections_total counter")
        lines.append(f'nexus_corrections_total {correction.get("total_corrections", 0)}')

        lines.append("# HELP nexus_correction_success_rate Correction success rate")
        lines.append("# TYPE nexus_correction_success_rate gauge")
        lines.append(f'nexus_correction_success_rate {correction.get("success_rate", 0)}')

        # Error recovery
        recovery = metrics.get("error_recovery", {})
        lines.append("# HELP nexus_error_recovery_total Total error recovery events")
        lines.append("# TYPE nexus_error_recovery_total counter")
        lines.append(f'nexus_error_recovery_total {recovery.get("total_events", 0)}')

        return "\n".join(lines) + "\n"


# =============================================================================
# ENTERPRISE: METRICS DASHBOARD
# =============================================================================

class MetricsDashboard:
    """Aggregated dashboard view of all system metrics.

    Provides a single-call method to gather metrics from all
    enterprise subsystems into a unified dashboard payload.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._custom_panels: Dict[str, Any] = {}

    def register_panel(self, name: str, data_fn: Any) -> None:
        """Register a custom dashboard panel.

        Args:
            name: Panel identifier.
            data_fn: Callable that returns panel data dict.
        """
        with self._lock:
            self._custom_panels[name] = data_fn

    def get_dashboard(self) -> Dict[str, Any]:
        """Generate the full dashboard payload.

        Returns:
            Dict with core metrics and any registered custom panels.
        """
        collector = get_paper_metrics()
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "core_metrics": collector.get_comprehensive_metrics(),
        }

        with self._lock:
            for name, fn in self._custom_panels.items():
                try:
                    dashboard[name] = fn() if callable(fn) else fn
                except Exception as e:
                    dashboard[name] = {"error": str(e)}

        return dashboard


# =============================================================================
# ENTERPRISE: REAL-TIME TELEMETRY
# =============================================================================

class RealTimeTelemetry:
    """Real-time metric streaming for live monitoring.

    Collects per-second snapshots of key metrics for real-time
    dashboards and alerting systems.
    """

    def __init__(self, window_size: int = 60) -> None:
        self._lock = threading.Lock()
        self._window_size = window_size
        self._latency_window: deque = deque(maxlen=window_size)
        self._throughput_window: deque = deque(maxlen=window_size)
        self._error_window: deque = deque(maxlen=window_size)
        self._last_snapshot = time.time()

    def record_request(self, latency_ms: float, success: bool) -> None:
        """Record a single request for telemetry.

        Args:
            latency_ms: Request latency.
            success: Whether the request succeeded.
        """
        with self._lock:
            self._latency_window.append(latency_ms)
            if not success:
                self._error_window.append(time.time())
            now = time.time()
            self._throughput_window.append(now)

    def get_snapshot(self) -> Dict[str, Any]:
        """Get a real-time metric snapshot.

        Returns:
            Dict with current latency, throughput, and error metrics.
        """
        with self._lock:
            lats = list(self._latency_window)
            now = time.time()
            # Count requests in last 60 seconds
            recent = [t for t in self._throughput_window if now - t < 60]
            errors = [t for t in self._error_window if now - t < 60]

            return {
                "current_latency_ms": round(lats[-1], 2) if lats else 0,
                "avg_latency_ms": round(sum(lats) / len(lats), 2) if lats else 0,
                "p95_latency_ms": round(
                    sorted(lats)[int(len(lats) * 0.95)] if len(lats) >= 20 else (max(lats) if lats else 0), 2
                ),
                "requests_per_minute": len(recent),
                "errors_per_minute": len(errors),
                "error_rate": round(len(errors) / max(len(recent), 1), 4),
                "window_size": self._window_size,
            }


# =============================================================================
# ENTERPRISE SINGLETONS
# =============================================================================

_prometheus_exporter: Optional[PrometheusExporter] = None
_prometheus_lock = threading.Lock()


def get_prometheus_exporter() -> PrometheusExporter:
    """Get or create singleton Prometheus exporter (thread-safe)."""
    global _prometheus_exporter
    if _prometheus_exporter is None:
        with _prometheus_lock:
            if _prometheus_exporter is None:
                _prometheus_exporter = PrometheusExporter()
    return _prometheus_exporter
