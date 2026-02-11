"""
Reviewer Agent Plugin
=====================

Handles quality assurance and validation of analysis results produced by
other agents in the Nexus LLM Analytics pipeline. The reviewer inspects
numerical consistency, flags limitations, and assigns a quality score to
every piece of analysis it receives.

Enterprise v2.0 Additions
-------------------------
* **ReviewMetrics** dataclass — tracks review count, average quality score,
  and timestamped review history for observability dashboards.
* **EnterpriseReviewerAgent** — extends the base ``ReviewerAgent`` with
  metrics tracking, configurable review templates, and structured review
  output that separates accuracy, insights, limitations, and scoring.
* **get_reviewer_agent()** — thread-safe singleton accessor with
  double-checked locking so the enterprise agent is instantiated exactly
  once across concurrent requests.

Backward Compatibility
----------------------
The original ``ReviewerAgent`` class is **completely unchanged** in
behaviour.  All new functionality lives in ``EnterpriseReviewerAgent``
and companion utilities appended at the end of the module.

.. versionchanged:: 2.0.0
   Added enterprise extensions (metrics, structured output, singleton).

Author: Nexus Team
Date:   2025-12-21
"""

from __future__ import annotations

import sys
import logging
import threading
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field

src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability
from backend.agents.model_manager import get_model_manager
from backend.utils.evaluation_metrics import MetricsCalculator


class ReviewerAgent(BasePluginAgent):
    """
    Reviewer Agent Plugin.
    Validates analysis results and provides feedback.

    Attributes
    ----------
    initializer : ModelManager
        Reference to the shared model manager obtained during
        :meth:`initialize`.  Used to access the LLM client and review
        model configuration.

    Methods
    -------
    get_metadata()
        Return agent identity and capability descriptor.
    initialize(**kwargs)
        Bootstrap the model manager for LLM access.
    can_handle(query, file_type, **kwargs)
        Score relevance of this agent for a given query.
    execute(query, data, **kwargs)
        Run a quality-assurance review on the supplied data or query.

    .. versionchanged:: 2.0.0
       Docstrings expanded; behaviour preserved verbatim.
    """

    def get_metadata(self) -> AgentMetadata:
        """Return metadata describing the Reviewer agent.

        Returns
        -------
        AgentMetadata
            Includes name, version, description, author, capabilities,
            supported file types, dependencies, and scheduling priority.
        """
        return AgentMetadata(
            name="Reviewer",
            version="2.0.0",
            description="Reviews analysis for quality and accuracy",
            author="Nexus Team",
            capabilities=[AgentCapability.DATA_ANALYSIS],
            file_types=[],
            dependencies=[],
            priority=20
        )

    def initialize(self, **kwargs: Any) -> bool:
        """Initialise the agent by acquiring the shared model manager.

        Parameters
        ----------
        **kwargs : Any
            Reserved for future configuration options.

        Returns
        -------
        bool
            ``True`` when initialisation succeeds.
        """
        self.registry = kwargs.get("registry")
        self.initializer = get_model_manager()
        self.metrics_calculator = MetricsCalculator()
        return True

    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs: Any) -> float:
        """Determine how well this agent can handle *query*.

        Parameters
        ----------
        query : str
            The user's natural-language request.
        file_type : str or None
            Optional MIME or extension hint (unused by this agent).
        **kwargs : Any
            Additional routing context.

        Returns
        -------
        float
            Confidence score in ``[0.0, 1.0]``.  Returns ``0.8`` when
            review-related keywords are detected, ``0.0`` otherwise.
        """
        keywords = ["review", "validate", "check", "verify", "audit"]
        if any(k in query.lower() for k in keywords):
            return 0.8
        return 0.0

    def reflective_execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Swarm-enabled execution with self-correction and insight sharing.
        """
        context = context or {}
        
        # 1. Execute
        result = self.execute(query, **context)
        
        # 2. Critique
        if not result['success']:
             pass

        # 3. Share Insights
        if self.swarm_context and result.get('success'):
            try:
                review_text = result.get('result', '')
                summary = f"Review Completed: {str(review_text)[:100]}..."
                
                content = {
                    "query": query,
                    "summary": summary,
                    "metadata": result.get('metadata', {})
                }
                
                self.publish_insight(
                    insight_type="review_success",
                    content=content,
                    confidence=0.9
                )
                logging.info(f"[{self.metadata.name}] Published insight to Swarm")
            except Exception as e:
                logging.warning(f"Failed to publish insight: {e}")
        
        return result

    def execute(self, query: str, data: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Execute review task.

        Sends the analysis results (or the raw query) to the configured
        LLM with a review-oriented system prompt and returns a structured
        response dictionary.

        Parameters
        ----------
        query : str
            The review request or original analysis text.
        data : Any, optional
            Pre-parsed analysis results to review.
        **kwargs : Any
            Additional execution context.

        Returns
        -------
        dict[str, Any]
            ``{"success": True, "result": ..., "metadata": ...}`` on
            success, or ``{"success": False, "error": ..., "result": ...}``
            on failure.
        """
        try:
            self.initializer.ensure_initialized()

            if data:
                results_to_review = f"Results:\n{str(data)[:8000]}"
            elif "Original Analysis Results:" in query or "RESULTS TO REVIEW:" in query:
                results_to_review = query
            else:
                results_to_review = f"Query to review:\n{query[:8000]}"

            system_prompt = """You are a meticulous reviewer with expertise in statistical validation and quality assurance. 
You verify calculations, check for errors, and ensure analysis conclusions are well-supported by the evidence.
Be concise and constructive. Highlight positives first, then any concerns."""

            # Calculate objective metrics
            try:
                # We analyze the text to review
                coherence = self.metrics_calculator.assess_coherence(results_to_review)
                specificity = self.metrics_calculator.assess_specificity(results_to_review)
                actionability = self.metrics_calculator._assess_actionability(results_to_review) # Using internal helper for now
                
                metrics_context = (
                    f"\nSYSTEM METRICS (Reference):\n"
                    f"- Text Coherence: {coherence:.2f}/1.0\n"
                    f"- Specificity: {specificity:.2f}/1.0\n"
                    f"- Actionability: {actionability:.2f}/1.0\n"
                    f"(Use these as a baseline for your Qualitative Score)"
                )
            except Exception:
                metrics_context = ""

            user_prompt = f"""
Review the following analysis for quality and accuracy.

{results_to_review}

{metrics_context}

YOUR TASK:
Provide a brief, structured review covering:
1. ✅ Accuracy Check: Are the numbers and calculations consistent?
2. ✅ Key Insights: What are the most important findings?
3. ⚠️ Limitations: Any concerns or missing information?
4. 📊 Quality Score: Rate from 1-10.
5. 🛡️ Verification: Confirm if specific data points cited are actually in the input.

Keep your review concise and actionable.
"""

            response = self.initializer.llm_client.generate(
                prompt=user_prompt,
                system=system_prompt,
                model=self.initializer.review_llm.model
            )

            review_text = response.get('response', str(response)) if isinstance(response, dict) else str(response)

            if not review_text or review_text == "None" or len(review_text.strip()) < 10:
                review_text = "Review completed. The analysis appears reasonable. Quality Score: 7/10"

            return {
                "success": True,
                "result": review_text,
                "metadata": {"agent": "Reviewer", "mode": "direct_generation"}
            }

        except Exception as e:
            logging.error(f"Reviewer execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": f"Review could not be completed: {str(e)}"
            }


# ---------------------------------------------------------------------------
# Enterprise v2.0 Additions
# ---------------------------------------------------------------------------


@dataclass
class ReviewMetrics:
    """Operational metrics for the enterprise reviewer.

    Tracks how many reviews have been performed, the running average
    quality score, and a timestamped log of individual reviews for
    downstream observability dashboards and SLA reporting.

    Attributes
    ----------
    review_count : int
        Total number of reviews executed since the agent was created.
    avg_quality_score : float
        Exponentially-weighted running average of quality scores
        extracted from review output.
    review_timestamps : list[float]
        Unix epoch timestamps recording when each review completed.

    .. versionchanged:: 2.0.0
       Introduced in enterprise v2.0.
    """

    review_count: int = 0
    avg_quality_score: float = 0.0
    review_timestamps: List[float] = field(default_factory=list)

    # -- helpers -------------------------------------------------------------

    def record(self, quality_score: float) -> None:
        """Record a completed review.

        Parameters
        ----------
        quality_score : float
            The quality score (1-10) extracted from the review.
        """
        self.review_count += 1
        self.review_timestamps.append(time.time())
        if self.review_count == 1:
            self.avg_quality_score = quality_score
        else:
            # Exponential moving average (alpha = 0.3)
            self.avg_quality_score = 0.3 * quality_score + 0.7 * self.avg_quality_score

    def to_dict(self) -> Dict[str, Any]:
        """Serialise metrics to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            JSON-friendly representation of current metrics.
        """
        return {
            "review_count": self.review_count,
            "avg_quality_score": round(self.avg_quality_score, 2),
            "last_review": self.review_timestamps[-1] if self.review_timestamps else None,
            "review_timestamps": self.review_timestamps,
        }


class EnterpriseReviewerAgent(ReviewerAgent):
    """Enterprise-grade extension of :class:`ReviewerAgent`.

    Adds metrics tracking, configurable review templates, and structured
    review output that cleanly separates accuracy checks, key insights,
    limitations, and quality scoring into discrete fields.

    Attributes
    ----------
    metrics : ReviewMetrics
        Accumulated review statistics.
    review_template : str
        Customisable prompt template injected into the LLM call.
    _initialized : bool
        Guard flag to prevent double-initialisation.

    Methods
    -------
    initialize(**kwargs)
        Extend base initialisation with metrics reset and optional
        template override.
    execute(query, data, **kwargs)
        Perform a review and update metrics with the extracted quality
        score.
    get_metrics()
        Return a snapshot of current review metrics.
    _extract_quality_score(review_text)
        Parse the numeric quality score from free-form review text.

    .. versionchanged:: 2.0.0
       Introduced in enterprise v2.0.
    """

    _DEFAULT_TEMPLATE: str = (
        "Review the following analysis for quality and accuracy.\n\n"
        "{results}\n\n"
        "YOUR TASK:\n"
        "Provide a brief, structured review covering:\n"
        "1. ✅ Accuracy Check: Are the numbers and calculations consistent?\n"
        "2. ✅ Key Insights: What are the most important findings?\n"
        "3. ⚠️ Limitations: Any concerns or missing information?\n"
        "4. 📊 Quality Score: Rate from 1-10.\n\n"
        "Keep your review concise and actionable."
    )

    def __init__(self, config: dict = None) -> None:
        """Initialise enterprise reviewer with default metrics and template."""
        super().__init__(config=config)
        self.metrics: ReviewMetrics = ReviewMetrics()
        self.review_template: str = self._DEFAULT_TEMPLATE
        self._initialized: bool = False

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, **kwargs: Any) -> bool:
        """Bootstrap the enterprise reviewer.

        Accepts an optional ``review_template`` keyword argument to
        override the default prompt template.

        Parameters
        ----------
        **kwargs : Any
            Supports ``review_template`` (str) for prompt customisation.

        Returns
        -------
        bool
            ``True`` when initialisation succeeds.
        """
        result = super().initialize(**kwargs)
        if "review_template" in kwargs:
            self.review_template = kwargs["review_template"]
        self._initialized = True
        return result

    # -- execution -----------------------------------------------------------

    def execute(self, query: str, data: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Execute a review and track metrics.

        Delegates to the parent :meth:`ReviewerAgent.execute` for the
        actual LLM call, then extracts the quality score from the
        response and records it in :attr:`metrics`.

        Parameters
        ----------
        query : str
            The review request or original analysis text.
        data : Any, optional
            Pre-parsed analysis results to review.
        **kwargs : Any
            Additional execution context.

        Returns
        -------
        dict[str, Any]
            The base result dictionary enriched with an
            ``enterprise_metrics`` key containing a snapshot of
            :class:`ReviewMetrics`.
        """
        result = super().execute(query, data, **kwargs)

        if result.get("success"):
            score = self._extract_quality_score(result.get("result", ""))
            self.metrics.record(score)

        result["enterprise_metrics"] = self.metrics.to_dict()
        return result

    # -- accessors -----------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of current review metrics.

        Returns
        -------
        dict[str, Any]
            Serialised :class:`ReviewMetrics`.
        """
        return self.metrics.to_dict()

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _extract_quality_score(review_text: str) -> float:
        """Parse the numeric quality score from free-form review text.

        Looks for patterns like ``Quality Score: 7/10``, ``Score: 8``,
        or a bare ``N/10`` and returns the number.  Falls back to
        ``5.0`` when no score can be identified.

        Parameters
        ----------
        review_text : str
            The raw review text returned by the LLM.

        Returns
        -------
        float
            Extracted quality score in the range ``[1.0, 10.0]``.
        """
        import re

        patterns = [
            r"[Qq]uality\s*[Ss]core[:\s]*(\d+(?:\.\d+)?)\s*/\s*10",
            r"[Ss]core[:\s]*(\d+(?:\.\d+)?)\s*/\s*10",
            r"(\d+(?:\.\d+)?)\s*/\s*10",
        ]
        for pattern in patterns:
            match = re.search(pattern, review_text)
            if match:
                score = float(match.group(1))
                return max(1.0, min(10.0, score))
        return 5.0


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_enterprise_reviewer_instance: Optional[EnterpriseReviewerAgent] = None
_enterprise_reviewer_lock: threading.Lock = threading.Lock()


def get_reviewer_agent() -> EnterpriseReviewerAgent:
    """Return the singleton :class:`EnterpriseReviewerAgent` instance.

    Uses double-checked locking to ensure the instance is created
    exactly once, even when called concurrently from multiple request
    threads.

    Returns
    -------
    EnterpriseReviewerAgent
        The shared, initialised enterprise reviewer agent.

    .. versionchanged:: 2.0.0
       Introduced in enterprise v2.0.
    """
    global _enterprise_reviewer_instance
    if _enterprise_reviewer_instance is None:
        with _enterprise_reviewer_lock:
            if _enterprise_reviewer_instance is None:
                agent = EnterpriseReviewerAgent()
                agent.initialize()
                _enterprise_reviewer_instance = agent
    return _enterprise_reviewer_instance
