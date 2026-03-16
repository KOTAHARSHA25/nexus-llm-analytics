"""
Reporter Agent Plugin — Enterprise v2.0
========================================

Handles report generation by compiling analysis results into professional,
stakeholder-ready documents in Markdown and PDF formats. Leverages the
platform's LLM pipeline to transform raw data into structured business
reports with executive summaries, key findings, and actionable
recommendations.

Enterprise v2.0 Additions
-------------------------
* ``ReportMetrics`` dataclass — tracks generation counts, format breakdown
  (PDF / Markdown), average generation time, and first/last timestamps.
* ``EnterpriseReporterAgent`` — extends the base ``ReporterAgent`` with
  live metrics collection, configurable report templates, and a
  ``batch_generate`` helper for bulk report production.
* ``get_reporter_agent()`` — thread-safe singleton accessor using
  double-checked locking so only one enterprise reporter instance exists
  per process.

Backward Compatibility
----------------------
The original ``ReporterAgent`` class is **completely unchanged** in
behaviour. All new functionality is additive and lives in the
``EnterpriseReporterAgent`` subclass and supporting utilities appended at
the end of the module.

.. versionchanged:: 2.0.0
   Added ``ReportMetrics``, ``EnterpriseReporterAgent``, and
   ``get_reporter_agent`` for enterprise-grade report management.

Author: Nexus Team
Date:   2025-12-01
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
from backend.core.enhanced_reports import PDFReportGenerator, ReportTemplate


class ReporterAgent(BasePluginAgent):
    """
    Reporter Agent Plugin.
    Compiles analysis results into professional reports.

    Attributes
    ----------
    initializer : ModelManager
        Reference to the shared model-manager instance used for LLM calls.
    pdf_generator : PDFReportGenerator
        Utility that converts Markdown report text into styled PDF files.

    Methods
    -------
    get_metadata() -> AgentMetadata
        Return static metadata describing this agent's capabilities.
    initialize(**kwargs) -> bool
        Perform lazy setup of LLM client and PDF generator.
    can_handle(query, file_type, **kwargs) -> float
        Score how well this agent matches the incoming request.
    execute(query, data, **kwargs) -> Dict[str, Any]
        Generate a report for the given query and data payload.

    .. versionchanged:: 2.0.0
       Docstring enriched with Attributes / Methods sections; runtime
       behaviour is identical to v1.x.
    """
    
    def get_metadata(self) -> AgentMetadata:
        """Return metadata describing the Reporter agent.

        Returns
        -------
        AgentMetadata
            Includes name, version, supported file types, and dependencies.
        """
        return AgentMetadata(
            name="Reporter",
            version="2.0.0",
            description="Compiles comprehensive business reports",
            author="Nexus Team",
            capabilities=[AgentCapability.REPORTING],
            file_types=[".pdf", ".md", ".txt"],
            dependencies=["reportlab"],
            priority=20
        )
    
    def initialize(self, **kwargs: Any) -> bool:
        """Initialise the LLM client and PDF generator.

        Parameters
        ----------
        **kwargs : Any
            Reserved for future configuration overrides.

        Returns
        -------
        bool
            ``True`` when initialisation succeeds.
        """
        self.registry = kwargs.get("registry")
        self.initializer = get_model_manager()
        self.pdf_generator = PDFReportGenerator()
        return True
    
    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs: Any) -> float:
        """Score how well this agent can handle *query*.

        Parameters
        ----------
        query : str
            The user's natural-language request.
        file_type : str, optional
            MIME or extension hint for the target file.
        **kwargs : Any
            Additional routing context.

        Returns
        -------
        float
            Confidence score in ``[0.0, 1.0]``.  Returns ``0.8`` when
            report-related keywords are detected, ``0.0`` otherwise.
        """
        query_lower = query.lower()
        if "summary statistics" in query_lower or "summary stats" in query_lower:
            return 0.0
        keywords = ["report", "writeup", "document results", "generate report"]
        if any(k in query_lower for k in keywords):
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
                # result['result'] can be a string (markdown) or a dict (pdf info)
                res_val = result.get('result')
                summary = "Report Generated"
                
                if isinstance(res_val, str):
                    summary = f"Report Generated (Markdown): {res_val[:100]}..."
                elif isinstance(res_val, dict):
                     if 'pdf_path' in res_val:
                         summary = f"PDF Report Generated: {res_val.get('pdf_path')}"
                     else:
                         summary = f"Report Generated: {str(res_val)[:100]}..."

                content = {
                    "query": query,
                    "summary": summary,
                    "metadata": result.get('metadata', {})
                }
                
                self.publish_insight(
                    insight_type="reporting_success",
                    content=content,
                    confidence=0.9
                )
                logging.info(f"[{self.metadata.name}] Published insight to Swarm")
            except Exception as e:
                logging.warning(f"Failed to publish insight: {e}")
        
        return result

    def execute(self, query: str, data: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Execute reporting task.

        Generates a professional Markdown report via the LLM and,
        optionally, converts it to PDF when requested.

        Parameters
        ----------
        query : str
            The user's report request.
        data : Any, optional
            Analysis payload to summarise in the report.
        **kwargs : Any
            Accepts ``format`` (``'markdown'`` | ``'pdf'``), ``title``,
            ``company_name``, ``execution_time``, and ``filename``.

        Returns
        -------
        Dict[str, Any]
            ``{"success": True, "result": ..., "metadata": {...}}`` on
            success, or ``{"success": False, "error": ...}`` on failure.
        """
        try:
            self.initializer.ensure_initialized()
            
            # --- SWARM CAPABILITY: Active Data Gathering ---
            # If no data is provided, try to fetch it from the Data Analyst
            if not data and self.registry:
                logging.info(f"ReporterAgent: No data provided for '{query}'. Delegating to DataAnalyst...")
                analysis_result = self.delegate(
                    agent_name="DataAnalyst",
                    query=query,
                    data=None,
                    **kwargs
                )
                
                if analysis_result.get("success"):
                    data = analysis_result.get("result")
                    logging.info("ReporterAgent: Successfully retrieved analysis data.")
                else:
                    logging.warning(f"ReporterAgent: Failed to gather data: {analysis_result.get('error')}")
                    data = "No data could be automatically retrieved for this report."
            # -----------------------------------------------

            results_context = f"Analysis Results:\n{str(data)[:8000]}"
            
            system_prompt = """You are a skilled business analyst and technical writer. 
You excel at transforming complex data analysis into clear, actionable reports for stakeholders at all levels.
Your goal is to create comprehensive and professional analysis reports."""

            user_prompt = f"""
Create a professional report based on the request: "{query}"

DATA/ANALYSIS TO REPORT ON:
{results_context}

REPORT STRUCTURE:
1. Executive Summary
   - Brief overview of the goal and main outcome.
2. Key Findings
   - Bullet points of the most important metrics or discoveries.
3. Detailed Analysis
   - In-depth explanation of the data evidence.
4. Recommendations
   - Actionable next steps based on the data.

FORMATTING INSTRUCTIONS:
- Use Professional Markdown.
- Use bolding for key terms.
- Keep sections clear and distinct.
- Do NOT include any 'Thought/Action' trace. Just write the report.
"""
            
            response = self.initializer.llm_client.generate(
                prompt=user_prompt,
                system=system_prompt,
                model=self.initializer.primary_llm.model
            )
            
            report_text = response.get('response', str(response)) if isinstance(response, dict) else str(response)
            
            output_format = kwargs.get('format', 'markdown').lower()
            
            if output_format == 'markdown' and ('pdf' in query.lower() or 'portable document' in query.lower()):
                output_format = 'pdf'
                
            result_payload = {"report_text": report_text}
            
            if output_format == 'pdf':
                try:
                    title = kwargs.get('title', f"Analysis Report - {query[:30]}...")
                    template = ReportTemplate(title=title)
                    if 'company_name' in kwargs:
                        template.company_name = kwargs['company_name']
                    
                    self.pdf_generator.template = template

                    analysis_data = {
                        'query': query,
                        'result': report_text,
                        'success': True,
                        'execution_time': kwargs.get('execution_time', 0),
                        'filename': kwargs.get('filename', 'Generated Report'),
                        'type': 'Detailed Analysis'
                    }
                    
                    pdf_path = self.pdf_generator.generate_report([analysis_data])
                    result_payload['pdf_path'] = pdf_path
                    logging.info(f"PDF Report generated at: {pdf_path}")
                except Exception as pdf_error:
                    logging.error(f"PDF generation failed: {pdf_error}")
                    result_payload['pdf_error'] = str(pdf_error)
            
            return {
                "success": True,
                "result": result_payload if output_format == 'pdf' else report_text,
                "metadata": {"agent": "Reporter", "mode": "direct_generation", "format": output_format}
            }
            
        except Exception as e:
            logging.error(f"Reporter execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# ---------------------------------------------------------------------------
# Enterprise v2.0 Additions
# ---------------------------------------------------------------------------

@dataclass
class ReportMetrics:
    """Tracks cumulative report-generation statistics.

    Attributes
    ----------
    reports_generated : int
        Total number of reports produced.
    pdf_count : int
        Number of reports rendered as PDF.
    markdown_count : int
        Number of reports returned as Markdown.
    total_generation_time : float
        Cumulative wall-clock seconds spent generating reports.
    first_report_at : float | None
        Epoch timestamp of the first report, or ``None`` if no reports yet.
    last_report_at : float | None
        Epoch timestamp of the most recent report.

    .. versionadded:: 2.0.0
    """

    reports_generated: int = 0
    pdf_count: int = 0
    markdown_count: int = 0
    total_generation_time: float = 0.0
    first_report_at: Optional[float] = None
    last_report_at: Optional[float] = None

    # -- derived helpers -----------------------------------------------------

    @property
    def avg_generation_time(self) -> float:
        """Return average generation time in seconds, or ``0.0``."""
        if self.reports_generated == 0:
            return 0.0
        return self.total_generation_time / self.reports_generated

    def record(self, fmt: str, elapsed: float) -> None:
        """Record a single report generation event.

        Parameters
        ----------
        fmt : str
            Output format — ``'pdf'`` or ``'markdown'``.
        elapsed : float
            Wall-clock seconds the generation took.
        """
        now = time.time()
        self.reports_generated += 1
        self.total_generation_time += elapsed
        if fmt == "pdf":
            self.pdf_count += 1
        else:
            self.markdown_count += 1
        if self.first_report_at is None:
            self.first_report_at = now
        self.last_report_at = now

    def to_dict(self) -> Dict[str, Any]:
        """Serialise metrics to a plain dictionary."""
        return {
            "reports_generated": self.reports_generated,
            "pdf_count": self.pdf_count,
            "markdown_count": self.markdown_count,
            "avg_generation_time": round(self.avg_generation_time, 4),
            "total_generation_time": round(self.total_generation_time, 4),
            "first_report_at": self.first_report_at,
            "last_report_at": self.last_report_at,
        }


class EnterpriseReporterAgent(ReporterAgent):
    """Enterprise-grade extension of :class:`ReporterAgent`.

    Adds live metrics tracking, configurable default templates, and a
    convenience ``batch_generate`` method for producing multiple reports in
    one call.

    Attributes
    ----------
    metrics : ReportMetrics
        Cumulative statistics for all reports produced by this instance.
    default_template : ReportTemplate | None
        Optional template applied to every PDF report unless overridden.

    .. versionadded:: 2.0.0
    """

    def __init__(self, config: dict = None) -> None:
        super().__init__(config=config)
        self.metrics: ReportMetrics = ReportMetrics()
        self.default_template: Optional[ReportTemplate] = None
        self._lock: threading.Lock = threading.Lock()

    # -- configuration helpers ------------------------------------------------

    def set_default_template(self, template: ReportTemplate) -> None:
        """Set a default PDF template for all subsequent reports.

        Parameters
        ----------
        template : ReportTemplate
            The template to apply by default.
        """
        self.default_template = template

    # -- overridden execute with metrics -------------------------------------

    def execute(self, query: str, data: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Execute a report generation with metrics tracking.

        Delegates to :meth:`ReporterAgent.execute` and records elapsed
        time and output format into :attr:`metrics`.

        Parameters
        ----------
        query : str
            The user's report request.
        data : Any, optional
            Analysis payload to summarise.
        **kwargs : Any
            Forwarded to the base ``execute`` method; additionally accepts
            ``template`` to override the default PDF template.

        Returns
        -------
        Dict[str, Any]
            Same contract as :meth:`ReporterAgent.execute`, with an extra
            ``metrics`` key in the ``metadata`` dict.
        """
        # Apply default template if none provided
        if self.default_template and "template" not in kwargs:
            kwargs.setdefault("title", self.default_template.title)
            if hasattr(self.default_template, "company_name"):
                kwargs.setdefault("company_name", self.default_template.company_name)

        start = time.time()
        result = super().execute(query, data, **kwargs)
        elapsed = time.time() - start

        fmt = "markdown"
        if result.get("success") and isinstance(result.get("metadata"), dict):
            fmt = result["metadata"].get("format", "markdown")

        with self._lock:
            self.metrics.record(fmt, elapsed)

        # Inject metrics snapshot into response metadata
        if isinstance(result.get("metadata"), dict):
            result["metadata"]["metrics"] = self.metrics.to_dict()

        return result

    # -- batch helper --------------------------------------------------------

    def batch_generate(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate multiple reports sequentially.

        Parameters
        ----------
        tasks : list of dict
            Each dict must contain ``query`` (str) and may contain
            ``data``, ``format``, ``title``, and any other kwargs
            accepted by :meth:`execute`.

        Returns
        -------
        list of dict
            One result dict per task, in the same order as *tasks*.
        """
        results: List[Dict[str, Any]] = []
        for task in tasks:
            query = task.pop("query")
            data = task.pop("data", None)
            result = self.execute(query, data, **task)
            results.append(result)
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of current metrics.

        Returns
        -------
        Dict[str, Any]
            Serialised :class:`ReportMetrics`.
        """
        with self._lock:
            return self.metrics.to_dict()


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_reporter_instance: Optional[EnterpriseReporterAgent] = None
_reporter_lock: threading.Lock = threading.Lock()


def get_reporter_agent() -> EnterpriseReporterAgent:
    """Return the process-wide :class:`EnterpriseReporterAgent` singleton.

    Uses double-checked locking to avoid contention after the first call
    while remaining thread-safe during initialisation.

    Returns
    -------
    EnterpriseReporterAgent
        The shared enterprise reporter instance.

    .. versionadded:: 2.0.0
    """
    global _reporter_instance
    if _reporter_instance is None:
        with _reporter_lock:
            if _reporter_instance is None:
                instance = EnterpriseReporterAgent()
                instance.initialize()
                _reporter_instance = instance
    return _reporter_instance
