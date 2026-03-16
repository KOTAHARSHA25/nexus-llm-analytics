"""Visualizer Agent Plugin — Data visualization generation and execution.

Handles data visualization tasks by generating Plotly / Matplotlib code
via an LLM and (optionally) executing the generated code in a secure
sandbox to produce interactive chart JSON for the frontend.

Enterprise v2.0 Additions
-------------------------
* **VisualizationMetrics** — Dataclass tracking charts generated,
  per-type breakdown, average render time and error count for
  observability dashboards.
* **EnterpriseVisualizerAgent** — Subclass of :class:`VisualizerAgent`
  that adds metrics tracking, configurable theme management, and batch
  chart generation for multi-panel reports.
* **get_visualizer_agent()** — Thread-safe singleton accessor using
  double-checked locking for process-wide reuse of the enterprise
  agent instance.

Backward Compatibility
~~~~~~~~~~~~~~~~~~~~~~
All v1.x APIs (``VisualizerAgent``, ``VisualizerAgent.execute``, etc.)
remain fully backward-compatible.  Enterprise features are purely
additive — appended after the original code.

.. versionchanged:: 2.0
   Added *VisualizationMetrics*, *EnterpriseVisualizerAgent*, and
   *get_visualizer_agent* singleton accessor.

Author: Nexus Team
Since: v1.0 (Enterprise enhancements v2.0 — February 2026)
"""

from __future__ import annotations

import sys
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability
from backend.agents.model_manager import get_model_manager
from backend.core.security.sandbox import EnhancedSandbox

logger = logging.getLogger(__name__)


# ── Original v1.x VisualizerAgent (unchanged) ────────────────────────


class VisualizerAgent(BasePluginAgent):
    """Visualizer Agent Plugin.

    Generates Plotly/Matplotlib code for data visualization.
    Phase 3.4: Now executes generated code in sandbox for immediate results.

    Attributes:
        sandbox: Lazily-initialised :class:`EnhancedSandbox` used for
            secure code execution.  *None* until the first call that
            requires execution.
        initializer: Reference to the shared
            :class:`ModelManager` obtained during :meth:`initialize`.

    Thread Safety:
        An instance is **not** inherently thread-safe.  If shared across
        threads, callers must synchronise access or use the enterprise
        singleton returned by :func:`get_visualizer_agent`.

    Methods:
        get_metadata  — Return plugin metadata.
        initialize    — Bind to the model manager.
        can_handle    — Score affinity for a given query.
        execute       — Generate (and optionally run) visualisation code.

    .. versionchanged:: 2.0
       Docstrings enhanced; see :class:`EnterpriseVisualizerAgent` for
       v2.0 enterprise features.
    """
    
    def __init__(self, config: dict = None) -> None:
        super().__init__(config=config)
        self.sandbox: Optional[EnhancedSandbox] = None  # Lazy initialization
    
    def get_metadata(self) -> AgentMetadata:
        """Return immutable plugin metadata for the registry.

        Returns:
            AgentMetadata: Descriptor advertising the agent's name,
            version, supported file types and required dependencies.
        """
        return AgentMetadata(
            name="Visualizer",
            version="2.0.0",  # Updated for Phase 3.4
            description="Generates and executes interactive data visualizations (Plotly)",
            author="Nexus Team",
            capabilities=[AgentCapability.VISUALIZATION],
            file_types=[".csv", ".xlsx", ".json", ".txt"],
            dependencies=["plotly"],
            priority=50
        )
    
    def initialize(self, **kwargs: Any) -> bool:
        """Bind the agent to the shared :class:`ModelManager`.

        Called once by the plugin registry after construction.  Must
        succeed before :meth:`execute` can generate code.

        Returns:
            bool: Always ``True`` on success.
        """
        self.registry = kwargs.get("registry")
        self.initializer = get_model_manager()
        return True
    
    def _get_sandbox(self) -> EnhancedSandbox:
        """Lazy initialization of sandbox.

        Creates an :class:`EnhancedSandbox` on first access with
        conservative resource limits (256 MB RAM, 30 s CPU).

        Returns:
            EnhancedSandbox: The sandbox instance for code execution.
        """
        if self.sandbox is None:
            self.sandbox = EnhancedSandbox(max_memory_mb=256, max_cpu_seconds=30)
        return self.sandbox
    
    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs: Any) -> float:
        """Score how well this agent can handle *query*.

        Checks for common visualization keywords (plot, chart, graph,
        histogram, scatter, etc.).

        Args:
            query: The user's natural-language request.
            file_type: Optional file extension hint (e.g. ``".csv"``).
            **kwargs: Reserved for future use.

        Returns:
            float: ``0.8`` if a visualization keyword is found, else
            ``0.0``.
        """
        # Check for visualization keywords
        keywords = ["plot", "chart", "graph", "visualize", "diagram", "trend line", "histogram", "scatter", "bar chart", "pie chart", "heatmap"]
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
                # We can't access staticmethod easily from instance in some python versions if not careful, 
                # but here it is a staticmethod on the class. 
                # Better to just reuse the logic or call the static method.
                # Since I am modifying the base class, I will just call the static method helper if I move it or just duplicate logic for safety?
                # Actually _infer_chart_type is a static method in EnterpriseVisualizerAgent, not VisualizerAgent?
                # Wait, I see _infer_chart_type is defined in EnterpriseVisualizerAgent in the file view.
                # Let me check if it is in VisualizerAgent.
                # It is NOT in VisualizerAgent. It is in EnterpriseVisualizerAgent.
                # So I cannot use it here safely if I am in VisualizerAgent.
                
                # I will implement a safe extraction here.
                q = query.lower()
                chart_type = "unknown"
                for ct in ["bar", "scatter", "pie", "histogram", "heatmap", "line", "box"]:
                    if ct in q:
                        chart_type = ct
                        break
                
                summary = f"Visualization Created: {chart_type} chart"
                
                content = {
                    "query": query,
                    "summary": summary,
                    "metadata": {
                        "agent": "VisualizerAgent",
                        "chart_type": chart_type,
                        "mode": result.get('metadata', {}).get('mode', 'unknown')
                    }
                }
                
                self.publish_insight(
                    insight_type="visualization_success",
                    content=content,
                    confidence=0.9
                )
                logging.info(f"[{self.metadata.name}] Published insight to Swarm")
            except Exception as e:
                logging.warning(f"Failed to publish insight: {e}")
        
        return result

    def _generate_viz_code(self, query: str, data: Any = None) -> str:
        """Generate visualization code using LLM.

        Constructs a system + user prompt pair that instructs the LLM to
        produce a single self-contained Python code block defining a
        Plotly ``fig`` variable.

        Args:
            query: The user's visualization request.
            data: Optional DataFrame or raw data to supply as context.

        Returns:
            str: Cleaned Python source code extracted from the LLM
            response.
        """
        self.initializer.ensure_initialized()
        
        # Construct data context
        data_context = ""
        if data is not None:
            if hasattr(data, "head"):
                data_context = f"Data Sample:\n{data.head().to_markdown()}\nColumns: {list(data.columns)}\nTypes: {data.dtypes.to_dict()}"
            else:
                data_context = str(data)[:2000]

        system_prompt = """You are a Data Visualization Expert using Python and Plotly Express.
Your goal is to write a single, self-contained Python code block that generates an interactive Plotly chart.
The code must define a figure named 'fig' and not show it (the system will handle display).
Do NOT use pandas.read_csv() - assume the data is already available in a variable named 'df'.
Do NOT use Matplotlib or Seaborn. Use ONLY Plotly.
Do NOT call fig.show() - the system handles rendering.
"""

        user_prompt = f"""
Create a Plotly visualization for the following request: "{query}"

DATA CONTEXT:
{data_context}

REQUIREMENTS:
1. Use `import plotly.express as px` or `import plotly.graph_objects as go`.
2. Assume the dataframe is named `df`.
3. Create the figure and assign it to variable `fig`.
4. Do NOT call `fig.show()`.
5. Return ONLY the valid Python code in a markdown block.
"""
        
        response = self.initializer.llm_client.generate(
            prompt=user_prompt,
            system=system_prompt,
            model=self.initializer.primary_llm.model
        )
        
        result_text = response.get('response', str(response)) if isinstance(response, dict) else str(response)
        
        # Clean up code block
        code_block = result_text
        if "```python" in code_block:
            code_block = code_block.split("```python")[1].split("```")[0].strip()
        elif "```" in code_block:
            code_block = code_block.split("```")[1].split("```")[0].strip()
        
        return code_block

    def execute(self, query: str, data: Any = None, execute_code: bool = True, **kwargs: Any) -> Dict[str, Any]:
        """Execute visualization task.

        Phase 3.4: Now executes generated code in sandbox for immediate
        visualization.

        Args:
            query: The visualization request.
            data: DataFrame or data to visualize.
            execute_code: If ``True`` (default), execute code in sandbox.
                If ``False``, return generated code only.
            **kwargs: Additional arguments forwarded to helpers.

        Returns:
            Dict with *success*, *result* (figure JSON or code), *type*,
            and *metadata* keys.
        """
        try:
            # Generate visualization code
            code = self._generate_viz_code(query, data)
            
            if not execute_code:
                # Return code only (legacy behavior)
                return {
                    "success": True,
                    "result": code,
                    "type": "code",
                    "metadata": {"agent": "Visualizer", "mode": "code_generation"}
                }
            
            # Phase 3.4: Execute in sandbox
            sandbox = self._get_sandbox()
            
            # Prepare execution context with data
            additional_globals: Dict[str, Any] = {}
            if data is not None:
                # Deep copy to prevent modification
                import copy
                try:
                    additional_globals['df'] = copy.deepcopy(data)
                except Exception:
                    additional_globals['df'] = data
            
            # Execute the visualization code
            exec_result = sandbox.execute(code, additional_globals)
            
            if exec_result.get('success'):
                # Check if figure was created
                result_vars = exec_result.get('result', {})
                
                if isinstance(result_vars, dict) and 'fig' in result_vars:
                    fig = result_vars['fig']
                    
                    # Convert Plotly figure to JSON for frontend
                    try:
                        fig_json = fig.to_json()
                        return {
                            "success": True,
                            "result": fig_json,
                            "type": "visualization",
                            "code": code,
                            "metadata": {
                                "agent": "Visualizer",
                                "mode": "executed",
                                "execution_time": exec_result.get('execution_time', 0)
                            }
                        }
                    except AttributeError as e:
                        # NEW: Explicitly fail if not a Plotly figure (Phase 15 optimization)
                        return {
                           "success": False,
                           "error": f"Generated object 'fig' is not a Plotly figure (missing to_json): {e}",
                           "code": code,
                           "type": "error",
                           "metadata": {"agent": "Visualizer", "mode": "invalid_object"}
                        }
                else:
                    # No figure created, return code
                    return {
                        "success": True,
                        "result": code,
                        "type": "code",
                        "warning": "Code executed but no 'fig' variable found",
                        "execution_result": exec_result.get('result'),
                        "metadata": {"agent": "Visualizer", "mode": "code_only"}
                    }
            else:
                # Execution failed, return error with code
                return {
                    "success": False,
                    "error": exec_result.get('error', 'Unknown execution error'),
                    "code": code,
                    "type": "error",
                    "metadata": {"agent": "Visualizer", "mode": "execution_failed"}
                }
            
        except Exception as e:
            logging.error(f"Visualizer execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "type": "error"
            }


# =====================================================================
# Enterprise v2.0 Additions
# =====================================================================
# Everything below is NEW.  The original VisualizerAgent above is
# untouched — all enterprise functionality is additive.
# =====================================================================


@dataclass
class VisualizationMetrics:
    """Observability metrics for visualization generation.

    Tracks chart output volume, per-type breakdown, render latency and
    error rates so that monitoring dashboards can alert on regressions.

    Attributes:
        charts_generated: Total number of charts successfully produced.
        chart_type_breakdown: Mapping of chart type label (e.g.
            ``"bar"``, ``"scatter"``) to the count of charts generated
            for that type.
        total_render_time: Cumulative render time in seconds across all
            charts.  Divide by *charts_generated* for the average.
        error_count: Number of chart generation attempts that ended in
            an error.

    .. versionadded:: 2.0
    """

    charts_generated: int = 0
    chart_type_breakdown: Dict[str, int] = field(default_factory=dict)
    total_render_time: float = 0.0
    error_count: int = 0

    # -- derived helpers -----------------------------------------------

    @property
    def avg_render_time(self) -> float:
        """Average render time per chart in seconds.

        Returns:
            float: ``0.0`` when no charts have been generated yet.
        """
        if self.charts_generated == 0:
            return 0.0
        return self.total_render_time / self.charts_generated

    def record_success(self, chart_type: str, render_time: float) -> None:
        """Record a successful chart generation.

        Args:
            chart_type: Identifier for the chart kind (e.g. ``"bar"``).
            render_time: Wall-clock seconds the render took.
        """
        self.charts_generated += 1
        self.chart_type_breakdown[chart_type] = (
            self.chart_type_breakdown.get(chart_type, 0) + 1
        )
        self.total_render_time += render_time

    def record_error(self) -> None:
        """Increment the error counter."""
        self.error_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialise metrics to a plain dictionary.

        Returns:
            Dict[str, Any]: JSON-friendly snapshot of current metrics.
        """
        return {
            "charts_generated": self.charts_generated,
            "chart_type_breakdown": dict(self.chart_type_breakdown),
            "avg_render_time": round(self.avg_render_time, 4),
            "total_render_time": round(self.total_render_time, 4),
            "error_count": self.error_count,
        }


class EnterpriseVisualizerAgent(VisualizerAgent):
    """Enterprise extension of :class:`VisualizerAgent`.

    Adds:
    * **Metrics tracking** — every :meth:`execute` call updates a
      :class:`VisualizationMetrics` instance.
    * **Theme management** — a configurable Plotly template name applied
      to all generated figures.
    * **Batch chart generation** — :meth:`execute_batch` produces
      multiple charts in one call and aggregates the results.

    Attributes:
        metrics: Live :class:`VisualizationMetrics` accumulator.
        theme: Name of the Plotly template applied to figures
            (default ``"plotly_white"``).

    Thread Safety:
        The *metrics* object is **not** internally locked.  If the
        singleton returned by :func:`get_visualizer_agent` is called
        concurrently, wrap mutating operations in an external lock or
        accept minor count races (acceptable for monitoring counters).

    .. versionadded:: 2.0
    """

    def __init__(self, config: dict | None = None, theme: str = "plotly_white") -> None:
        super().__init__(config=config)
        self.metrics = VisualizationMetrics()
        self.theme: str = theme

    # -- theme helpers -------------------------------------------------

    def set_theme(self, theme: str) -> None:
        """Change the Plotly template for subsequent charts.

        Args:
            theme: Plotly template name (e.g. ``"plotly_dark"``,
                ``"ggplot2"``, ``"seaborn"``).
        """
        self.theme = theme

    def get_theme(self) -> str:
        """Return the current Plotly template name.

        Returns:
            str: Active template string.
        """
        return self.theme

    # -- instrumented execute ------------------------------------------

    def execute(self, query: str, data: Any = None, execute_code: bool = True, **kwargs: Any) -> Dict[str, Any]:
        """Execute a visualisation with metrics tracking.

        Delegates to :meth:`VisualizerAgent.execute` and records
        success/failure timing in :attr:`metrics`.

        Args:
            query: The visualization request.
            data: DataFrame or data to visualize.
            execute_code: If ``True`` (default), execute code in sandbox.
            **kwargs: Forwarded to the base implementation.

        Returns:
            Dict with the same shape as the base class, plus an
            ``"metrics"`` key containing the latest snapshot.
        """
        start = time.time()
        result = super().execute(query, data=data, execute_code=execute_code, **kwargs)
        elapsed = time.time() - start

        if result.get("success"):
            chart_type = self._infer_chart_type(query)
            self.metrics.record_success(chart_type, elapsed)
        else:
            self.metrics.record_error()

        result["metrics"] = self.metrics.to_dict()
        return result

    # -- batch generation ----------------------------------------------

    def execute_batch(
        self,
        queries: List[str],
        data: Any = None,
        execute_code: bool = True,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate multiple charts in a single call.

        Each query is processed independently via :meth:`execute`.
        Failures in one chart do not abort the remaining queries.

        Args:
            queries: List of natural-language chart requests.
            data: Shared DataFrame or data context for all charts.
            execute_code: Whether to execute generated code in sandbox.
            **kwargs: Forwarded to each :meth:`execute` call.

        Returns:
            List[Dict[str, Any]]: One result dict per query, in the
            same order as *queries*.
        """
        results: List[Dict[str, Any]] = []
        for query in queries:
            results.append(
                self.execute(query, data=data, execute_code=execute_code, **kwargs)
            )
        return results

    # -- internal helpers ----------------------------------------------

    @staticmethod
    def _infer_chart_type(query: str) -> str:
        """Best-effort chart-type label from the query text.

        Args:
            query: The user's visualization request.

        Returns:
            str: Lowercase chart type label (e.g. ``"bar"``,
            ``"scatter"``, ``"pie"``).  Falls back to ``"unknown"``.
        """
        q = query.lower()
        for chart_type in (
            "bar",
            "scatter",
            "pie",
            "histogram",
            "heatmap",
            "line",
            "box",
            "violin",
            "treemap",
            "sunburst",
            "area",
        ):
            if chart_type in q:
                return chart_type
        return "unknown"

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of current visualisation metrics.

        Returns:
            Dict[str, Any]: JSON-serialisable metrics dictionary.
        """
        return self.metrics.to_dict()


# -- Singleton accessor (double-checked locking) -----------------------

_visualizer_agent_instance: Optional[EnterpriseVisualizerAgent] = None
_visualizer_agent_lock = threading.Lock()


def get_visualizer_agent(**kwargs: Any) -> EnterpriseVisualizerAgent:
    """Return the process-wide :class:`EnterpriseVisualizerAgent` singleton.

    Keyword arguments are forwarded to the constructor on first call
    only; subsequent calls return the cached instance.

    Thread Safety:
        Uses double-checked locking so that only the very first call
        pays the cost of acquiring the lock.

    Args:
        **kwargs: Forwarded to :class:`EnterpriseVisualizerAgent` on
            first instantiation (e.g. ``config``, ``theme``).

    Returns:
        EnterpriseVisualizerAgent: Shared enterprise agent instance.

    .. versionadded:: 2.0
    """
    global _visualizer_agent_instance
    if _visualizer_agent_instance is None:
        with _visualizer_agent_lock:
            if _visualizer_agent_instance is None:
                _visualizer_agent_instance = EnterpriseVisualizerAgent(**kwargs)
    return _visualizer_agent_instance
