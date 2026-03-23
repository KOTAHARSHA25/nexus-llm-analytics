"""Dynamic Analysis Planner.

Generates domain-specific analysis plans based on data content,
enabling the system to handle any data domain (Biology, Sports,
Retail, Finance, etc.) without hard-coded agents.

Key classes:

* **DynamicPlanner** — Uses the LLM to inspect data previews and
  produce structured :class:`AnalysisPlan` objects.
* **AnalysisPlan** / **AnalysisStep** — Dataclasses representing
  a multi-step execution plan.
* **repair_json** / **safe_json_parse** — Robust JSON recovery
  utilities for malformed LLM outputs.

Enterprise v2.0 Additions
-------------------------
* **PlanningMetrics** — Tracks planner invocations, LLM latency,
  fallback rates, and average plan confidence for observability.

All v1.x APIs (``DynamicPlanner``, ``get_dynamic_planner``,
``AnalysisPlan``, ``AnalysisStep``, ``repair_json``,
``safe_json_parse``) remain fully backward-compatible.

Author: Nexus Team
Since: v1.0 (Enterprise enhancements v2.0 — February 2026)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.agents.model_manager import get_model_manager

logger = logging.getLogger(__name__)


def repair_json(malformed_json: str) -> Optional[dict]:
    """
    Attempt to repair common JSON malformations from LLMs.
    
    Common issues:
    - Trailing commas
    - Single quotes instead of double
    - Unquoted keys
    - Missing closing brackets
    - Conversational text around JSON
    """
    text = malformed_json.strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try raw_decode to handle trailing text after valid JSON
    try:
        if '{' in text:
            start = text.index('{')
            decoder = json.JSONDecoder()
            result, _ = decoder.raw_decode(text[start:])
            if isinstance(result, dict):
                return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Repair attempt 1: Extract JSON from markdown code block
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            text = match.group(1).strip()
    
    # Repair attempt 2: Extract the outermost { ... } block (greedy)
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            text = candidate  # Use extracted block for further repairs
    
    # Repair attempt 3: Remove trailing commas
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Repair attempt 4: Replace single quotes with double quotes
    text = re.sub(r"'([^']*)':", r'"\1":', text)
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
    
    # Try parsing after quotes and commas fix
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Repair attempt 5: Add missing closing brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    
    if open_braces > 0 or open_brackets > 0:
        attempts = [
            text + ']' * open_brackets + '}' * open_braces,
            text + '}' * open_braces + ']' * open_brackets,
        ]
        
        for attempt in attempts:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
    
    # Repair attempt 6: Try raw_decode after all repairs
    try:
        decoder = json.JSONDecoder()
        if '{' in text:
            result, _ = decoder.raw_decode(text[text.index('{'):])
            if isinstance(result, dict):
                return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


def safe_json_parse(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with automatic repair.
    Returns default if parsing fails completely.
    """
    result = repair_json(text)
    return result if result is not None else default


@dataclass
class AnalysisStep:
    """A single step in the dynamic analysis plan"""
    step_id: int
    description: str
    tool: str  # e.g., "python_pandas", "visualization", "rag"
    reasoning: str

@dataclass
class AnalysisPlan:
    """Complete analysis plan"""
    domain: str
    summary: str
    steps: List[AnalysisStep]
    confidence: float

class DynamicPlanner:
    """
    Planner that uses LLM to inspect data structure and generic meaningful analysis steps.
    """
    
    def __init__(self) -> None:
        self._initializer = get_model_manager()
        
    @property
    def llm(self) -> Any:
        """The underlying LLM client used for plan generation."""
        return self._initializer.primary_llm or self._initializer.llm_client

    def create_plan(self, query: str, data_preview: str, model: str | None = None) -> AnalysisPlan:
        """
        Generate a dynamic analysis plan with retry logic.
        
        Args:
            query: User's intent
            data_preview: String summary of the data (from DataOptimizer)
            model: Model to use for planning
            
        Returns:
            AnalysisPlan object
        """
        import time
        max_retries = 3
        backoff_factor = 2
        
        prompt = self._build_planning_prompt(query, data_preview)
        
        # Use smart model for planning
        planning_model = model or self._initializer.primary_llm
        
        # Extract string name if it's an object (OllamaLLM)
        if hasattr(planning_model, 'model'):
            planning_model = planning_model.model
            
        for attempt in range(max_retries):
            try:
                # Support both LangChain shim (.invoke) and custom offline client (.generate)
                if hasattr(self.llm, 'invoke'):
                    response = self.llm.invoke(prompt, model=str(planning_model))
                else:
                    response = self.llm.generate(
                        prompt=prompt,
                        model=str(planning_model),
                        adaptive_timeout=True
                    )
                
                # Normalize response string
                if isinstance(response, dict):
                    response_text = response.get('response', '')
                else:
                    response_text = str(response)
                    
                if not response_text:
                    # If empty response, raise error to trigger retry
                    raise ValueError("Empty or failed response from LLM")
                    
                return self._parse_plan(response_text)
                
            except Exception as e:
                logger.warning(f"Dynamic planning attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor ** attempt
                    time.sleep(sleep_time)
                else:
                    logger.error("Dynamic planning failed after all retries.", exc_info=True)
                    # Fallback plan
                    return AnalysisPlan(
                        domain="General",
                        summary="Fallback analysis due to planning error",
                        steps=[
                            AnalysisStep(1, f"Analyze the data to answer: {query}", "generic_analysis", "Fallback")
                        ],
                        confidence=0.1
                    )
    
    def _build_planning_prompt(self, query: str, data_preview: str) -> str:
        return f"""You are an Expert Data Science Planner.
Create a step-by-step analysis plan to answer the user's query based on the provided dataset.

USER QUERY: "{query}"

DATASET PREVIEW:
{data_preview}

INSTRUCTIONS:
1. Identify the DOMAIN of the data (e.g., Finance, Genomics, Sports, Web Logs).
2. Formulate a logical 2-4 step plan to answer the query.
3. If the query is vague (e.g., "analyze this"), propose 3 interesting angles found in the data.

Output ONLY valid JSON (no text before or after):
{{"domain": "Detected Domain", "summary": "Brief summary", "confidence": 0.9, "steps": [{{"id": 1, "description": "Step description", "tool": "python_pandas", "reasoning": "Why"}}]}}
"""

    def _parse_plan(self, llm_output: str) -> AnalysisPlan:
        """Parse LLM JSON output into AnalysisPlan object with automatic repair"""
        # Use repair_json for robust parsing
        plan_dict = repair_json(llm_output)
        
        if plan_dict is None:
            logger.debug("Failed to parse planning JSON after repair attempts. Using fallback.")
            return AnalysisPlan(
                domain="General",
                summary="Direct Analysis",
                steps=[AnalysisStep(1, llm_output[:200], "general_analysis", "JSON repair failed")],
                confidence=0.1
            )
        
        try:
            steps = [
                AnalysisStep(
                    step_id=s.get('id', i),
                    description=s.get('description', 'Unknown step'),
                    tool=s.get('tool', 'python_pandas'),
                    reasoning=s.get('reasoning', '')
                )
                for i, s in enumerate(plan_dict.get('steps', []), 1)
            ]
            
            return AnalysisPlan(
                domain=plan_dict.get('domain', 'General'),
                summary=plan_dict.get('summary', 'Analysis Plan'),
                steps=steps,
                confidence=float(plan_dict.get('confidence', 0.1))
            )
        except Exception as e:
            logger.warning("Failed to construct AnalysisPlan from parsed dict: %s", e)
            return AnalysisPlan(
                domain="General",
                summary="Direct Analysis",
                steps=[AnalysisStep(1, llm_output[:200], "general_analysis", "Plan construction failed")],
                confidence=0.1
            )
             
def get_dynamic_planner() -> DynamicPlanner:
    """Return a new :class:`DynamicPlanner` instance."""
    return DynamicPlanner()


# ============================================================================
# Enterprise v2.0 — PlanningMetrics
# ============================================================================

import threading as _threading
from dataclasses import dataclass as _dataclass, field as _field
import datetime as _dt


@_dataclass
class PlanningMetrics:
    """Aggregate metrics for the dynamic planner.

    Designed for export to observability dashboards so operators
    can monitor planner health at a glance.

    Attributes:
        invocations: Total planner invocations.
        fallbacks: Times the planner fell back to a default plan.
        total_latency_seconds: Cumulative LLM planning latency.
        avg_confidence: Running average of plan confidence scores.
        last_updated: ISO-8601 timestamp of last metric update.

    .. versionadded:: 2.0
    """

    invocations: int = 0
    fallbacks: int = 0
    total_latency_seconds: float = 0.0
    avg_confidence: float = 0.0
    last_updated: str = _field(
        default_factory=lambda: _dt.datetime.now().isoformat()
    )

    def record(self, latency_s: float, confidence: float, is_fallback: bool = False) -> None:
        """Record a single planner invocation.

        Args:
            latency_s: Wall-clock seconds taken by the LLM call.
            confidence: Plan confidence score (0.0–1.0).
            is_fallback: Whether the fallback plan was used.
        """
        self.invocations += 1
        self.total_latency_seconds += latency_s
        if is_fallback:
            self.fallbacks += 1
        # Incremental mean
        self.avg_confidence += (confidence - self.avg_confidence) / self.invocations
        self.last_updated = _dt.datetime.now().isoformat()

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot."""
        return {
            "invocations": self.invocations,
            "fallbacks": self.fallbacks,
            "fallback_rate": round(self.fallbacks / max(self.invocations, 1), 4),
            "avg_latency_s": round(self.total_latency_seconds / max(self.invocations, 1), 3),
            "avg_confidence": round(self.avg_confidence, 4),
            "last_updated": self.last_updated,
        }
