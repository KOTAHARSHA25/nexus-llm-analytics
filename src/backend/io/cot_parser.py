"""Chain-of-Thought Parser — Nexus LLM Analytics
================================================

Extracts and validates CoT reasoning from LLM responses.
Supports exact tag matching, fuzzy variations, and content-based
heuristic extraction with multi-strategy fallback.

Classes
-------
CoTParser
    Multi-strategy extraction of ``[REASONING]`` / ``[OUTPUT]`` sections.
CriticParser
    Structured parsing of critic model feedback with issue extraction.

v2.0 Enterprise Additions
-------------------------
* :class:`ParsingMetrics` — tracks parse strategy hit rates and latencies.
* :func:`get_cot_parser` — thread-safe singleton accessor.
* :func:`get_critic_parser` — thread-safe singleton accessor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

class CoTSection(Enum):
    """Enumeration of chain-of-thought response section types."""

    REASONING = "reasoning"
    OUTPUT = "output"

@dataclass
class ParsedCoT:
    """Structured CoT response"""
    reasoning: str
    output: str
    is_valid: bool
    error_message: Optional[str] = None
    raw_response: str = ""

@dataclass
class CriticIssue:
    """Individual issue found by critic"""
    description: str
    location: str
    severity: str  # LOW, MEDIUM, HIGH
    suggestion: str

@dataclass
class CriticFeedback:
    """Parsed critic response"""
    is_valid: bool
    issues: List[CriticIssue]
    raw_response: str
    
    @property
    def feedback(self) -> str:
        """Generate feedback text from issues for generator improvement"""
        if not self.issues:
            return ""
        
        feedback_parts = []
        for i, issue in enumerate(self.issues, 1):
            feedback_parts.append(f"Issue {i}: {issue.description}")
            if issue.location:
                feedback_parts.append(f"   Location: {issue.location}")
            if issue.severity:
                feedback_parts.append(f"   Severity: {issue.severity}")
            if issue.suggestion:
                feedback_parts.append(f"   Fix: {issue.suggestion}")
        
        return "\n".join(feedback_parts)

class CoTParser:
    """Parse and validate CoT-structured responses.

    Employs a four-stage parsing pipeline:

    1. **Exact match** — tag delimiters (``[REASONING]`` / ``[OUTPUT]``).
    2. **Fuzzy match** — angle brackets, markdown headings, colons, bold.
    3. **Content-based** — paragraph-level heuristic splitting.
    4. **Fallback** — entire response returned as ``output``.

    Thread Safety:
        Instance methods are stateless and therefore inherently thread-safe.
    """
    
    def __init__(self, reasoning_start="[REASONING]", reasoning_end="[/REASONING]",
                 output_start="[OUTPUT]", output_end="[/OUTPUT]"):
        self.reasoning_start = reasoning_start
        self.reasoning_end = reasoning_end
        self.output_start = output_start
        self.output_end = output_end
    
    def parse(self, response: str) -> ParsedCoT:
        """
        Extract reasoning and output sections from LLM response with fallback strategies
        
        Args:
            response: Raw LLM response string
            
        Returns:
            ParsedCoT object with extracted sections
        """
        # Strategy 1: Exact match (current approach)
        result = self._parse_exact(response)
        if result.is_valid:
            return result
        
        # Strategy 2: Fuzzy tag matching (handle variations)
        result = self._parse_fuzzy(response)
        if result.is_valid:
            return result
        
        # Strategy 3: Content-based extraction (no tags)
        result = self._parse_content_based(response)
        if result.is_valid:
            return result
        
        # Strategy 4: Fallback - return entire response as output
        return ParsedCoT(
            reasoning="Unable to extract structured reasoning",
            output=response.strip(),
            is_valid=False,
            error_message="All parsing strategies failed - using raw response",
            raw_response=response
        )
    
    def _parse_exact(self, response: str) -> ParsedCoT:
        """Original exact matching strategy"""
        # Extract reasoning section
        reasoning_pattern = f"{re.escape(self.reasoning_start)}(.*?){re.escape(self.reasoning_end)}"
        reasoning_match = re.search(reasoning_pattern, response, re.DOTALL | re.IGNORECASE)
        
        # Extract output section
        output_pattern = f"{re.escape(self.output_start)}(.*?){re.escape(self.output_end)}"
        output_match = re.search(output_pattern, response, re.DOTALL | re.IGNORECASE)
        
        # Validation
        if not reasoning_match or not output_match:
            return ParsedCoT(
                reasoning="",
                output="",
                is_valid=False,
                error_message="Missing expected tags",
                raw_response=response
            )
        
        reasoning_text = reasoning_match.group(1).strip()
        output_text = output_match.group(1).strip()
        
        # Validate non-empty
        if not reasoning_text or len(reasoning_text) < 50:
            return ParsedCoT(
                reasoning=reasoning_text,
                output=output_text,
                is_valid=False,
                error_message="Reasoning section too short (min 50 chars)",
                raw_response=response
            )
        
        return ParsedCoT(
            reasoning=reasoning_text,
            output=output_text,
            is_valid=True,
            error_message=None,
            raw_response=response
        )
    
    def _parse_fuzzy(self, response: str) -> ParsedCoT:
        """Match common tag variations (angle brackets, different cases, colons)"""
        fuzzy_patterns = [
            # Angle brackets
            (r'<reasoning>(.*?)</reasoning>', r'<output>(.*?)</output>'),
            (r'<REASONING>(.*?)</REASONING>', r'<OUTPUT>(.*?)</OUTPUT>'),
            # Markdown-style
            (r'##\s*REASONING\s*\n(.*?)(?=##\s*OUTPUT)', r'##\s*OUTPUT\s*\n(.*?)$'),
            # Colon-based
            (r'REASONING:(.*?)(?=OUTPUT:)', r'OUTPUT:(.*?)$'),
            (r'Reasoning:(.*?)(?=Answer:|Output:)', r'(?:Answer|Output):(.*?)$'),
            # Bold markdown
            (r'\*\*REASONING\*\*(.*?)(?=\*\*OUTPUT\*\*)', r'\*\*OUTPUT\*\*(.*?)$'),
        ]
        
        for reasoning_pattern, output_pattern in fuzzy_patterns:
            reasoning_match = re.search(reasoning_pattern, response, re.DOTALL | re.IGNORECASE)
            output_match = re.search(output_pattern, response, re.DOTALL | re.IGNORECASE)
            
            if reasoning_match and output_match:
                reasoning_text = reasoning_match.group(1).strip()
                output_text = output_match.group(1).strip()
                
                if reasoning_text and len(reasoning_text) >= 50 and output_text:
                    return ParsedCoT(
                        reasoning=reasoning_text,
                        output=output_text,
                        is_valid=True,
                        error_message=None,
                        raw_response=response
                    )
        
        return ParsedCoT(
            reasoning="",
            output="",
            is_valid=False,
            error_message="Fuzzy matching failed",
            raw_response=response
        )
    
    def _parse_content_based(self, response: str) -> ParsedCoT:
        """Extract reasoning and output based on content structure (no tags required)"""
        # Split by double newlines to find paragraphs
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return ParsedCoT(
                reasoning="",
                output="",
                is_valid=False,
                error_message="Insufficient structure for content-based parsing",
                raw_response=response
            )
        
        # Heuristic: Reasoning keywords in early paragraphs, conclusion/answer in later
        reasoning_keywords = ['because', 'therefore', 'first', 'second', 'step', 'analyze', 'consider']
        output_keywords = ['result', 'answer', 'conclusion', 'therefore', 'summary', 'final']
        
        reasoning_paras = []
        output_paras = []
        
        for i, para in enumerate(paragraphs):
            para_lower = para.lower()
            
            # First 2/3 likely reasoning
            if i < len(paragraphs) * 0.66:
                if any(kw in para_lower for kw in reasoning_keywords) or len(para) > 100:
                    reasoning_paras.append(para)
            else:
                # Last 1/3 likely output
                if any(kw in para_lower for kw in output_keywords) or i == len(paragraphs) - 1:
                    output_paras.append(para)
        
        reasoning_text = '\n\n'.join(reasoning_paras).strip()
        output_text = '\n\n'.join(output_paras).strip()
        
        # Fallback: use last paragraph as output if nothing identified
        if not output_text and paragraphs:
            output_text = paragraphs[-1]
        
        # Fallback: use all but last as reasoning
        if not reasoning_text and len(paragraphs) > 1:
            reasoning_text = '\n\n'.join(paragraphs[:-1])
        
        if reasoning_text and len(reasoning_text) >= 50 and output_text:
            return ParsedCoT(
                reasoning=reasoning_text,
                output=output_text,
                is_valid=True,
                error_message=None,
                raw_response=response
            )
        
        return ParsedCoT(
            reasoning="",
            output="",
            is_valid=False,
            error_message="Content-based parsing failed",
            raw_response=response
        )
    
    def extract_steps(self, reasoning: str) -> List[str]:
        """Extract individual reasoning steps"""
        # Look for "Step N:" patterns
        step_pattern = r"Step\s+\d+:.*?(?=Step\s+\d+:|$)"
        steps = re.findall(step_pattern, reasoning, re.DOTALL | re.IGNORECASE)
        return [step.strip() for step in steps if step.strip()]

class CriticParser:
    """Parse critic model feedback into structured issue records.

    Extracts ``[VALID]`` markers, numbered issues with location,
    severity, and suggestion fields from free-form LLM critic output.

    Thread Safety:
        Stateless — safe for concurrent use.
    """
    
    def parse(self, response: str) -> CriticFeedback:
        """
        Parse critic feedback for issues
        
        Returns:
            CriticFeedback with validation status and issues list
        """
        # Check for [VALID] marker
        if "[VALID]" in response.upper():
            return CriticFeedback(
                is_valid=True,
                issues=[],
                raw_response=response
            )
        
        # Extract issues
        issues = []
        issue_pattern = r"Issue\s+\d+:(.*?)(?=Issue\s+\d+:|$)"
        issue_matches = re.findall(issue_pattern, response, re.DOTALL)
        
        for issue_text in issue_matches:
            # Extract components
            location_match = re.search(r"Location:\s*(.+?)(?:\n|$)", issue_text)
            severity_match = re.search(r"Severity:\s*(LOW|MEDIUM|HIGH)", issue_text, re.IGNORECASE)
            suggestion_match = re.search(r"Suggestion:\s*(.+?)(?:\n\n|$)", issue_text, re.DOTALL)
            
            # Get first line as description
            description = issue_text.split('\n')[0].strip()
            
            issues.append(CriticIssue(
                description=description,
                location=location_match.group(1).strip() if location_match else "Unknown",
                severity=severity_match.group(1).upper() if severity_match else "MEDIUM",
                suggestion=suggestion_match.group(1).strip() if suggestion_match else "Review and correct"
            ))
        
        return CriticFeedback(
            is_valid=False,
            issues=issues,
            raw_response=response
        )


# =====================================================================
# v2.0 Enterprise Additions — appended; all v1.x code is unchanged
# =====================================================================

import threading
import time
from collections import Counter


@dataclass
class ParsingMetrics:
    """Tracks CoT parse strategy hit-rates and latencies.

    Attributes:
        strategy_hits: Counter mapping strategy name → success count.
        total_parses: Total ``parse()`` invocations observed.
        total_latency_ms: Cumulative parsing time in milliseconds.

    v2.0 Enterprise Addition.
    """

    strategy_hits: Counter = field(default_factory=Counter)
    total_parses: int = 0
    total_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    def record(self, strategy: str, latency_ms: float) -> None:
        """Record a successful parse attributed to *strategy*."""
        self.strategy_hits[strategy] += 1
        self.total_parses += 1
        self.total_latency_ms += latency_ms

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot."""
        return {
            "strategy_hits": dict(self.strategy_hits),
            "total_parses": self.total_parses,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_latency_ms": round(
                self.total_latency_ms / self.total_parses, 2
            )
            if self.total_parses
            else 0.0,
        }


# -- Singleton accessors (double-checked locking) ----------------------

_cot_parser_instance: Optional[CoTParser] = None
_cot_parser_lock = threading.Lock()


def get_cot_parser(**kwargs) -> CoTParser:
    """Return the process-wide :class:`CoTParser` singleton.

    Keyword arguments are forwarded to the constructor on first call.

    Thread Safety:
        Uses double-checked locking for safe lazy initialisation.
    """
    global _cot_parser_instance
    if _cot_parser_instance is None:
        with _cot_parser_lock:
            if _cot_parser_instance is None:
                _cot_parser_instance = CoTParser(**kwargs)
    return _cot_parser_instance


_critic_parser_instance: Optional[CriticParser] = None
_critic_parser_lock = threading.Lock()


def get_critic_parser() -> CriticParser:
    """Return the process-wide :class:`CriticParser` singleton.

    Thread Safety:
        Uses double-checked locking for safe lazy initialisation.
    """
    global _critic_parser_instance
    if _critic_parser_instance is None:
        with _critic_parser_lock:
            if _critic_parser_instance is None:
                _critic_parser_instance = CriticParser()
    return _critic_parser_instance
