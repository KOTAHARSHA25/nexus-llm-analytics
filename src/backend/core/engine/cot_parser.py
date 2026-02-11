"""Chain-of-Thought and Critic response parsers.

Provides lightweight, deterministic parsers for structured LLM output
(reasoning + final answer) and critic feedback extraction.

.. versionadded:: 2.0.0
   Added :class:`StreamingCoTParser`, :class:`MultiFormatParser`,
   :class:`ParseMetrics`, and :func:`get_parse_metrics`.

Backward Compatibility
----------------------
All v1.x public names remain at the same import paths.

Classes
-------
CoTParser
    Extracts ``[REASONING]`` / ``[OUTPUT]`` blocks from LLM responses.
CriticParser
    Determines validity and extracts issues from critic feedback.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional
import re

logger = logging.getLogger(__name__)

__all__ = [
    # v1.x (backward compatible)
    "CoTParser",
    "CriticFeedback",
    "CriticIssue",
    "CriticParser",
    "ParsedCoT",
    # v2.0 Enterprise additions
    "StreamingCoTParser",
    "MultiFormatParser",
    "ParseMetrics",
    "get_parse_metrics",
]

@dataclass
class CriticIssue:
    """A single issue raised by the critic LLM.

    Attributes:
        description: Human-readable explanation of the issue.
        location: Section of the response where the issue was found.
        severity: Issue severity level (HIGH, MEDIUM, or LOW).
        suggestion: Recommended fix or remediation.
    """

    description: str
    location: str
    severity: str  # HIGH, MEDIUM, LOW
    suggestion: str

@dataclass
class CriticFeedback:
    """Structured result returned by :class:`CriticParser`.

    Attributes:
        is_valid: True if the critic found no issues.
        issues: List of extracted :class:`CriticIssue` items.
        raw_response: Original unmodified critic LLM output.
        feedback: Optional human-readable feedback summary.
    """

    is_valid: bool
    issues: List[CriticIssue]
    raw_response: str
    feedback: str = ""

@dataclass
class ParsedCoT:
    """Parsed Chain-of-Thought response from an LLM.

    Attributes:
        is_valid: True if both reasoning and output were extracted.
        reasoning: Extracted reasoning section text.
        output: Extracted final output section text.
        error_message: Description of parsing failure, or None on success.
    """

    is_valid: bool
    reasoning: str
    output: str
    error_message: Optional[str] = None

class CoTParser:
    """Parser for Chain-of-Thought responses.

    Extracts ``[REASONING]`` and ``[OUTPUT]`` sections from structured
    LLM output using configurable delimiters.
    """

    def __init__(
        self,
        reasoning_start: str,
        reasoning_end: str,
        output_start: str,
        output_end: str,
    ) -> None:
        self.r_start = reasoning_start
        self.r_end = reasoning_end
        self.o_start = output_start
        self.o_end = output_end

    def parse(self, text: str) -> ParsedCoT:
        """Parse LLM output into reasoning and final output components.

        Args:
            text: Raw LLM response string.

        Returns:
            ParsedCoT with extracted reasoning and output sections.
        """
        if not text:
            return ParsedCoT(False, "", "", "Empty response")
            
        # Extract reasoning
        r_pattern = f"{re.escape(self.r_start)}(.*?){re.escape(self.r_end)}"
        r_match = re.search(r_pattern, text, re.DOTALL)
        
        # Extract output
        o_pattern = f"{re.escape(self.o_start)}(.*?){re.escape(self.o_end)}"
        o_match = re.search(o_pattern, text, re.DOTALL)
        
        reasoning = r_match.group(1).strip() if r_match else ""
        output = o_match.group(1).strip() if o_match else ""
        
        # Validation logic
        if not reasoning and not output:
            # Fallback for unstructured responses
            return ParsedCoT(True, "Implicit reasoning", text, None)
        
        if not reasoning:
            return ParsedCoT(False, "", output, "Missing reasoning section")
            
        if not output:
             return ParsedCoT(False, reasoning, "", "Missing output section")
             
        return ParsedCoT(True, reasoning, output, None)

    def extract_steps(self, reasoning: str) -> List[str]:
        """Extract numbered or bulleted steps from reasoning text.

        Args:
            reasoning: The reasoning section of a CoT response.

        Returns:
            List of step strings found (may be empty).
        """
        steps = []
        for line in reasoning.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line) or line.startswith('- '):
                steps.append(line)
        return steps

class CriticParser:
    """Parser for Critic responses.

    Heuristically determines whether critic output signals approval
    or lists issues, and extracts structured :class:`CriticIssue` items.
    """

    def parse(self, text: str) -> CriticFeedback:
        """Parse critic feedback into structured result.

        Args:
            text: Raw critic LLM response.

        Returns:
            CriticFeedback with validity flag and extracted issues.
        """
        # Heuristic parsing of critic output
        # Expecting "VALID" or "ISSUES: ..."
        
        is_valid = "NO ISSUES FOUND" in text.upper() or "VALID" in text.upper().split('\n')[0]
        
        issues = []
        if not is_valid:
            # Try to extract issues
            # Pattern: - Issue description (Severity: HIGH)
            lines = text.split('\n')
            for line in lines:
                if line.strip().startswith('-'):
                    desc = line.strip()[1:].strip()
                    severity = "MEDIUM"
                    if "HIGH" in desc.upper():
                        severity = "HIGH"
                    elif "LOW" in desc.upper():
                        severity = "LOW"
                    
                    issues.append(CriticIssue(
                        description=desc,
                        location="Unknown",
                        severity=severity,
                        suggestion="Review logic"
                    ))
        
        return CriticFeedback(
            is_valid=is_valid,
            issues=issues,
            raw_response=text,
            feedback=text
        )


# =============================================================================
# ENTERPRISE: STREAMING CoT PARSER
# =============================================================================

class StreamingCoTParser:
    """Incremental Chain-of-Thought parser for streaming LLM output.

    Processes tokens one at a time, yielding partial parse results as
    delimiters are detected.  This enables real-time UI updates while
    the LLM is still generating.

    .. code-block:: python

        parser = StreamingCoTParser("[REASONING]", "[/REASONING]",
                                    "[OUTPUT]", "[/OUTPUT]")
        for token in llm_stream:
            result = parser.feed(token)
            if result.reasoning:
                update_ui("reasoning", result.reasoning)
            if result.output:
                update_ui("output", result.output)

    Args:
        reasoning_start: Opening delimiter for reasoning section.
        reasoning_end: Closing delimiter for reasoning section.
        output_start: Opening delimiter for output section.
        output_end: Closing delimiter for output section.
    """

    def __init__(
        self,
        reasoning_start: str = "[REASONING]",
        reasoning_end: str = "[/REASONING]",
        output_start: str = "[OUTPUT]",
        output_end: str = "[/OUTPUT]",
    ) -> None:
        self.r_start = reasoning_start
        self.r_end = reasoning_end
        self.o_start = output_start
        self.o_end = output_end
        self._buffer = ""
        self._in_reasoning = False
        self._in_output = False
        self._reasoning_parts: List[str] = []
        self._output_parts: List[str] = []
        self._token_count = 0
        self._start_time = time.time()

    def feed(self, token: str) -> ParsedCoT:
        """Feed a single token and return the current parse state.

        Args:
            token: Next token from the LLM stream.

        Returns:
            :class:`ParsedCoT` reflecting the current accumulated state.
        """
        self._buffer += token
        self._token_count += 1

        # Check for delimiter transitions
        if self.r_start in self._buffer and not self._in_reasoning:
            self._in_reasoning = True
            self._buffer = self._buffer.split(self.r_start, 1)[1]

        if self.r_end in self._buffer and self._in_reasoning:
            content, self._buffer = self._buffer.split(self.r_end, 1)
            self._reasoning_parts.append(content)
            self._in_reasoning = False

        if self.o_start in self._buffer and not self._in_output:
            self._in_output = True
            self._buffer = self._buffer.split(self.o_start, 1)[1]

        if self.o_end in self._buffer and self._in_output:
            content, self._buffer = self._buffer.split(self.o_end, 1)
            self._output_parts.append(content)
            self._in_output = False

        # Accumulate content in active sections
        if self._in_reasoning:
            self._reasoning_parts.append(self._buffer)
            self._buffer = ""
        elif self._in_output:
            self._output_parts.append(self._buffer)
            self._buffer = ""

        reasoning = "".join(self._reasoning_parts).strip()
        output = "".join(self._output_parts).strip()

        return ParsedCoT(
            is_valid=bool(reasoning or output),
            reasoning=reasoning,
            output=output,
            error_message=None,
        )

    def finalize(self) -> ParsedCoT:
        """Finalize parsing after the stream ends.

        Returns:
            Final :class:`ParsedCoT` result.
        """
        # Flush any remaining buffered content
        if self._in_reasoning and self._buffer:
            self._reasoning_parts.append(self._buffer)
        elif self._in_output and self._buffer:
            self._output_parts.append(self._buffer)

        reasoning = "".join(self._reasoning_parts).strip()
        output = "".join(self._output_parts).strip()

        if not reasoning and not output:
            # Fallback: treat entire buffer as unstructured output
            full_text = "".join(self._reasoning_parts) + "".join(self._output_parts) + self._buffer
            return ParsedCoT(True, "Implicit reasoning", full_text.strip(), None)

        return ParsedCoT(
            is_valid=bool(reasoning and output),
            reasoning=reasoning,
            output=output,
            error_message="Missing output" if not output else ("Missing reasoning" if not reasoning else None),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return streaming parser statistics."""
        elapsed = time.time() - self._start_time
        return {
            "tokens_processed": self._token_count,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(self._token_count / max(elapsed, 0.001), 1),
            "reasoning_length": len("".join(self._reasoning_parts)),
            "output_length": len("".join(self._output_parts)),
        }

    def reset(self) -> None:
        """Reset the parser for reuse."""
        self._buffer = ""
        self._in_reasoning = False
        self._in_output = False
        self._reasoning_parts.clear()
        self._output_parts.clear()
        self._token_count = 0
        self._start_time = time.time()


# =============================================================================
# ENTERPRISE: MULTI-FORMAT PARSER
# =============================================================================

class MultiFormatParser:
    """Parser that auto-detects the CoT format used by the LLM.

    Supports multiple delimiter styles and falls back gracefully
    when the model does not follow any known format.

    Registered formats (tried in priority order):
    1. ``[REASONING]...[/REASONING] [OUTPUT]...[/OUTPUT]``
    2. ``### Reasoning\n...\n### Answer\n...``
    3. ``<think>...</think>`` (output = rest)
    4. Unstructured (full text as output)
    """

    _FORMATS = [
        # (name, reasoning_start, reasoning_end, output_start, output_end)
        ("bracketed", "[REASONING]", "[/REASONING]", "[OUTPUT]", "[/OUTPUT]"),
        ("markdown", "### Reasoning", "### Answer", "### Answer", "### End"),
    ]

    def parse(self, text: str) -> ParsedCoT:
        """Auto-detect format and parse.

        Args:
            text: Raw LLM response.

        Returns:
            :class:`ParsedCoT` parsed with the best-matching format.
        """
        if not text:
            return ParsedCoT(False, "", "", "Empty response")

        # Try each format in order
        for name, r_start, r_end, o_start, o_end in self._FORMATS:
            if r_start in text:
                parser = CoTParser(r_start, r_end, o_start, o_end)
                result = parser.parse(text)
                if result.is_valid:
                    return result

        # Try <think> tags (common in newer models)
        think_match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
        if think_match:
            return ParsedCoT(
                is_valid=True,
                reasoning=think_match.group(1).strip(),
                output=think_match.group(2).strip(),
            )

        # Fallback: entire text as output
        return ParsedCoT(True, "Implicit reasoning", text.strip(), None)


# =============================================================================
# ENTERPRISE: PARSE METRICS
# =============================================================================

class ParseMetrics:
    """Tracks parsing performance and format distribution.

    Records how often each format is used, parse success rates,
    and average parse times for monitoring and debugging.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_parses = 0
        self._successful_parses = 0
        self._format_counts: Dict[str, int] = defaultdict(int)
        self._parse_times: deque = deque(maxlen=500)
        self._failure_reasons: Dict[str, int] = defaultdict(int)

    def record_parse(
        self, success: bool, duration_ms: float,
        format_used: str = "unknown", error: Optional[str] = None,
    ) -> None:
        """Record a parse attempt.

        Args:
            success: Whether parsing succeeded.
            duration_ms: Parse time in milliseconds.
            format_used: Name of the format detected.
            error: Error message if parsing failed.
        """
        with self._lock:
            self._total_parses += 1
            if success:
                self._successful_parses += 1
            self._format_counts[format_used] += 1
            self._parse_times.append(duration_ms)
            if error:
                self._failure_reasons[error] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Return parse metrics summary."""
        with self._lock:
            times = list(self._parse_times)
            return {
                "total_parses": self._total_parses,
                "success_rate": round(
                    self._successful_parses / max(self._total_parses, 1), 4
                ),
                "avg_parse_time_ms": round(sum(times) / len(times), 2) if times else 0,
                "format_distribution": dict(self._format_counts),
                "top_failure_reasons": dict(
                    sorted(self._failure_reasons.items(), key=lambda x: -x[1])[:5]
                ),
            }


# =============================================================================
# SINGLETON
# =============================================================================

_parse_metrics: Optional[ParseMetrics] = None
_parse_metrics_lock = threading.Lock()


def get_parse_metrics() -> ParseMetrics:
    """Get or create the singleton :class:`ParseMetrics` (thread-safe)."""
    global _parse_metrics
    if _parse_metrics is None:
        with _parse_metrics_lock:
            if _parse_metrics is None:
                _parse_metrics = ParseMetrics()
    return _parse_metrics
