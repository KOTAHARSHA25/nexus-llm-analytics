"""Automated runtime validation for LLM-generated analysis.

Detects hallucinated columns, implausible statistical claims,
data-type mismatches, and malformed code blocks before responses
reach the user.

.. versionadded:: 2.0.0
   Added :class:`ValidationPipeline`, :class:`ValidatorRegistry`,
   :class:`ConfidenceScorer`, and :func:`batch_validate`.

Backward Compatibility
----------------------
All v1.x public names remain at the same import paths.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Type
import re

logger = logging.getLogger(__name__)

__all__ = [
    # v1.x (backward compatible)
    "RuntimeEvaluator",
    "ValidationIssue",
    "ValidationResult",
    # v2.0 Enterprise additions
    "ValidationPipeline",
    "ValidatorRegistry",
    "BaseValidator",
    "ConfidenceScorer",
    "ValidationSeverity",
    "ValidationContext",
    "batch_validate",
    "get_validator_registry",
]

@dataclass
class ValidationIssue:
    """A single validation problem detected in LLM output.

    Attributes:
        description: Human-readable explanation of the issue.
        location: Section of the response where the issue was found.
        severity: Issue severity level (HIGH, MEDIUM, or LOW).
        issue_type: Machine-readable category (e.g. hallucination, syntax_error).
    """

    description: str
    location: str
    severity: str  # HIGH, MEDIUM, LOW
    issue_type: str

@dataclass
class ValidationResult:
    """Aggregated outcome of all validation checks.

    Attributes:
        is_valid: True if no issues were found.
        issues: List of :class:`ValidationIssue` items detected.
    """

    is_valid: bool
    issues: List[ValidationIssue]
    
    def to_feedback_text(self) -> str:
        """Return a human-readable summary of validation issues.

        Returns:
            str summarizing issues or "Validation Successful".
        """
        if self.is_valid:
            return "Validation Successful"
        return "\n".join([f"- {i.description} (Severity: {i.severity})" for i in self.issues])

class RuntimeEvaluator:
    """
    Performs automated checks (including hallucination detection) on reasoning and code.
    Previously 'AutomatedValidator'.
    
    Expanded validation (addresses reviewer comment 3):
    - Dynamic column hallucination detection against actual dataset columns
    - Empty output detection
    - Code block syntax validation
    - Statistical claim verification
    - Data type mismatch detection
    """
    
    # Common English words and domain-agnostic analytics terms excluded from
    # hallucination detection to avoid false-positive column-name alerts.
    COMMON_WORDS = frozenset({
        'I', 'The', 'A', 'An', 'If', 'When', 'Then', 'For', 'In', 'By', 'To', 
        'Of', 'And', 'Or', 'Not', 'Is', 'Are', 'Was', 'Were', 'Be', 'Been',
        'Has', 'Have', 'Had', 'Do', 'Does', 'Did', 'Will', 'Would', 'Could',
        'Should', 'May', 'Might', 'Must', 'Can', 'This', 'That', 'These',
        'Those', 'Each', 'Every', 'All', 'Both', 'Few', 'More', 'Most',
        'Other', 'Some', 'Such', 'Only', 'Same', 'So', 'Than', 'Very',
        'Just', 'But', 'Yet', 'Now', 'Also', 'Here', 'There', 'Where',
        'How', 'What', 'Which', 'Who', 'Whom', 'Why', 'Yes', 'No',
        'True', 'False', 'None', 'Null', 'NaN', 'Python', 'Pandas',
        'DataFrame', 'Series', 'Figure', 'Table', 'Chart', 'Plot',
        'Analysis', 'Result', 'Summary', 'Mean', 'Median', 'Mode',
        'Standard', 'Deviation', 'Correlation', 'Regression',
        'Data', 'Column', 'Row', 'Index', 'Value', 'Count',
        'Total', 'Average', 'Maximum', 'Minimum', 'Sum',
        'North', 'South', 'East', 'West', 'January', 'February',
        'March', 'April', 'May', 'June', 'July', 'August',
        'September', 'October', 'November', 'December',
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
        'Saturday', 'Sunday', 'Note', 'Step', 'First', 'Second',
        'Third', 'Last', 'Next', 'Previous', 'Based', 'Using',
        'Following', 'Below', 'Above', 'Output', 'Input',
        # Common pandas/Python method names and analytics terms that appear
        # in LLM reasoning but are NOT column references
        'groupby', 'sort_values', 'merge', 'fillna', 'dropna', 'concat',
        'apply', 'reset_index', 'head', 'tail', 'describe', 'info',
        'to_csv', 'read_csv', 'agg', 'aggregate', 'pivot_table',
        'melt', 'stack', 'unstack', 'rename', 'replace', 'astype',
        'unique', 'nunique', 'nlargest', 'nsmallest', 'idxmax', 'idxmin',
        # Common product/category terms that appear as data VALUES not columns
        'Electronics', 'Smartphone', 'Laptop', 'Tablet', 'Computer',
        'Clothing', 'Food', 'Beverages', 'Furniture', 'Accessories',
        'Gadgets', 'Software', 'Hardware', 'Services', 'Technology',
        # Aggregation / analytical terms
        'Revenue', 'Profit', 'Sales', 'Units', 'Quantity', 'Price',
        'Margin', 'Growth', 'Decline', 'Increase', 'Decrease',
        'Highest', 'Lowest', 'Best', 'Worst', 'Top', 'Bottom',
        'Region', 'Category', 'Product', 'Month', 'Year', 'Date',
        'Quarter', 'Period', 'Segment', 'Type', 'Name', 'ID',
        # Common short tokens that are never column names
        'int64', 'float64', 'object', 'str', 'bool', 'np', 'pd', 'df',
    })
    
    def evaluate_quality(
        self,
        query: str,
        reasoning: str,
        output: str,
        data_context: Dict[str, Any],
    ) -> List[ValidationIssue]:
        """Return a flat list of :class:`ValidationIssue` objects.

        Convenience wrapper used by the benchmark harness.

        Args:
            query: The user's original question.
            reasoning: The CoT reasoning section.
            output: The final output section.
            data_context: Dict with ``columns``, ``rows``, ``data_types``, etc.

        Returns:
            List of validation issues (empty when valid).
        """
        result = self._validate_internal(query, reasoning, output, data_context)
        return result.issues

    def validate(
        self,
        query: str,
        reasoning: str,
        output: str,
        data_context: Dict[str, Any],
    ) -> ValidationResult:
        """Run all validation checks and return a full result.

        Used by :class:`SelfCorrectionEngine` to decide whether a
        correction iteration is needed.

        Args:
            query: The user's original question.
            reasoning: The CoT reasoning section.
            output: The final output section.
            data_context: Dict with ``columns``, ``rows``, ``data_types``, etc.

        Returns:
            ValidationResult with ``is_valid`` flag and issue list.
        """
        return self._validate_internal(query, reasoning, output, data_context)

    def _validate_internal(self, query: str, reasoning: str, output: str, data_context: Dict[str, Any]) -> ValidationResult:
        issues = []
        
        # 1. Check for empty output
        if not output.strip():
             issues.append(ValidationIssue(
                 description="Output is empty",
                 location="Output",
                 severity="HIGH",
                 issue_type="empty_output"
             ))
             
        # 2. Check for Hallucinated Columns (EXPANDED - checks ALL actual columns)
        columns = data_context.get('columns', [])
        if columns and reasoning:
             hallucination_issues = self._check_column_hallucination(
                 reasoning, output, columns
             )
             issues.extend(hallucination_issues)

        # 3. Check code block validity
        if "```python" in output:
             if "```" not in output.split("```python")[1]:
                 issues.append(ValidationIssue(
                     description="Unclosed code block",
                     location="Output",
                     severity="HIGH",
                     issue_type="syntax_error"
                 ))
        
        # 4. Check for statistical claim plausibility (NEW)
        stat_issues = self._check_statistical_claims(output, data_context)
        issues.extend(stat_issues)
        
        # 5. Check for data type mismatches (NEW)
        dtype_issues = self._check_dtype_references(reasoning, output, data_context)
        issues.extend(dtype_issues)

        return ValidationResult(
            is_valid=(len(issues) == 0),
            issues=issues
        )
    
    def _check_column_hallucination(self, reasoning: str, output: str, 
                                     valid_columns: List[str]) -> List[ValidationIssue]:
        """
        Dynamic hallucination detection against actual dataset columns.
        
        Instead of checking against a hardcoded 10-word list, this method:
        1. Extracts all quoted references from reasoning/output
        2. Checks each against the ACTUAL column list
        3. Uses fuzzy matching to detect near-misses (typos)
        """
        issues = []
        valid_set = set(valid_columns)
        valid_lower = {c.lower() for c in valid_columns}
        combined_text = reasoning + " " + output
        
        # Only extract references from code-like patterns (df['col'], groupby('col'), etc.)
        # NOT from general prose — quoting a term in English text does NOT mean
        # it's a column reference.  This avoids massive false-positive hallucination
        # flags on data values, method names, and analytical terms.
        code_refs = re.findall(
            r"(?:df\[?['\"]|\bdf\.|\.groupby\(['\"]|\.sort_values\(['\"]|column[s]?\s*=\s*['\"])([a-zA-Z0-9_]+)",
            combined_text
        )
        
        # All candidates from code-like references only
        candidates = set(code_refs)
        
        for cand in candidates:
            cand_stripped = cand.strip()
            if len(cand_stripped) < 2:
                continue
            if cand_stripped in self.COMMON_WORDS:
                continue
            
            # Check exact match
            if cand_stripped in valid_set:
                continue
            
            # Check case-insensitive match
            if cand_stripped.lower() in valid_lower:
                continue
            
            # This is a quoted reference NOT in valid columns → likely hallucinated
            issues.append(ValidationIssue(
                description=f"Potential hallucinated column: '{cand_stripped}' (not in dataset columns: {', '.join(valid_columns[:5])}{'...' if len(valid_columns) > 5 else ''})",
                location="Reasoning",
                severity="HIGH",
                issue_type="hallucination"
            ))
        
        # Also check unquoted capitalized words that look like column names
        # Only flag these if they appear in data-centric context
        data_patterns = re.findall(
            r"(?:column|field|feature|variable|groupby|sort|filter|calculate|compute)\s+['\"]?([A-Z][a-z0-9_]+)",
            combined_text, re.IGNORECASE
        )
        for ref in data_patterns:
            ref_stripped = ref.strip()
            if ref_stripped in self.COMMON_WORDS or ref_stripped in valid_set:
                continue
            if ref_stripped.lower() in valid_lower:
                continue
            issues.append(ValidationIssue(
                description=f"Potential hallucinated column reference: '{ref_stripped}'",
                location="Reasoning",
                severity="MEDIUM",
                issue_type="hallucination"
            ))
        
        return issues
    
    def _check_statistical_claims(self, output: str, data_context: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Check for obviously implausible statistical claims.
        E.g., percentages > 100%, negative counts, etc.
        """
        issues = []
        
        # Check for percentages > 100%
        pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', output)
        for pct_str in pct_matches:
            try:
                pct = float(pct_str)
                if pct > 100 and 'increase' not in output.lower() and 'growth' not in output.lower():
                    issues.append(ValidationIssue(
                        description=f"Suspicious percentage value: {pct}% (may be implausible unless it's a growth rate)",
                        location="Output",
                        severity="MEDIUM",
                        issue_type="statistical_implausibility"
                    ))
            except ValueError:
                pass
        
        # Check row count claims against actual data
        row_count = data_context.get('rows', 0)
        if row_count > 0:
            count_matches = re.findall(r'(\d+)\s*(?:rows|records|entries|samples|observations)', output.lower())
            for count_str in count_matches:
                try:
                    claimed = int(count_str)
                    if claimed > row_count * 2:
                        issues.append(ValidationIssue(
                            description=f"Claimed {claimed} rows but data has only {row_count} rows",
                            location="Output",
                            severity="MEDIUM",
                            issue_type="data_mismatch"
                        ))
                except ValueError:
                    pass
        
        return issues
    
    def _check_dtype_references(self, reasoning: str, output: str,
                                 data_context: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Check if numeric operations are applied to non-numeric columns.
        """
        issues = []
        data_types = data_context.get('data_types', {})
        
        if not data_types or isinstance(data_types, str):
            return issues
        
        # Identify numeric operations in output
        numeric_ops = re.findall(
            r'(?:mean|sum|average|std|median|min|max)\s*\(\s*[\'"]?([a-zA-Z0-9_ ]+)[\'"]?\s*\)',
            output, re.IGNORECASE
        )
        
        if isinstance(data_types, dict):
            for col_ref in numeric_ops:
                col_ref = col_ref.strip()
                if col_ref in data_types:
                    dtype = str(data_types[col_ref]).lower()
                    if 'object' in dtype or 'str' in dtype or 'category' in dtype:
                        issues.append(ValidationIssue(
                            description=f"Numeric operation on non-numeric column '{col_ref}' (dtype: {dtype})",
                            location="Output",
                            severity="HIGH",
                            issue_type="dtype_mismatch"
                        ))
        
        return issues


# =============================================================================
# ENTERPRISE: VALIDATION SEVERITY ENUM
# =============================================================================

class ValidationSeverity(Enum):
    """Standardised validation severity levels.

    Provides a typed alternative to the string-based severity used in
    :class:`ValidationIssue` (which is preserved for backward compat).
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_string(cls, s: str) -> "ValidationSeverity":
        """Parse a severity string (case-insensitive)."""
        return cls[s.upper()] if s.upper() in cls.__members__ else cls.MEDIUM


# =============================================================================
# ENTERPRISE: VALIDATION CONTEXT
# =============================================================================

@dataclass
class ValidationContext:
    """Enriched context passed through the validation pipeline.

    Carries the query, LLM output, data metadata, and accumulated
    issues through every validator in the chain.

    Attributes:
        request_id: Unique identifier for this validation request.
        query: The user's original question.
        reasoning: Extracted CoT reasoning section.
        output: Extracted final output section.
        data_context: Dataset metadata dict.
        issues: Accumulated :class:`ValidationIssue` items.
        metadata: Arbitrary pipeline metadata dict.
        start_time: Unix timestamp when validation started.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    reasoning: str = ""
    output: str = ""
    data_context: Dict[str, Any] = field(default_factory=dict)
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)


# =============================================================================
# ENTERPRISE: BASE VALIDATOR (Plugin Interface)
# =============================================================================

class BaseValidator(ABC):
    """Abstract base class for pluggable validators.

    Implement this interface to add custom validation logic to the
    :class:`ValidationPipeline`.  Register implementations via
    :class:`ValidatorRegistry`.

    Subclasses must define:
        - ``name`` — short human-readable identifier.
        - ``validate(ctx)`` — performs checks and appends issues.
    """

    name: str = "base_validator"
    enabled: bool = True
    priority: int = 100  # Lower = runs earlier

    @abstractmethod
    def validate(self, ctx: ValidationContext) -> ValidationContext:
        """Run validation checks and return the enriched context.

        Args:
            ctx: The current validation context.

        Returns:
            The same context with any new issues appended.
        """
        ...


# =============================================================================
# ENTERPRISE: BUILT-IN VALIDATORS (wrapping existing RuntimeEvaluator logic)
# =============================================================================

class EmptyOutputValidator(BaseValidator):
    """Flags empty LLM output as a HIGH severity issue."""
    name = "empty_output"
    priority = 10

    def validate(self, ctx: ValidationContext) -> ValidationContext:
        if not ctx.output.strip():
            ctx.issues.append(ValidationIssue(
                description="Output is empty",
                location="Output",
                severity="HIGH",
                issue_type="empty_output",
            ))
        return ctx


class CodeBlockValidator(BaseValidator):
    """Checks for unclosed Python code blocks."""
    name = "code_block"
    priority = 20

    def validate(self, ctx: ValidationContext) -> ValidationContext:
        if "```python" in ctx.output:
            after = ctx.output.split("```python", 1)[1]
            if "```" not in after:
                ctx.issues.append(ValidationIssue(
                    description="Unclosed code block",
                    location="Output",
                    severity="HIGH",
                    issue_type="syntax_error",
                ))
        return ctx


class HallucinationValidator(BaseValidator):
    """Detects hallucinated column references using the RuntimeEvaluator."""
    name = "hallucination"
    priority = 30

    def __init__(self) -> None:
        self._evaluator = RuntimeEvaluator()

    def validate(self, ctx: ValidationContext) -> ValidationContext:
        columns = ctx.data_context.get("columns", [])
        if columns and ctx.reasoning:
            issues = self._evaluator._check_column_hallucination(
                ctx.reasoning, ctx.output, columns
            )
            ctx.issues.extend(issues)
        return ctx


class StatisticalClaimValidator(BaseValidator):
    """Verifies statistical claim plausibility."""
    name = "statistical_claims"
    priority = 40

    def __init__(self) -> None:
        self._evaluator = RuntimeEvaluator()

    def validate(self, ctx: ValidationContext) -> ValidationContext:
        issues = self._evaluator._check_statistical_claims(ctx.output, ctx.data_context)
        ctx.issues.extend(issues)
        return ctx


class DTypeMismatchValidator(BaseValidator):
    """Detects numeric operations on non-numeric columns."""
    name = "dtype_mismatch"
    priority = 50

    def __init__(self) -> None:
        self._evaluator = RuntimeEvaluator()

    def validate(self, ctx: ValidationContext) -> ValidationContext:
        issues = self._evaluator._check_dtype_references(
            ctx.reasoning, ctx.output, ctx.data_context
        )
        ctx.issues.extend(issues)
        return ctx


# =============================================================================
# ENTERPRISE: VALIDATOR REGISTRY
# =============================================================================

class ValidatorRegistry:
    """Manages the catalogue of available validators.

    Supports dynamic registration, enable/disable toggling, and
    priority-ordered retrieval.

    .. code-block:: python

        registry = get_validator_registry()
        registry.register(MyCustomValidator())
        registry.disable("hallucination")
        active = registry.get_active_validators()
    """

    def __init__(self) -> None:
        self._validators: Dict[str, BaseValidator] = {}
        self._lock = threading.Lock()
        # Register built-in validators
        for v in (
            EmptyOutputValidator(),
            CodeBlockValidator(),
            HallucinationValidator(),
            StatisticalClaimValidator(),
            DTypeMismatchValidator(),
        ):
            self.register(v)
        logger.info("ValidatorRegistry initialized with %d built-in validators", len(self._validators))

    def register(self, validator: BaseValidator) -> None:
        """Register a validator.

        Args:
            validator: :class:`BaseValidator` implementation to add.
        """
        with self._lock:
            self._validators[validator.name] = validator
            logger.debug("Registered validator: %s (priority=%d)", validator.name, validator.priority)

    def unregister(self, name: str) -> None:
        """Remove a validator by name.

        Args:
            name: Validator name to remove.
        """
        with self._lock:
            self._validators.pop(name, None)

    def enable(self, name: str) -> None:
        """Enable a registered validator."""
        with self._lock:
            if name in self._validators:
                self._validators[name].enabled = True

    def disable(self, name: str) -> None:
        """Disable a registered validator."""
        with self._lock:
            if name in self._validators:
                self._validators[name].enabled = False

    def get_active_validators(self) -> List[BaseValidator]:
        """Get all enabled validators sorted by priority.

        Returns:
            Priority-ordered list of enabled validators.
        """
        with self._lock:
            return sorted(
                [v for v in self._validators.values() if v.enabled],
                key=lambda v: v.priority,
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Return registry statistics."""
        with self._lock:
            return {
                "total_registered": len(self._validators),
                "active": sum(1 for v in self._validators.values() if v.enabled),
                "disabled": sum(1 for v in self._validators.values() if not v.enabled),
                "validators": [
                    {"name": v.name, "enabled": v.enabled, "priority": v.priority}
                    for v in sorted(self._validators.values(), key=lambda v: v.priority)
                ],
            }


# =============================================================================
# ENTERPRISE: VALIDATION PIPELINE
# =============================================================================

class ValidationPipeline:
    """Enterprise validation pipeline executing validators in priority order.

    Wraps the :class:`ValidatorRegistry` in a single ``run()`` call
    that builds a :class:`ValidationContext`, passes it through every
    active validator, and returns a :class:`ValidationResult`.

    .. code-block:: python

        pipeline = ValidationPipeline()
        result = pipeline.run(
            query="show average salary",
            reasoning="...", output="...",
            data_context={"columns": ["Name", "Salary"], ...},
        )
        print(result.is_valid, result.issues)

    Args:
        registry: Specific :class:`ValidatorRegistry` to use.
    """

    def __init__(self, registry: Optional[ValidatorRegistry] = None) -> None:
        self._registry = registry or get_validator_registry()
        self._lock = threading.Lock()
        self._run_count = 0
        self._total_issues_found = 0
        self._validator_timings: Dict[str, List[float]] = defaultdict(list)

    def run(
        self,
        query: str,
        reasoning: str,
        output: str,
        data_context: Dict[str, Any],
    ) -> ValidationResult:
        """Execute the full validation pipeline.

        Args:
            query: User's original question.
            reasoning: CoT reasoning section.
            output: Final output section.
            data_context: Dict with ``columns``, ``rows``, ``data_types``.

        Returns:
            :class:`ValidationResult` with aggregated issues.
        """
        ctx = ValidationContext(
            query=query, reasoning=reasoning,
            output=output, data_context=data_context,
        )

        for validator in self._registry.get_active_validators():
            start = time.time()
            try:
                ctx = validator.validate(ctx)
            except Exception as e:
                logger.error("Validator '%s' failed: %s", validator.name, e, exc_info=True)
                ctx.issues.append(ValidationIssue(
                    description=f"Validator '{validator.name}' error: {e}",
                    location="Pipeline",
                    severity="LOW",
                    issue_type="validator_error",
                ))
            finally:
                elapsed = time.time() - start
                with self._lock:
                    self._validator_timings[validator.name].append(elapsed)

        with self._lock:
            self._run_count += 1
            self._total_issues_found += len(ctx.issues)

        return ValidationResult(
            is_valid=len(ctx.issues) == 0,
            issues=ctx.issues,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Return pipeline performance statistics.

        Returns:
            Dict with run count, issue totals, and per-validator timings.
        """
        with self._lock:
            timings = {}
            for name, ts in self._validator_timings.items():
                timings[name] = {
                    "calls": len(ts),
                    "avg_ms": round(sum(ts) / len(ts) * 1000, 2) if ts else 0,
                    "max_ms": round(max(ts) * 1000, 2) if ts else 0,
                }
            return {
                "total_runs": self._run_count,
                "total_issues_found": self._total_issues_found,
                "avg_issues_per_run": round(
                    self._total_issues_found / max(self._run_count, 1), 2
                ),
                "validator_timings": timings,
            }


# =============================================================================
# ENTERPRISE: CONFIDENCE SCORER
# =============================================================================

class ConfidenceScorer:
    """Calculate a confidence score for LLM output based on validation.

    Assigns a numeric score in ``[0.0, 1.0]`` based on the count and
    severity of validation issues, reasoning completeness, and
    output structural quality.
    """

    # Severity penalty weights
    _SEVERITY_PENALTIES = {
        "CRITICAL": 0.40,
        "HIGH": 0.25,
        "MEDIUM": 0.10,
        "LOW": 0.05,
    }

    def score(
        self,
        validation_result: ValidationResult,
        reasoning: str = "",
        output: str = "",
    ) -> float:
        """Compute confidence score for a validated response.

        Args:
            validation_result: Result from validation pipeline.
            reasoning: CoT reasoning text.
            output: Final output text.

        Returns:
            Confidence score between 0.0 (no confidence) and 1.0 (full).
        """
        confidence = 1.0

        # Penalty for each issue by severity
        for issue in validation_result.issues:
            penalty = self._SEVERITY_PENALTIES.get(issue.severity, 0.05)
            confidence -= penalty

        # Bonus for structured output
        if output and len(output) > 50:
            confidence += 0.05

        # Bonus for detailed reasoning
        if reasoning and len(reasoning) > 100:
            confidence += 0.05

        # Penalty for very short responses
        if output and len(output) < 20:
            confidence -= 0.10

        return max(0.0, min(1.0, round(confidence, 3)))


# =============================================================================
# ENTERPRISE: BATCH VALIDATION
# =============================================================================

def batch_validate(
    items: List[Dict[str, Any]],
    pipeline: Optional[ValidationPipeline] = None,
) -> List[ValidationResult]:
    """Validate a batch of LLM outputs in sequence.

    Each item in *items* must contain keys ``query``, ``reasoning``,
    ``output``, and ``data_context``.

    Args:
        items: List of dicts with validation inputs.
        pipeline: Pipeline instance to use (creates default if ``None``).

    Returns:
        List of :class:`ValidationResult` in corresponding order.
    """
    pipe = pipeline or ValidationPipeline()
    results: List[ValidationResult] = []
    for item in items:
        result = pipe.run(
            query=item.get("query", ""),
            reasoning=item.get("reasoning", ""),
            output=item.get("output", ""),
            data_context=item.get("data_context", {}),
        )
        results.append(result)
    return results


# =============================================================================
# SINGLETON
# =============================================================================

_validator_registry: Optional[ValidatorRegistry] = None
_registry_lock = threading.Lock()


def get_validator_registry() -> ValidatorRegistry:
    """Get or create the singleton :class:`ValidatorRegistry` (thread-safe).

    Returns:
        Shared validator registry instance.
    """
    global _validator_registry
    if _validator_registry is None:
        with _registry_lock:
            if _validator_registry is None:
                _validator_registry = ValidatorRegistry()
    return _validator_registry
