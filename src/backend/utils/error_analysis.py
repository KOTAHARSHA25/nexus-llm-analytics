"""
Error Analysis and Failure Categorization Module
=================================================

This module provides comprehensive error analysis capabilities for
understanding failure patterns and improving system reliability.

Features:
- Error categorization and classification
- Failure pattern detection
- Root cause analysis
- Error clustering
- Actionable recommendations
"""

import re
import math
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from enum import Enum


class ErrorCategory(Enum):
    """Categories of errors in LLM systems."""
    FACTUAL = "factual"           # Incorrect facts
    INCOMPLETE = "incomplete"      # Missing information
    IRRELEVANT = "irrelevant"      # Off-topic response
    FORMATTING = "formatting"      # Wrong format/structure
    HALLUCINATION = "hallucination"  # Made up information
    AMBIGUOUS = "ambiguous"        # Unclear response
    CONTRADICTORY = "contradictory"  # Self-contradicting
    OUTDATED = "outdated"          # Stale information
    TRUNCATED = "truncated"        # Cut off response
    TIMEOUT = "timeout"            # Response timeout
    MODEL_ERROR = "model_error"    # API/model failure
    UNKNOWN = "unknown"


class Severity(Enum):
    """Error severity levels."""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    INFO = 0


@dataclass
class ErrorInstance:
    """Represents a single error instance."""
    error_id: str
    query: str
    response: str
    expected: Optional[str]
    category: ErrorCategory
    severity: Severity
    details: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ErrorPattern:
    """Represents a recurring error pattern."""
    pattern_id: str
    category: ErrorCategory
    description: str
    examples: List[ErrorInstance]
    frequency: int
    affected_domains: Set[str]
    root_cause: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class ErrorAnalysisReport:
    """Complete error analysis report."""
    total_samples: int
    total_errors: int
    error_rate: float
    errors_by_category: Dict[ErrorCategory, int]
    errors_by_severity: Dict[Severity, int]
    patterns: List[ErrorPattern]
    recommendations: List[str]
    domain_analysis: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any]


class ErrorClassifier:
    """
    Classifies errors into categories based on patterns and heuristics.
    
    Uses keyword matching, pattern detection, and statistical analysis
    to categorize different types of errors.
    """
    
    def __init__(self):
        """Initialize error classifier with detection patterns."""
        self.patterns = {
            ErrorCategory.FACTUAL: [
                r'incorrect',
                r'wrong\s+(date|number|name|value)',
                r'error\s+in\s+fact',
                r'inaccurate'
            ],
            ErrorCategory.INCOMPLETE: [
                r'missing\s+information',
                r'incomplete',
                r'not\s+enough\s+detail',
                r'lacks'
            ],
            ErrorCategory.IRRELEVANT: [
                r'off[- ]topic',
                r'not\s+relevant',
                r'unrelated',
                r'doesn\'t\s+address'
            ],
            ErrorCategory.FORMATTING: [
                r'wrong\s+format',
                r'formatting\s+issue',
                r'structure\s+error'
            ],
            ErrorCategory.HALLUCINATION: [
                r'made\s+up',
                r'fabricat',
                r'hallucin',
                r'doesn\'t\s+exist'
            ],
            ErrorCategory.CONTRADICTORY: [
                r'contradict',
                r'inconsistent',
                r'conflicts?\s+with'
            ],
            ErrorCategory.TRUNCATED: [
                r'cut\s+off',
                r'truncat',
                r'incomplete\s+sentence'
            ],
            ErrorCategory.TIMEOUT: [
                r'timeout',
                r'timed?\s+out',
                r'no\s+response'
            ]
        }
    
    def classify(
        self, 
        response: str, 
        expected: Optional[str] = None,
        error_details: Optional[str] = None
    ) -> Tuple[ErrorCategory, Severity, str]:
        """
        Classify an error based on response and context.
        
        Args:
            response: The actual response
            expected: Expected response (if available)
            error_details: Additional error information
            
        Returns:
            Tuple of (category, severity, details)
        """
        text_to_analyze = f"{response} {error_details or ''}"
        
        # Check patterns
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_to_analyze, re.IGNORECASE):
                    severity = self._determine_severity(category, response, expected)
                    details = self._generate_details(category, response, expected)
                    return category, severity, details
        
        # Statistical analysis if expected is available
        if expected:
            similarity = self._calculate_similarity(response, expected)
            
            if similarity < 0.2:
                return ErrorCategory.IRRELEVANT, Severity.HIGH, \
                    f"Response has very low similarity ({similarity:.2f}) to expected"
            elif similarity < 0.5:
                return ErrorCategory.INCOMPLETE, Severity.MEDIUM, \
                    f"Response partially matches expected ({similarity:.2f})"
        
        # Check for empty or very short responses
        if len(response.strip()) < 10:
            return ErrorCategory.TRUNCATED, Severity.HIGH, "Response is too short"
        
        return ErrorCategory.UNKNOWN, Severity.MEDIUM, "Unable to classify error"
    
    def _determine_severity(
        self, 
        category: ErrorCategory, 
        response: str, 
        expected: Optional[str]
    ) -> Severity:
        """Determine error severity based on category and context."""
        severity_map = {
            ErrorCategory.HALLUCINATION: Severity.CRITICAL,
            ErrorCategory.FACTUAL: Severity.HIGH,
            ErrorCategory.CONTRADICTORY: Severity.HIGH,
            ErrorCategory.MODEL_ERROR: Severity.HIGH,
            ErrorCategory.TIMEOUT: Severity.MEDIUM,
            ErrorCategory.INCOMPLETE: Severity.MEDIUM,
            ErrorCategory.IRRELEVANT: Severity.MEDIUM,
            ErrorCategory.TRUNCATED: Severity.MEDIUM,
            ErrorCategory.FORMATTING: Severity.LOW,
            ErrorCategory.AMBIGUOUS: Severity.LOW,
            ErrorCategory.OUTDATED: Severity.LOW,
        }
        return severity_map.get(category, Severity.MEDIUM)
    
    def _generate_details(
        self, 
        category: ErrorCategory, 
        response: str, 
        expected: Optional[str]
    ) -> str:
        """Generate detailed error description."""
        templates = {
            ErrorCategory.FACTUAL: "Response contains factual inaccuracies",
            ErrorCategory.INCOMPLETE: "Response is missing key information",
            ErrorCategory.IRRELEVANT: "Response does not address the query",
            ErrorCategory.FORMATTING: "Response has formatting issues",
            ErrorCategory.HALLUCINATION: "Response contains fabricated information",
            ErrorCategory.CONTRADICTORY: "Response contains contradictory statements",
            ErrorCategory.TRUNCATED: "Response was cut off or incomplete",
            ErrorCategory.TIMEOUT: "Request timed out before response"
        }
        return templates.get(category, f"Error category: {category.value}")
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


class ErrorPatternDetector:
    """
    Detects recurring error patterns across multiple error instances.
    
    Uses clustering and frequency analysis to identify systemic issues.
    """
    
    def __init__(self, min_pattern_frequency: int = 3):
        """
        Initialize pattern detector.
        
        Args:
            min_pattern_frequency: Minimum occurrences to be considered a pattern
        """
        self.min_frequency = min_pattern_frequency
    
    def detect_patterns(self, errors: List[ErrorInstance]) -> List[ErrorPattern]:
        """
        Detect recurring patterns in error list.
        
        Args:
            errors: List of error instances
            
        Returns:
            List of detected error patterns
        """
        patterns = []
        
        # Group by category
        category_groups: Dict[ErrorCategory, List[ErrorInstance]] = defaultdict(list)
        for error in errors:
            category_groups[error.category].append(error)
        
        # Detect patterns within each category
        pattern_id = 0
        for category, category_errors in category_groups.items():
            if len(category_errors) >= self.min_frequency:
                # Extract domains
                domains = set()
                for error in category_errors:
                    if 'domain' in error.context:
                        domains.add(error.context['domain'])
                
                # Create pattern
                pattern = ErrorPattern(
                    pattern_id=f"PATTERN_{pattern_id:03d}",
                    category=category,
                    description=f"Recurring {category.value} errors",
                    examples=category_errors[:5],  # Top 5 examples
                    frequency=len(category_errors),
                    affected_domains=domains,
                    root_cause=self._infer_root_cause(category, category_errors),
                    recommendation=self._generate_recommendation(category, category_errors)
                )
                patterns.append(pattern)
                pattern_id += 1
        
        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        
        return patterns
    
    def _infer_root_cause(
        self, 
        category: ErrorCategory, 
        errors: List[ErrorInstance]
    ) -> str:
        """Infer potential root cause based on error pattern."""
        root_causes = {
            ErrorCategory.FACTUAL: "Model training data may be outdated or contain inaccuracies",
            ErrorCategory.INCOMPLETE: "Context window limitations or retrieval gaps",
            ErrorCategory.IRRELEVANT: "Query understanding or routing issues",
            ErrorCategory.FORMATTING: "Prompt template or instruction clarity issues",
            ErrorCategory.HALLUCINATION: "Insufficient grounding or overconfidence in generation",
            ErrorCategory.CONTRADICTORY: "Multiple conflicting sources or context confusion",
            ErrorCategory.TRUNCATED: "Token limit exceeded or generation cutoff",
            ErrorCategory.TIMEOUT: "Model latency or API rate limiting",
            ErrorCategory.MODEL_ERROR: "API instability or model availability issues"
        }
        return root_causes.get(category, "Root cause requires further investigation")
    
    def _generate_recommendation(
        self, 
        category: ErrorCategory, 
        errors: List[ErrorInstance]
    ) -> str:
        """Generate actionable recommendation for error pattern."""
        recommendations = {
            ErrorCategory.FACTUAL: "Implement fact-checking layer and RAG with authoritative sources",
            ErrorCategory.INCOMPLETE: "Increase context window and improve retrieval coverage",
            ErrorCategory.IRRELEVANT: "Enhance query classification and routing logic",
            ErrorCategory.FORMATTING: "Refine prompt templates with explicit format instructions",
            ErrorCategory.HALLUCINATION: "Add hallucination detection and verification step",
            ErrorCategory.CONTRADICTORY: "Implement consistency checking across response",
            ErrorCategory.TRUNCATED: "Implement token counting and response chunking",
            ErrorCategory.TIMEOUT: "Add retry logic and fallback model configuration",
            ErrorCategory.MODEL_ERROR: "Implement circuit breaker and model fallback chain"
        }
        return recommendations.get(category, "Review error instances for specific improvements")


class ErrorAnalyzer:
    """
    Main error analysis class that combines classification and pattern detection.
    
    Provides comprehensive error analysis for benchmark evaluation.
    """
    
    def __init__(self):
        """Initialize error analyzer."""
        self.classifier = ErrorClassifier()
        self.pattern_detector = ErrorPatternDetector()
    
    def analyze_failures(
        self,
        results: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> ErrorAnalysisReport:
        """
        Analyze failures from benchmark results.
        
        Args:
            results: List of evaluation results with scores
            threshold: Score threshold below which is considered failure
            
        Returns:
            Comprehensive error analysis report
        """
        errors: List[ErrorInstance] = []
        
        # Identify and classify errors
        for i, result in enumerate(results):
            score = result.get('score', result.get('quality_score', 0))
            
            if score < threshold:
                query = result.get('query', f'Query {i}')
                response = result.get('response', '')
                expected = result.get('expected', None)
                error_details = result.get('error_details', None)
                
                category, severity, details = self.classifier.classify(
                    response, expected, error_details
                )
                
                error = ErrorInstance(
                    error_id=f"ERR_{i:04d}",
                    query=query,
                    response=response[:500] if response else "",  # Truncate for storage
                    expected=expected[:500] if expected else None,
                    category=category,
                    severity=severity,
                    details=details,
                    context={
                        'domain': result.get('domain', 'unknown'),
                        'complexity': result.get('complexity', 'medium'),
                        'score': score,
                        'model': result.get('model', 'unknown')
                    }
                )
                errors.append(error)
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(errors)
        
        # Aggregate statistics
        errors_by_category: Dict[ErrorCategory, int] = defaultdict(int)
        errors_by_severity: Dict[Severity, int] = defaultdict(int)
        
        for error in errors:
            errors_by_category[error.category] += 1
            errors_by_severity[error.severity] += 1
        
        # Domain analysis
        domain_errors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        domain_totals: Dict[str, int] = defaultdict(int)
        
        for result in results:
            domain = result.get('domain', 'unknown')
            domain_totals[domain] += 1
        
        for error in errors:
            domain = error.context.get('domain', 'unknown')
            domain_errors[domain][error.category.value] += 1
        
        domain_analysis = {}
        for domain, totals in domain_totals.items():
            domain_analysis[domain] = {
                'total_samples': totals,
                'total_errors': sum(domain_errors[domain].values()),
                'error_rate': sum(domain_errors[domain].values()) / totals if totals > 0 else 0,
                'by_category': dict(domain_errors[domain])
            }
        
        # Generate recommendations
        recommendations = self._generate_global_recommendations(errors, patterns)
        
        return ErrorAnalysisReport(
            total_samples=len(results),
            total_errors=len(errors),
            error_rate=len(errors) / len(results) if results else 0,
            errors_by_category=dict(errors_by_category),
            errors_by_severity=dict(errors_by_severity),
            patterns=patterns,
            recommendations=recommendations,
            domain_analysis=domain_analysis,
            metadata={
                'timestamp': datetime.now().isoformat(),
                'threshold': threshold,
                'n_patterns_detected': len(patterns)
            }
        )
    
    def _generate_global_recommendations(
        self, 
        errors: List[ErrorInstance], 
        patterns: List[ErrorPattern]
    ) -> List[str]:
        """Generate prioritized global recommendations."""
        recommendations = []
        
        if not errors:
            recommendations.append("✅ No errors detected - system performing well")
            return recommendations
        
        # Priority 1: Critical errors
        critical_count = sum(1 for e in errors if e.severity == Severity.CRITICAL)
        if critical_count > 0:
            recommendations.append(
                f"🚨 CRITICAL: {critical_count} critical errors require immediate attention"
            )
        
        # Priority 2: High-frequency patterns
        for pattern in patterns[:3]:  # Top 3 patterns
            recommendations.append(
                f"📋 {pattern.category.value.upper()}: {pattern.recommendation}"
            )
        
        # Priority 3: Domain-specific issues
        domain_error_rates = {}
        for error in errors:
            domain = error.context.get('domain', 'unknown')
            if domain not in domain_error_rates:
                domain_error_rates[domain] = 0
            domain_error_rates[domain] += 1
        
        if domain_error_rates:
            worst_domain = max(domain_error_rates, key=domain_error_rates.get)
            recommendations.append(
                f"🎯 Focus on '{worst_domain}' domain with {domain_error_rates[worst_domain]} errors"
            )
        
        return recommendations
    
    def generate_report_text(self, report: ErrorAnalysisReport) -> str:
        """Generate human-readable error analysis report."""
        lines = [
            "=" * 70,
            "  ERROR ANALYSIS REPORT",
            "=" * 70,
            "",
            f"📊 Summary:",
            f"   Total Samples: {report.total_samples}",
            f"   Total Errors: {report.total_errors}",
            f"   Error Rate: {report.error_rate:.1%}",
            "",
            "📈 Errors by Category:",
        ]
        
        for category, count in sorted(report.errors_by_category.items(), 
                                      key=lambda x: x[1], reverse=True):
            pct = count / report.total_errors * 100 if report.total_errors > 0 else 0
            lines.append(f"   {category.value:20} {count:4d} ({pct:5.1f}%)")
        
        lines.extend([
            "",
            "⚠️  Errors by Severity:",
        ])
        
        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
            count = report.errors_by_severity.get(severity, 0)
            pct = count / report.total_errors * 100 if report.total_errors > 0 else 0
            lines.append(f"   {severity.name:12} {count:4d} ({pct:5.1f}%)")
        
        if report.patterns:
            lines.extend([
                "",
                "🔍 Detected Patterns:",
            ])
            for pattern in report.patterns[:5]:
                lines.append(f"   [{pattern.pattern_id}] {pattern.category.value}: {pattern.frequency} occurrences")
                if pattern.root_cause:
                    lines.append(f"      Root Cause: {pattern.root_cause[:60]}...")
        
        lines.extend([
            "",
            "💡 Recommendations:",
        ])
        for rec in report.recommendations:
            lines.append(f"   {rec}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def export_to_json(self, report: ErrorAnalysisReport, filepath: str) -> None:
        """Export error analysis report to JSON."""
        data = {
            'total_samples': report.total_samples,
            'total_errors': report.total_errors,
            'error_rate': report.error_rate,
            'errors_by_category': {k.value: v for k, v in report.errors_by_category.items()},
            'errors_by_severity': {k.name: v for k, v in report.errors_by_severity.items()},
            'patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'category': p.category.value,
                    'description': p.description,
                    'frequency': p.frequency,
                    'affected_domains': list(p.affected_domains),
                    'root_cause': p.root_cause,
                    'recommendation': p.recommendation
                }
                for p in report.patterns
            ],
            'recommendations': report.recommendations,
            'domain_analysis': report.domain_analysis,
            'metadata': report.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class ErrorImpactAnalyzer:
    """
    Analyzes the business/research impact of errors.
    
    Helps prioritize error fixes based on impact analysis.
    """
    
    def calculate_impact_score(
        self, 
        error: ErrorInstance, 
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate impact score for an error.
        
        Args:
            error: Error instance to analyze
            weights: Optional custom weights for impact factors
            
        Returns:
            Impact score between 0 and 1
        """
        weights = weights or {
            'severity': 0.4,
            'frequency': 0.3,
            'domain_importance': 0.2,
            'recency': 0.1
        }
        
        # Severity component
        severity_scores = {
            Severity.CRITICAL: 1.0,
            Severity.HIGH: 0.75,
            Severity.MEDIUM: 0.5,
            Severity.LOW: 0.25,
            Severity.INFO: 0.1
        }
        severity_score = severity_scores.get(error.severity, 0.5)
        
        # Domain importance (could be configured)
        domain_importance = 0.5  # Default medium importance
        
        # Calculate weighted impact
        impact = (
            weights['severity'] * severity_score +
            weights['domain_importance'] * domain_importance
        )
        
        return min(1.0, max(0.0, impact))
    
    def prioritize_fixes(
        self, 
        errors: List[ErrorInstance]
    ) -> List[Tuple[ErrorInstance, float]]:
        """
        Prioritize errors for fixing based on impact.
        
        Args:
            errors: List of error instances
            
        Returns:
            List of (error, impact_score) sorted by priority
        """
        prioritized = []
        
        for error in errors:
            impact = self.calculate_impact_score(error)
            prioritized.append((error, impact))
        
        # Sort by impact (highest first)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        
        return prioritized


def run_error_analysis(
    benchmark_results: List[Dict[str, Any]],
    output_dir: Optional[str] = None
) -> ErrorAnalysisReport:
    """
    Run comprehensive error analysis on benchmark results.
    
    Args:
        benchmark_results: List of evaluation results
        output_dir: Optional directory to save reports
        
    Returns:
        Complete error analysis report
    """
    analyzer = ErrorAnalyzer()
    
    print("\n" + "=" * 70)
    print("  RUNNING ERROR ANALYSIS")
    print("=" * 70)
    
    # Analyze failures
    report = analyzer.analyze_failures(benchmark_results)
    
    # Print report
    print(analyzer.generate_report_text(report))
    
    # Export if directory provided
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, 'error_analysis.json')
        analyzer.export_to_json(report, json_path)
        print(f"\n💾 Report exported to: {json_path}")
    
    return report


# Example usage
if __name__ == "__main__":
    # Sample benchmark results with some failures
    sample_results = [
        {'query': 'What is Python?', 'response': 'Python is a programming language', 'score': 0.9, 'domain': 'technology'},
        {'query': 'Explain machine learning', 'response': '', 'score': 0.1, 'domain': 'technology'},
        {'query': 'What is 2+2?', 'response': 'The answer is 5', 'score': 0.2, 'domain': 'math', 'error_details': 'incorrect factual'},
        {'query': 'Write a poem', 'response': 'Here is a partial...', 'score': 0.3, 'domain': 'creative', 'error_details': 'truncated'},
        {'query': 'Explain quantum physics', 'response': 'Quantum physics studies atoms', 'score': 0.4, 'domain': 'science', 'error_details': 'incomplete'},
        {'query': 'What is the capital?', 'response': 'The weather is nice today', 'score': 0.15, 'domain': 'geography', 'error_details': 'irrelevant'},
    ]
    
    # Add more samples for pattern detection
    for i in range(20):
        sample_results.append({
            'query': f'Query {i}',
            'response': 'Partial response...' if i % 2 == 0 else 'Complete response',
            'score': 0.3 if i % 2 == 0 else 0.8,
            'domain': 'general',
            'error_details': 'incomplete' if i % 2 == 0 else None
        })
    
    report = run_error_analysis(sample_results, 'benchmarks/results')
    print(f"\n✅ Error analysis complete! Found {report.total_errors} errors in {report.total_samples} samples.")
