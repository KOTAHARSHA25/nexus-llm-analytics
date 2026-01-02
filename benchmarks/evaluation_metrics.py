"""
═══════════════════════════════════════════════════════════════════════════════
NEXUS LLM ANALYTICS - EVALUATION METRICS
═══════════════════════════════════════════════════════════════════════════════

Phase 4.2: Research-grade evaluation metrics for measuring system performance.

Metrics Categories:
1. ACCURACY METRICS - Correctness of outputs
2. EFFICIENCY METRICS - Speed and resource usage
3. QUALITY METRICS - Response quality and completeness
4. SYSTEM METRICS - Model selection, routing, review efficacy

Usage:
    from benchmarks.evaluation_metrics import MetricsCalculator, ResearchMetrics
    
    calculator = MetricsCalculator()
    metrics = calculator.evaluate_response(query, response, ground_truth)

Version: 1.0.0
"""

import math
import re
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter
import statistics


@dataclass
class AccuracyMetrics:
    """Accuracy-related metrics"""
    exact_match: float  # Binary: 0 or 1
    fuzzy_match: float  # 0.0 - 1.0 similarity
    numeric_accuracy: float  # For numeric outputs
    factual_consistency: float  # Does response align with data
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EfficiencyMetrics:
    """Efficiency and performance metrics"""
    latency_seconds: float
    tokens_per_second: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    model_calls: int
    iterations: int
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityMetrics:
    """Response quality metrics"""
    completeness: float  # Required elements present
    coherence: float  # Logical flow
    relevance: float  # Relevance to query
    specificity: float  # Specific vs vague answers
    actionability: float  # Contains actionable insights
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SystemMetrics:
    """System-level metrics"""
    model_selection_correct: bool
    routing_efficiency: float  # Right complexity assessment
    review_applied: bool
    review_improved: bool  # Did review improve output
    cache_hit: bool
    fallback_used: bool
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResearchMetrics:
    """Complete research metrics bundle"""
    query_id: str
    accuracy: AccuracyMetrics
    efficiency: EfficiencyMetrics
    quality: QualityMetrics
    system: SystemMetrics
    overall_score: float  # Weighted combination
    
    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "accuracy": self.accuracy.to_dict(),
            "efficiency": self.efficiency.to_dict(),
            "quality": self.quality.to_dict(),
            "system": self.system.to_dict(),
            "overall_score": self.overall_score
        }


class MetricsCalculator:
    """
    Calculates research-grade metrics for system evaluation.
    
    Supports:
    - Text similarity metrics (BLEU, ROUGE-like, Jaccard)
    - Numeric accuracy validation
    - Completeness scoring
    - Quality assessment
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize with optional custom weights.
        
        Args:
            weights: Dict mapping metric categories to weights
                     Default: accuracy=0.3, quality=0.3, efficiency=0.2, system=0.2
        """
        self.weights = weights or {
            "accuracy": 0.30,
            "quality": 0.30,
            "efficiency": 0.20,
            "system": 0.20
        }
    
    # =========================================================================
    # TEXT SIMILARITY METRICS
    # =========================================================================
    
    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        words1 = set(self._tokenize(text1))
        words2 = set(self._tokenize(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity using word frequencies"""
        words1 = self._tokenize(text1)
        words2 = self._tokenize(text2)
        
        counter1 = Counter(words1)
        counter2 = Counter(words2)
        
        all_words = set(counter1.keys()) | set(counter2.keys())
        
        if not all_words:
            return 0.0
        
        dot_product = sum(counter1.get(w, 0) * counter2.get(w, 0) for w in all_words)
        magnitude1 = math.sqrt(sum(v ** 2 for v in counter1.values()))
        magnitude2 = math.sqrt(sum(v ** 2 for v in counter2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def calculate_bleu_1(self, reference: str, candidate: str) -> float:
        """Calculate BLEU-1 (unigram precision)"""
        ref_words = self._tokenize(reference)
        cand_words = self._tokenize(candidate)
        
        if not cand_words:
            return 0.0
        
        ref_counter = Counter(ref_words)
        cand_counter = Counter(cand_words)
        
        matches = 0
        for word, count in cand_counter.items():
            matches += min(count, ref_counter.get(word, 0))
        
        return matches / len(cand_words)
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L (longest common subsequence) F1 score"""
        ref_words = self._tokenize(reference)
        cand_words = self._tokenize(candidate)
        
        lcs_length = self._lcs_length(ref_words, cand_words)
        
        if not ref_words or not cand_words:
            return 0.0
        
        precision = lcs_length / len(cand_words)
        recall = lcs_length / len(ref_words)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    # =========================================================================
    # NUMERIC ACCURACY
    # =========================================================================
    
    def extract_numbers(self, text: str) -> List[float]:
        """Extract numeric values from text"""
        # Match integers, decimals, percentages, currency
        patterns = [
            r'[-+]?\d+\.?\d*%?',  # Basic numbers and percentages
            r'\$[\d,]+\.?\d*',     # Currency
            r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?'  # Numbers with commas
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Clean and convert
                    cleaned = match.replace(',', '').replace('$', '').replace('%', '')
                    numbers.append(float(cleaned))
                except ValueError:
                    continue
        
        return numbers
    
    def calculate_numeric_accuracy(
        self, 
        response: str, 
        expected_values: List[float],
        tolerance: float = 0.05
    ) -> float:
        """
        Calculate accuracy of numeric values in response.
        
        Args:
            response: The response text
            expected_values: Expected numeric values
            tolerance: Acceptable relative error (5% default)
        
        Returns:
            Accuracy score 0.0 - 1.0
        """
        if not expected_values:
            return 1.0  # No numerics expected
        
        extracted = self.extract_numbers(response)
        
        if not extracted:
            return 0.0
        
        matches = 0
        for expected in expected_values:
            for found in extracted:
                if expected == 0:
                    if abs(found) < 0.01:  # Close to zero
                        matches += 1
                        break
                else:
                    relative_error = abs(found - expected) / abs(expected)
                    if relative_error <= tolerance:
                        matches += 1
                        break
        
        return matches / len(expected_values)
    
    # =========================================================================
    # COMPLETENESS SCORING
    # =========================================================================
    
    def calculate_completeness(
        self, 
        response: str, 
        required_elements: List[str]
    ) -> Tuple[float, int, int]:
        """
        Calculate completeness based on required elements.
        
        Returns:
            Tuple of (score, found_count, total_count)
        """
        if not required_elements:
            return 1.0, 0, 0
        
        response_lower = response.lower()
        found = sum(1 for elem in required_elements if elem.lower() in response_lower)
        
        return found / len(required_elements), found, len(required_elements)
    
    # =========================================================================
    # QUALITY ASSESSMENT
    # =========================================================================
    
    def assess_coherence(self, response: str) -> float:
        """
        Assess logical coherence of response.
        
        Heuristics:
        - Sentence count and structure
        - Transition words presence
        - Consistent formatting
        """
        if not response or len(response) < 10:
            return 0.0
        
        score = 0.0
        
        # Check for multiple sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 2:
            score += 0.3
        
        # Check for transition words
        transitions = ['however', 'therefore', 'additionally', 'furthermore',
                      'first', 'second', 'finally', 'in conclusion', 'moreover',
                      'consequently', 'as a result', 'on the other hand']
        response_lower = response.lower()
        transition_count = sum(1 for t in transitions if t in response_lower)
        score += min(0.3, transition_count * 0.1)
        
        # Check for structured formatting
        if any(marker in response for marker in ['- ', '• ', '1.', '2.', '*']):
            score += 0.2
        
        # Check for reasonable length
        word_count = len(response.split())
        if 50 <= word_count <= 1000:
            score += 0.2
        elif word_count > 1000:
            score += 0.1  # Too long might be verbose
        
        return min(1.0, score)
    
    def assess_specificity(self, response: str) -> float:
        """
        Assess how specific vs vague the response is.
        
        Specific responses have:
        - Concrete numbers
        - Named entities
        - Specific actions/recommendations
        """
        if not response:
            return 0.0
        
        score = 0.0
        
        # Check for numbers
        numbers = self.extract_numbers(response)
        if numbers:
            score += min(0.4, len(numbers) * 0.1)
        
        # Check for specific terms (not vague)
        vague_terms = ['some', 'many', 'various', 'several', 'often', 
                       'sometimes', 'generally', 'usually', 'might', 'may']
        specific_terms = ['specifically', 'exactly', 'precisely', 'total of',
                         'approximately', 'measured at', 'calculated as']
        
        response_lower = response.lower()
        vague_count = sum(1 for t in vague_terms if t in response_lower)
        specific_count = sum(1 for t in specific_terms if t in response_lower)
        
        score += min(0.3, specific_count * 0.1)
        score -= min(0.2, vague_count * 0.05)
        
        # Check for actionable items
        action_indicators = ['recommend', 'should', 'need to', 'must', 
                            'action', 'step', 'implement', 'consider']
        action_count = sum(1 for a in action_indicators if a in response_lower)
        score += min(0.3, action_count * 0.1)
        
        return max(0.0, min(1.0, score))
    
    def assess_relevance(self, query: str, response: str) -> float:
        """Assess relevance of response to query"""
        query_keywords = set(self._tokenize(query))
        response_keywords = set(self._tokenize(response))
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'can',
                    'of', 'in', 'to', 'for', 'on', 'with', 'at', 'by', 'from',
                    'what', 'which', 'who', 'how', 'when', 'where', 'why', 'this',
                    'that', 'these', 'those', 'it', 'its', 'and', 'or', 'but', 'if'}
        
        query_keywords -= stopwords
        response_keywords -= stopwords
        
        if not query_keywords:
            return 0.5  # Can't assess
        
        overlap = query_keywords & response_keywords
        return len(overlap) / len(query_keywords)
    
    # =========================================================================
    # FULL EVALUATION
    # =========================================================================
    
    def evaluate_response(
        self,
        query_id: str,
        query: str,
        response: str,
        ground_truth: Optional[str] = None,
        expected_values: Optional[List[float]] = None,
        required_elements: Optional[List[str]] = None,
        execution_context: Optional[Dict] = None
    ) -> ResearchMetrics:
        """
        Perform full evaluation of a response.
        
        Args:
            query_id: Unique identifier for the query
            query: The original query
            response: The system's response
            ground_truth: Optional reference answer
            expected_values: Optional expected numeric values
            required_elements: Required keywords/phrases
            execution_context: Dict with execution details (latency, model, etc.)
        
        Returns:
            ResearchMetrics with all evaluation scores
        """
        context = execution_context or {}
        
        # Calculate accuracy metrics
        if ground_truth:
            exact_match = 1.0 if response.strip() == ground_truth.strip() else 0.0
            fuzzy_match = self.calculate_cosine_similarity(response, ground_truth)
        else:
            exact_match = 0.0
            fuzzy_match = 0.0
        
        numeric_accuracy = self.calculate_numeric_accuracy(
            response, 
            expected_values or []
        )
        
        # Factual consistency - based on elements and query relevance
        relevance = self.assess_relevance(query, response)
        factual_consistency = (relevance + numeric_accuracy) / 2 if expected_values else relevance
        
        accuracy = AccuracyMetrics(
            exact_match=exact_match,
            fuzzy_match=round(fuzzy_match, 4),
            numeric_accuracy=round(numeric_accuracy, 4),
            factual_consistency=round(factual_consistency, 4)
        )
        
        # Calculate efficiency metrics
        efficiency = EfficiencyMetrics(
            latency_seconds=context.get("latency_seconds", 0.0),
            tokens_per_second=context.get("tokens_per_second", 0.0),
            total_tokens=context.get("total_tokens", 0),
            input_tokens=context.get("input_tokens", 0),
            output_tokens=context.get("output_tokens", len(response.split())),
            model_calls=context.get("model_calls", 1),
            iterations=context.get("iterations", 1)
        )
        
        # Calculate quality metrics
        completeness_score, found, total = self.calculate_completeness(
            response, 
            required_elements or []
        )
        coherence = self.assess_coherence(response)
        specificity = self.assess_specificity(response)
        actionability = self._assess_actionability(response)
        
        quality = QualityMetrics(
            completeness=round(completeness_score, 4),
            coherence=round(coherence, 4),
            relevance=round(relevance, 4),
            specificity=round(specificity, 4),
            actionability=round(actionability, 4)
        )
        
        # Calculate system metrics
        system = SystemMetrics(
            model_selection_correct=context.get("model_correct", True),
            routing_efficiency=context.get("routing_efficiency", 1.0),
            review_applied=context.get("review_applied", False),
            review_improved=context.get("review_improved", False),
            cache_hit=context.get("cache_hit", False),
            fallback_used=context.get("fallback_used", False)
        )
        
        # Calculate overall score
        accuracy_avg = (accuracy.fuzzy_match + accuracy.numeric_accuracy + accuracy.factual_consistency) / 3
        quality_avg = (quality.completeness + quality.coherence + quality.relevance + quality.specificity) / 4
        efficiency_score = min(1.0, 10.0 / max(0.1, efficiency.latency_seconds))  # Faster is better
        system_score = 0.5 + (0.25 if system.model_selection_correct else 0) + (0.25 if not system.fallback_used else 0)
        
        overall = (
            self.weights["accuracy"] * accuracy_avg +
            self.weights["quality"] * quality_avg +
            self.weights["efficiency"] * efficiency_score +
            self.weights["system"] * system_score
        )
        
        return ResearchMetrics(
            query_id=query_id,
            accuracy=accuracy,
            efficiency=efficiency,
            quality=quality,
            system=system,
            overall_score=round(overall, 4)
        )
    
    def _assess_actionability(self, response: str) -> float:
        """Assess if response contains actionable insights"""
        response_lower = response.lower()
        
        action_phrases = [
            'recommend', 'should', 'need to', 'consider', 'suggest',
            'action', 'step', 'implement', 'optimize', 'improve',
            'increase', 'decrease', 'focus on', 'prioritize', 'address'
        ]
        
        count = sum(1 for phrase in action_phrases if phrase in response_lower)
        return min(1.0, count * 0.15)


class AggregateMetrics:
    """Aggregate metrics across multiple evaluations"""
    
    def __init__(self, metrics_list: List[ResearchMetrics]):
        self.metrics = metrics_list
        self.n = len(metrics_list)
    
    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics"""
        if not self.metrics:
            return {}
        
        # Collect all scores
        overall_scores = [m.overall_score for m in self.metrics]
        latencies = [m.efficiency.latency_seconds for m in self.metrics]
        completeness = [m.quality.completeness for m in self.metrics]
        
        return {
            "total_evaluations": self.n,
            "overall_score": {
                "mean": round(statistics.mean(overall_scores), 4),
                "std": round(statistics.stdev(overall_scores), 4) if self.n > 1 else 0,
                "min": round(min(overall_scores), 4),
                "max": round(max(overall_scores), 4),
                "median": round(statistics.median(overall_scores), 4)
            },
            "latency_seconds": {
                "mean": round(statistics.mean(latencies), 4),
                "std": round(statistics.stdev(latencies), 4) if self.n > 1 else 0,
                "min": round(min(latencies), 4),
                "max": round(max(latencies), 4),
                "p95": round(sorted(latencies)[int(0.95 * self.n)] if self.n > 1 else latencies[0], 4)
            },
            "completeness": {
                "mean": round(statistics.mean(completeness), 4),
                "std": round(statistics.stdev(completeness), 4) if self.n > 1 else 0
            },
            "model_selection_accuracy": round(
                sum(1 for m in self.metrics if m.system.model_selection_correct) / self.n, 4
            ),
            "review_improvement_rate": round(
                sum(1 for m in self.metrics if m.system.review_improved) / 
                max(1, sum(1 for m in self.metrics if m.system.review_applied)), 4
            ),
            "cache_hit_rate": round(
                sum(1 for m in self.metrics if m.system.cache_hit) / self.n, 4
            )
        }
    
    def calculate_by_category(self, category_key: str = "complexity") -> Dict[str, Dict]:
        """Group metrics by a category (requires metadata in query_id)"""
        # This would require query metadata - placeholder for extension
        return {}


# Convenience function
def evaluate(
    query: str,
    response: str,
    query_id: str = "eval",
    ground_truth: Optional[str] = None,
    expected_values: Optional[List[float]] = None,
    required_elements: Optional[List[str]] = None,
    execution_context: Optional[Dict] = None
) -> ResearchMetrics:
    """Quick evaluation function"""
    calculator = MetricsCalculator()
    return calculator.evaluate_response(
        query_id=query_id,
        query=query,
        response=response,
        ground_truth=ground_truth,
        expected_values=expected_values,
        required_elements=required_elements,
        execution_context=execution_context
    )


__all__ = [
    'MetricsCalculator',
    'ResearchMetrics',
    'AccuracyMetrics',
    'EfficiencyMetrics',
    'QualityMetrics',
    'SystemMetrics',
    'AggregateMetrics',
    'evaluate'
]
