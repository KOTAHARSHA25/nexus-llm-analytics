"""
═══════════════════════════════════════════════════════════════════════════════
NEXUS LLM ANALYTICS - ADVANCED EVALUATION METRICS
═══════════════════════════════════════════════════════════════════════════════

Phase 4 Enhancement: Advanced NLP and statistical metrics for research-grade
evaluation. Extends base evaluation with:

1. TF-IDF Similarity
2. N-gram Analysis (BLEU-2, BLEU-3, BLEU-4)
3. Semantic Chunking
4. Statistical Significance Testing
5. Effect Size Calculation (Cohen's d)
6. Confidence Intervals

Version: 1.1.0
"""

import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from functools import lru_cache


@dataclass
class AdvancedSimilarityMetrics:
    """Advanced similarity metrics"""
    tfidf_similarity: float
    bleu_1: float
    bleu_2: float
    bleu_3: float
    bleu_4: float
    bleu_weighted: float  # Weighted average
    rouge_l: float
    meteor_score: float  # Simplified METEOR
    jaccard: float
    cosine: float
    
    def to_dict(self) -> dict:
        return {
            "tfidf_similarity": round(self.tfidf_similarity, 4),
            "bleu": {
                "bleu_1": round(self.bleu_1, 4),
                "bleu_2": round(self.bleu_2, 4),
                "bleu_3": round(self.bleu_3, 4),
                "bleu_4": round(self.bleu_4, 4),
                "weighted": round(self.bleu_weighted, 4)
            },
            "rouge_l": round(self.rouge_l, 4),
            "meteor": round(self.meteor_score, 4),
            "jaccard": round(self.jaccard, 4),
            "cosine": round(self.cosine, 4)
        }


@dataclass
class StatisticalResult:
    """Result of statistical significance test"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # at alpha=0.05
    effect_size: float
    effect_size_interpretation: str  # "small", "medium", "large"
    confidence_interval: Tuple[float, float]
    confidence_level: float
    
    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
            "effect_size": round(self.effect_size, 4),
            "effect_interpretation": self.effect_size_interpretation,
            "confidence_interval": (
                round(self.confidence_interval[0], 4),
                round(self.confidence_interval[1], 4)
            ),
            "confidence_level": self.confidence_level
        }


class AdvancedMetricsCalculator:
    """
    Advanced metrics calculator with TF-IDF, n-gram analysis, and statistical tests.
    """
    
    def __init__(self, corpus: Optional[List[str]] = None):
        """
        Initialize with optional corpus for IDF calculation.
        
        Args:
            corpus: List of documents for computing IDF weights
        """
        self.corpus = corpus or []
        self._idf_cache: Dict[str, float] = {}
        self._update_idf()
    
    def add_to_corpus(self, document: str):
        """Add a document to corpus and update IDF"""
        self.corpus.append(document)
        self._update_idf()
    
    def _update_idf(self):
        """Compute IDF weights for all terms in corpus"""
        if not self.corpus:
            return
        
        doc_freq = defaultdict(int)
        for doc in self.corpus:
            terms = set(self._tokenize(doc))
            for term in terms:
                doc_freq[term] += 1
        
        n_docs = len(self.corpus)
        for term, freq in doc_freq.items():
            self._idf_cache[term] = math.log(n_docs / (1 + freq)) + 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
        return re.findall(r'\b\w+\b', text.lower())
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from token list"""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    # =========================================================================
    # TF-IDF SIMILARITY
    # =========================================================================
    
    def calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate term frequency"""
        if not tokens:
            return {}
        
        counter = Counter(tokens)
        max_freq = max(counter.values())
        
        # Normalized TF
        return {term: 0.5 + 0.5 * count / max_freq for term, count in counter.items()}
    
    def calculate_tfidf_vector(self, text: str) -> Dict[str, float]:
        """Calculate TF-IDF vector for text"""
        tokens = self._tokenize(text)
        tf = self.calculate_tf(tokens)
        
        tfidf = {}
        for term, tf_val in tf.items():
            idf = self._idf_cache.get(term, 1.0)  # Default IDF=1 for unknown terms
            tfidf[term] = tf_val * idf
        
        return tfidf
    
    def calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF based cosine similarity"""
        vec1 = self.calculate_tfidf_vector(text1)
        vec2 = self.calculate_tfidf_vector(text2)
        
        all_terms = set(vec1.keys()) | set(vec2.keys())
        
        if not all_terms:
            return 0.0
        
        dot_product = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in all_terms)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    # =========================================================================
    # N-GRAM BLEU SCORES
    # =========================================================================
    
    def calculate_bleu_n(self, reference: str, candidate: str, n: int) -> float:
        """Calculate BLEU-n score"""
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        if len(cand_tokens) < n:
            return 0.0
        
        ref_ngrams = Counter(self._get_ngrams(ref_tokens, n))
        cand_ngrams = Counter(self._get_ngrams(cand_tokens, n))
        
        matches = 0
        for ngram, count in cand_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        return matches / len(list(self._get_ngrams(cand_tokens, n)))
    
    def calculate_bleu_all(self, reference: str, candidate: str) -> Tuple[float, float, float, float, float]:
        """
        Calculate BLEU 1-4 scores and weighted average.
        
        Returns:
            Tuple of (bleu_1, bleu_2, bleu_3, bleu_4, weighted_average)
        """
        bleu_1 = self.calculate_bleu_n(reference, candidate, 1)
        bleu_2 = self.calculate_bleu_n(reference, candidate, 2)
        bleu_3 = self.calculate_bleu_n(reference, candidate, 3)
        bleu_4 = self.calculate_bleu_n(reference, candidate, 4)
        
        # Geometric mean with smoothing for zero values
        scores = [max(s, 1e-10) for s in [bleu_1, bleu_2, bleu_3, bleu_4]]
        weighted = (scores[0] * scores[1] * scores[2] * scores[3]) ** 0.25
        
        # Brevity penalty
        ref_len = len(self._tokenize(reference))
        cand_len = len(self._tokenize(candidate))
        
        if cand_len < ref_len:
            bp = math.exp(1 - ref_len / max(1, cand_len))
        else:
            bp = 1.0
        
        weighted *= bp
        
        return bleu_1, bleu_2, bleu_3, bleu_4, weighted
    
    # =========================================================================
    # METEOR SCORE (Simplified)
    # =========================================================================
    
    def calculate_meteor(self, reference: str, candidate: str) -> float:
        """
        Calculate simplified METEOR score.
        
        METEOR considers:
        - Exact matches
        - Stemming (simplified: prefix matching)
        - Word order (fragmentation penalty)
        """
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        # Stage 1: Exact matches
        ref_counter = Counter(ref_tokens)
        cand_counter = Counter(cand_tokens)
        
        exact_matches = 0
        for token in cand_counter:
            if token in ref_counter:
                exact_matches += min(cand_counter[token], ref_counter[token])
        
        # Stage 2: Stem/prefix matches (4-char prefix)
        unmatched_ref = {t for t in ref_tokens if t not in cand_counter or cand_counter.get(t, 0) < ref_counter[t]}
        unmatched_cand = {t for t in cand_tokens if t not in ref_counter or ref_counter.get(t, 0) < cand_counter[t]}
        
        stem_matches = 0
        for cand_word in unmatched_cand:
            for ref_word in unmatched_ref:
                if len(cand_word) >= 4 and len(ref_word) >= 4:
                    if cand_word[:4] == ref_word[:4]:
                        stem_matches += 1
                        unmatched_ref.discard(ref_word)
                        break
        
        total_matches = exact_matches + stem_matches * 0.8  # Stem matches weighted less
        
        precision = total_matches / len(cand_tokens)
        recall = total_matches / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        # F-mean with recall weighted more (alpha=0.9)
        alpha = 0.9
        f_mean = 1 / (alpha / max(0.001, precision) + (1 - alpha) / max(0.001, recall))
        
        # Fragmentation penalty
        chunks = self._count_chunks(ref_tokens, cand_tokens)
        frag = chunks / max(1, total_matches)
        penalty = 0.5 * (frag ** 3)
        
        return f_mean * (1 - penalty)
    
    def _count_chunks(self, ref_tokens: List[str], cand_tokens: List[str]) -> int:
        """Count number of chunks (contiguous matches)"""
        if not cand_tokens:
            return 0
        
        ref_set = set(ref_tokens)
        in_chunk = False
        chunks = 0
        
        for token in cand_tokens:
            if token in ref_set:
                if not in_chunk:
                    chunks += 1
                    in_chunk = True
            else:
                in_chunk = False
        
        return chunks
    
    # =========================================================================
    # ROUGE-L (Enhanced)
    # =========================================================================
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L F1 score with LCS"""
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        # Convert to tuple for caching
        lcs_len = self._lcs_length(tuple(ref_tokens), tuple(cand_tokens))
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        precision = lcs_len / len(cand_tokens)
        recall = lcs_len / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    @lru_cache(maxsize=1000)
    def _lcs_length(self, seq1: Tuple[str, ...], seq2: Tuple[str, ...]) -> int:
        """Calculate LCS length with memoization"""
        m, n = len(seq1), len(seq2)
        
        # Space-optimized DP
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, [0] * (n + 1)
        
        return prev[n] if m > 0 else 0
    
    # =========================================================================
    # COMBINED SIMILARITY
    # =========================================================================
    
    def calculate_all_similarity(self, text1: str, text2: str) -> AdvancedSimilarityMetrics:
        """Calculate all similarity metrics at once"""
        bleu_1, bleu_2, bleu_3, bleu_4, bleu_weighted = self.calculate_bleu_all(text1, text2)
        
        # Basic metrics
        tokens1 = set(self._tokenize(text1))
        tokens2 = set(self._tokenize(text2))
        
        jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2) if tokens1 | tokens2 else 0
        
        # Cosine
        counter1 = Counter(self._tokenize(text1))
        counter2 = Counter(self._tokenize(text2))
        all_terms = set(counter1.keys()) | set(counter2.keys())
        dot = sum(counter1.get(t, 0) * counter2.get(t, 0) for t in all_terms)
        mag1 = math.sqrt(sum(v**2 for v in counter1.values()))
        mag2 = math.sqrt(sum(v**2 for v in counter2.values()))
        cosine = dot / (mag1 * mag2) if mag1 and mag2 else 0
        
        return AdvancedSimilarityMetrics(
            tfidf_similarity=self.calculate_tfidf_similarity(text1, text2),
            bleu_1=bleu_1,
            bleu_2=bleu_2,
            bleu_3=bleu_3,
            bleu_4=bleu_4,
            bleu_weighted=bleu_weighted,
            rouge_l=self.calculate_rouge_l(text1, text2),
            meteor_score=self.calculate_meteor(text1, text2),
            jaccard=jaccard,
            cosine=cosine
        )
    
    # =========================================================================
    # STATISTICAL TESTS
    # =========================================================================
    
    def welch_t_test(
        self, 
        sample1: List[float], 
        sample2: List[float],
        alpha: float = 0.05
    ) -> StatisticalResult:
        """
        Perform Welch's t-test for comparing two samples.
        
        This test is more robust than Student's t-test when
        variances are unequal.
        """
        n1, n2 = len(sample1), len(sample2)
        
        if n1 < 2 or n2 < 2:
            return StatisticalResult(
                test_name="welch_t",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                effect_size=0.0,
                effect_size_interpretation="insufficient_data",
                confidence_interval=(0, 0),
                confidence_level=1 - alpha
            )
        
        mean1, mean2 = statistics.mean(sample1), statistics.mean(sample2)
        var1, var2 = statistics.variance(sample1), statistics.variance(sample2)
        
        # Welch's t-statistic
        se = math.sqrt(var1/n1 + var2/n2)
        if se == 0:
            t_stat = 0.0
        else:
            t_stat = (mean1 - mean2) / se
        
        # Welch-Satterthwaite degrees of freedom
        num = (var1/n1 + var2/n2) ** 2
        denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
        df = num / denom if denom != 0 else n1 + n2 - 2
        
        # Approximate p-value using t-distribution approximation
        p_value = self._t_to_p(abs(t_stat), df) * 2  # Two-tailed
        
        # Effect size (Cohen's d)
        pooled_std = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for difference of means
        t_crit = self._t_critical(alpha/2, df)
        margin = t_crit * se
        ci = (mean1 - mean2 - margin, mean1 - mean2 + margin)
        
        return StatisticalResult(
            test_name="welch_t",
            statistic=t_stat,
            p_value=min(1.0, p_value),
            significant=p_value < alpha,
            effect_size=abs(effect_size),
            effect_size_interpretation=self._interpret_cohens_d(abs(effect_size)),
            confidence_interval=ci,
            confidence_level=1 - alpha
        )
    
    def _t_to_p(self, t: float, df: float) -> float:
        """Approximate p-value from t-statistic (one-tailed)"""
        # Using approximation formula
        x = df / (df + t**2)
        if df <= 0:
            return 0.5
        
        # Beta incomplete function approximation
        return 0.5 * self._beta_inc(df/2, 0.5, x)
    
    def _beta_inc(self, a: float, b: float, x: float) -> float:
        """Incomplete beta function approximation"""
        if x < 0 or x > 1:
            return 0.5
        
        # Simple approximation using continued fraction
        # For more accuracy, use scipy.special.betainc
        try:
            result = (x ** a) * ((1-x) ** b)
            result *= math.gamma(a + b) / (math.gamma(a) * math.gamma(b) * a)
            return min(1.0, result * 50)  # Scaling factor
        except (ValueError, OverflowError):
            return 0.5
    
    def _t_critical(self, alpha: float, df: float) -> float:
        """Approximate critical t-value"""
        # Approximation for large df
        if df > 30:
            # Normal approximation
            z_scores = {0.025: 1.96, 0.05: 1.645, 0.01: 2.576, 0.005: 2.807}
            return z_scores.get(alpha, 1.96)
        
        # Simple lookup for common values
        t_table = {
            (0.025, 10): 2.228,
            (0.025, 20): 2.086,
            (0.025, 30): 2.042,
            (0.05, 10): 1.812,
            (0.05, 20): 1.725,
            (0.05, 30): 1.697
        }
        
        closest = min(t_table.keys(), key=lambda k: abs(k[0]-alpha) + abs(k[1]-df))
        return t_table[closest]
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def bootstrap_confidence_interval(
        self,
        scores: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval using bootstrap resampling.
        """
        import random
        
        if len(scores) < 2:
            mean = scores[0] if scores else 0
            return (mean, mean)
        
        bootstrap_means = []
        n = len(scores)
        
        for _ in range(n_bootstrap):
            sample = [random.choice(scores) for _ in range(n)]
            bootstrap_means.append(statistics.mean(sample))
        
        bootstrap_means.sort()
        
        lower_idx = int((1 - confidence) / 2 * n_bootstrap)
        upper_idx = int((1 + confidence) / 2 * n_bootstrap)
        
        return (bootstrap_means[lower_idx], bootstrap_means[upper_idx - 1])
    
    def compare_systems(
        self,
        system_a_scores: List[float],
        system_b_scores: List[float],
        metric_name: str = "overall_score"
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison of two systems.
        
        Returns statistical tests, effect sizes, and recommendations.
        """
        # Descriptive statistics
        a_mean = statistics.mean(system_a_scores)
        b_mean = statistics.mean(system_b_scores)
        a_std = statistics.stdev(system_a_scores) if len(system_a_scores) > 1 else 0
        b_std = statistics.stdev(system_b_scores) if len(system_b_scores) > 1 else 0
        
        # Statistical test
        test_result = self.welch_t_test(system_a_scores, system_b_scores)
        
        # Bootstrap CIs
        a_ci = self.bootstrap_confidence_interval(system_a_scores)
        b_ci = self.bootstrap_confidence_interval(system_b_scores)
        
        # Improvement calculation
        improvement = (a_mean - b_mean) / max(0.001, b_mean) * 100
        
        return {
            "metric": metric_name,
            "system_a": {
                "mean": round(a_mean, 4),
                "std": round(a_std, 4),
                "ci_95": (round(a_ci[0], 4), round(a_ci[1], 4)),
                "n": len(system_a_scores)
            },
            "system_b": {
                "mean": round(b_mean, 4),
                "std": round(b_std, 4),
                "ci_95": (round(b_ci[0], 4), round(b_ci[1], 4)),
                "n": len(system_b_scores)
            },
            "difference": {
                "raw": round(a_mean - b_mean, 4),
                "percent": round(improvement, 2),
                "favors": "system_a" if a_mean > b_mean else "system_b"
            },
            "statistical_test": test_result.to_dict(),
            "recommendation": self._generate_recommendation(test_result, improvement)
        }
    
    def _generate_recommendation(
        self, 
        test_result: StatisticalResult, 
        improvement_percent: float
    ) -> str:
        """Generate human-readable recommendation"""
        if not test_result.significant:
            return "No statistically significant difference detected. Consider larger sample size."
        
        if test_result.effect_size_interpretation == "negligible":
            return "Statistically significant but practically negligible difference."
        elif test_result.effect_size_interpretation == "small":
            return f"Small but significant effect ({improvement_percent:.1f}% difference)."
        elif test_result.effect_size_interpretation == "medium":
            return f"Meaningful improvement detected ({improvement_percent:.1f}% difference)."
        else:
            return f"Large, significant improvement ({improvement_percent:.1f}% difference)."


# =============================================================================
# SEMANTIC CHUNKING FOR LONG DOCUMENTS
# =============================================================================

class SemanticChunker:
    """
    Chunk long texts into semantically coherent segments.
    
    Useful for evaluating long-form responses where different
    sections address different aspects of the query.
    """
    
    def __init__(self, max_chunk_size: int = 200, overlap: int = 20):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Split text into sentence-based chunks"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sentence in sentences:
            words = len(sentence.split())
            
            if current_len + words > self.max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep overlap
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s.split()) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s.split())
                    else:
                        break
                current_chunk = overlap_sentences
                current_len = overlap_len
            
            current_chunk.append(sentence)
            current_len += words
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_by_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text by structural elements (headers, bullets, paragraphs).
        
        Returns list of dicts with chunk text and type.
        """
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Detect type
            if re.match(r'^#+\s', para):
                chunk_type = "header"
            elif re.match(r'^[-*•]\s', para):
                chunk_type = "bullet_list"
            elif re.match(r'^\d+\.\s', para):
                chunk_type = "numbered_list"
            else:
                chunk_type = "paragraph"
            
            chunks.append({
                "text": para,
                "type": chunk_type,
                "word_count": len(para.split())
            })
        
        return chunks


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'AdvancedMetricsCalculator',
    'AdvancedSimilarityMetrics',
    'StatisticalResult',
    'SemanticChunker'
]
