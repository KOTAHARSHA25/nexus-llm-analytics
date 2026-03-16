"""Semantic Mapper — Intelligent query-to-analysis mapping.

Provides dynamic, domain-agnostic query routing using configurable
concept patterns and confidence-scored mappings between natural
language queries and analysis capabilities.

Enterprise v2.0 Additions
-------------------------
* **MappingAuditEntry** — Dataclass capturing each semantic-mapping
  decision (query, matched concept, confidence) for auditing.
* **MappingAuditLog** — Thread-safe rolling log of mapping decisions
  with summary statistics and export capability.

All v1.x APIs (``SemanticMapper``, ``get_semantic_mapper``) remain
fully backward-compatible.

Author: Nexus Team
Since: v1.0 (Enterprise enhancements v2.0 — February 2026)
"""

import logging
from typing import Any, Dict, List, Optional, Set
import pandas as pd

logger = logging.getLogger(__name__)


class SemanticMapper:
    """
    Maps user's column names to standard analytical concepts.
    
    Example mappings:
    - "gross_inflow" -> "revenue" (finance)
    - "patient_count" -> "count" (healthcare)
    - "survival_rate" -> "rate" (medical)
    - "daily_sales" -> "revenue" (retail)
    """
    
    # Priority weights for concept disambiguation (higher = more specific)
    CONCEPT_PRIORITY = {
        'revenue': 10,  # High priority for business metrics
        'cost': 10,
        'profit': 10,
        'date': 9,      # Boost date priority
        'price': 9,
        'count': 9,
        'rate': 8,
        'performance': 7,
        'category': 5,  # Lower priority (generic)
        'id': 6,
        'status': 6,
        'location': 6
    }
    
    # Standard concept categories with comprehensive pattern matching
    CONCEPT_PATTERNS = {
        # Monetary concepts
        'revenue': [
            'revenue', 'sales', 'income', 'inflow', 'earnings', 'receipts', 
            'proceeds', 'turnover', 'gross', 'total_sales', 'net_sales',
            'billing', 'invoices', 'collections'
        ],
        'cost': [
            'cost', 'expense', 'expenditure', 'outflow', 'spending', 'outlay',
            'payment', 'disbursement', 'charge', 'fee', 'cogs', 'opex', 'capex',
            'total'  # Added for cost-total patterns
        ],
        'profit': [
            'profit', 'margin', 'earnings', 'net_income', 'gain', 'surplus',
            'ebitda', 'operating_income', 'gross_profit', 'net_profit'
        ],
        'price': [
            'price', 'rate', 'amount', 'value', 'worth', 'cost_per', 
            'unit_price', 'pricing', 'valuation', 'appraisal'
        ],
        
        # Quantity concepts
        'count': [
            'count', 'number', 'quantity', 'total', 'volume', 'amount', 
            'patients', 'users', 'customers', 'clients', 'visits', 'sessions',
            'transactions', 'orders', 'cases', 'instances', 'records',
            'population', 'size', 'frequency'
        ],
        
        # Time concepts
        'date': [
            'date', 'datetime', 'timestamp', 'time', 'period', 'year', 
            'month', 'day', 'week', 'quarter', 'fiscal', 'admission',
            'discharge', 'created', 'updated', 'modified', 'start', 'end'
        ],
        
        # Categorical concepts
        'category': [
            'category', 'type', 'class', 'group', 'segment', 'region', 
            'department', 'product', 'service', 'diagnosis', 'specialty',
            'division', 'sector', 'industry', 'channel', 'source'
        ],
        
        # Identifier concepts
        'id': [
            'id', 'identifier', 'code', 'key', 'ref', 'number', 'ssn',
            'patient_id', 'customer_id', 'order_id', 'transaction_id',
            'account', 'serial', 'sku', 'barcode'
        ],
        
        # Rate/Ratio concepts
        'rate': [
            'rate', 'ratio', 'percentage', 'percent', 'proportion', 'share', 
            'fraction', 'metric', 'kpi', 'index', 'score', 'rating',
            'conversion', 'retention', 'churn', 'mortality', 'survival',
            'growth', 'change', 'variance'
        ],
        
        # Status concepts
        'status': [
            'status', 'state', 'condition', 'phase', 'stage', 'active', 
            'flag', 'indicator', 'outcome', 'result', 'disposition',
            'approval', 'completion', 'progress'
        ],
        
        # Performance metrics
        'performance': [
            'performance', 'efficiency', 'productivity', 'utilization',
            'throughput', 'capacity', 'yield', 'quality', 'satisfaction',
            'nps', 'csat', 'engagement'
        ],
        
        # Location concepts
        'location': [
            'location', 'place', 'site', 'facility', 'branch', 'store',
            'office', 'city', 'state', 'country', 'zip', 'postal', 'address',
            'geography', 'territory', 'zone'
        ]
    }
    
    def __init__(self, custom_patterns: Optional[Dict[str, List[str]]] = None):
        """Initialize semantic mapper with cached mappings.
        
        Args:
            custom_patterns: Optional user-defined patterns to merge with defaults
        """
        self._cached_mappings: Dict[str, Dict[str, str]] = {}
        self._custom_patterns = custom_patterns or {}
        logger.debug("SemanticMapper initialized")
    
    def infer_column_concepts(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Analyze DataFrame columns and map each to a standard concept.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict mapping column_name -> standard_concept
        """
        # Handle edge cases
        if df is None or len(df.columns) == 0:
            logger.warning("Empty or invalid DataFrame provided")
            return {}
        
        # Check cache first — use hash of ALL column names to avoid collisions
        import hashlib as _hl
        # Cache key based on ALL columns to ensure uniqueness
        cache_key = _hl.md5(','.join(sorted(df.columns)).encode()).hexdigest()
        if cache_key in self._cached_mappings:
            return self._cached_mappings[cache_key]
        
        mappings = {}
        
        # Merge custom patterns with defaults
        all_patterns = {**self.CONCEPT_PATTERNS, **self._custom_patterns}
        
        for col in df.columns:
            # Normalize: replace separators with spaces, handle special chars
            col_lower = col.lower()
            for sep in ['_', '-', '.', '/', '(', ')', '%', ',']:
                col_lower = col_lower.replace(sep, ' ')
            col_tokens = set(col_lower.split())
            
            # Try to match against known patterns with priority scoring
            matched_concept = None
            best_score = 0
            
            # Special case: If 'date' or 'time' is in column, prioritize date detection
            if 'date' in col_lower or 'time' in col_lower:
                for pattern in all_patterns.get('date', []):
                    if pattern in col_lower:
                        matched_concept = 'date'
                        best_score = 999  # Very high score to override
                        break
            
            for concept, patterns in all_patterns.items():
                priority = self.CONCEPT_PRIORITY.get(concept, 5)
                
                for pattern in patterns:
                    # Calculate match score
                    if pattern in col_lower:
                        # Exact token match = highest score
                        if pattern in col_tokens:
                            score = len(pattern) * 2 * priority
                        # Substring match = lower score
                        else:
                            score = len(pattern) * priority
                        
                        if score > best_score:
                            matched_concept = concept
                            best_score = score
            
            # Fallback: Infer from dtype if no pattern match
            if matched_concept is None:
                try:
                    # Check if column name is truly generic (no recognizable words)
                    is_generic = len([t for t in col_tokens if len(t) > 2]) == 0  # Only short tokens
                    
                    dtype = str(df[col].dtype)
                    if 'datetime' in dtype:
                        matched_concept = 'date'
                    elif len(df) > 0 and not is_generic:  # Only infer from data if we have rows AND meaningful name
                        if 'float' in dtype or 'int' in dtype:
                            # Check if it's a count (all integers) or continuous
                            if df[col].dtype in ['int64', 'int32']:
                                matched_concept = 'count'
                            else:
                                matched_concept = 'numeric'
                        elif 'object' in dtype or 'category' in dtype:
                            # Check uniqueness ratio to infer category vs id
                            uniqueness = df[col].nunique() / len(df)
                            if uniqueness > 0.9:
                                matched_concept = 'id'
                            else:
                                matched_concept = 'category'
                        else:
                            matched_concept = 'unknown'
                    else:
                        # Generic name or no data - mark as unknown
                        matched_concept = 'unknown'
                except Exception as e:
                    logger.warning(f"Error inferring concept for column {col}: {e}")
                    matched_concept = 'unknown'
            
            mappings[col] = matched_concept
        
        # Cache the result
        self._cached_mappings[cache_key] = mappings
        logger.debug(f"Inferred {len(mappings)} column concepts: {mappings}")
        
        return mappings
    
    def get_columns_for_concept(self, df: pd.DataFrame, concept: str) -> List[str]:
        """
        Get all columns that map to a given concept.
        
        Args:
            df: DataFrame to analyze
            concept: Target concept (e.g., 'revenue', 'count')
            
        Returns:
            List of column names matching the concept
        """
        if df is None or df.empty:
            return []
        
        mappings = self.infer_column_concepts(df)
        return [col for col, mapped_concept in mappings.items() if mapped_concept == concept]
    
    def get_concept_confidence(self, df: pd.DataFrame, column: str) -> float:
        """
        Get confidence score for a column's concept mapping.
        
        Args:
            df: DataFrame to analyze
            column: Column name to check
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if column not in df.columns:
            return 0.0
        
        col_lower = column.lower()
        # Normalize separators
        for sep in ['_', '-', '.', '/', '(', ')', '%', ',']:
            col_lower = col_lower.replace(sep, ' ')
        col_tokens = set(col_lower.split())
        
        # Merge custom patterns
        all_patterns = {**self.CONCEPT_PATTERNS, **self._custom_patterns}
        
        # Check pattern matches
        for concept, patterns in all_patterns.items():
            for pattern in patterns:
                if pattern in col_tokens:
                    return 1.0  # Exact token match = high confidence
                elif pattern in col_lower:
                    return 0.7  # Substring match = medium confidence
        
        # Dtype-only inference = lower confidence
        return 0.3
    
    def validate_mappings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate column concept mappings and return diagnostics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict with validation results and warnings
        """
        mappings = self.infer_column_concepts(df)
        
        # Count concepts
        concept_counts = {}
        for concept in mappings.values():
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        # Identify potential issues
        warnings = []
        
        if concept_counts.get('unknown', 0) > len(df.columns) * 0.3:
            warnings.append(f"High number of unknown concepts: {concept_counts.get('unknown', 0)}/{len(df.columns)}")
        
        if len([c for c in mappings.values() if c in ['revenue', 'cost', 'profit']]) == 0:
            warnings.append("No financial columns detected - ensure column names are descriptive")
        
        if concept_counts.get('id', 0) > 5:
            warnings.append(f"Many ID columns detected ({concept_counts.get('id', 0)}) - consider excluding some")
        
        # Check confidence
        low_confidence_cols = []
        for col in df.columns:
            if self.get_concept_confidence(df, col) < 0.5:
                low_confidence_cols.append(col)
        
        return {
            'mappings': mappings,
            'concept_counts': concept_counts,
            'warnings': warnings,
            'low_confidence_columns': low_confidence_cols,
            'total_columns': len(df.columns),
            'mapped_columns': len([c for c in mappings.values() if c != 'unknown'])
        }
    
    def enhance_query_context(self, query: str, df: pd.DataFrame) -> str:
        """
        Enhance the query with column concept information and synonym expansion.
        This helps LLMs understand what columns to use for domain-agnostic queries.
        
        Example:
            Query: "What is the total revenue?"
            Enhanced: "What is the total revenue?\n[Column Concepts: revenue: gross_inflow, daily_sales]"
        
        Args:
            query: Original user query
            df: DataFrame being analyzed
            
        Returns:
            Enhanced query with concept hints
        """
        mappings = self.infer_column_concepts(df)
        
        # Group columns by concept
        concept_groups: Dict[str, List[str]] = {}
        for col, concept in mappings.items():
            if concept not in ['unknown', 'numeric']:
                if concept not in concept_groups:
                    concept_groups[concept] = []
                concept_groups[concept].append(col)
        
        # Synonym expansion: map query terms to detected column concepts
        query_lower = query.lower()
        synonym_hints = self._expand_query_synonyms(query_lower, mappings, df)
        
        # Build concept info string
        concept_info = []
        for concept, cols in concept_groups.items():
            if cols and concept in ['revenue', 'cost', 'profit', 'count', 'rate', 'performance', 'date', 'category', 'location']:
                concept_info.append(f"{concept}: {', '.join(cols[:3])}")  # Limit to 3 columns
        
        parts = []
        if concept_info:
            parts.append(f"[Column Concepts: {'; '.join(concept_info)}]")
        if synonym_hints:
            parts.append(f"[Column Hints: {'; '.join(synonym_hints)}]")
        
        if parts:
            enhanced = f"{query}\n\n{chr(10).join(parts)}"
            logger.debug(f"Enhanced query with {len(concept_info)} concepts, {len(synonym_hints)} synonym hints")
            return enhanced
        
        return query
    
    def _expand_query_synonyms(self, query_lower: str, mappings: Dict[str, str], df: pd.DataFrame) -> List[str]:
        """
        Expand query terms using detected column names as synonyms.
        Maps user's natural language terms to actual column names in the data.
        
        Example: User says "earnings" but column is "gross_inflow" -> hint: "earnings -> gross_inflow"
        """
        QUERY_SYNONYMS = {
            'revenue': ['revenue', 'sales', 'income', 'earnings', 'money', 'billing'],
            'cost': ['cost', 'expense', 'spending', 'expenditure', 'outflow'],
            'profit': ['profit', 'margin', 'gain', 'net income', 'bottom line'],
            'count': ['count', 'number', 'how many', 'total', 'volume', 'frequency'],
            'date': ['date', 'time', 'when', 'period', 'month', 'year', 'quarter', 'day'],
            'category': ['category', 'type', 'group', 'segment', 'by', 'per'],
            'rate': ['rate', 'percentage', 'ratio', 'growth', 'change', 'trend'],
            'location': ['location', 'region', 'city', 'country', 'area', 'where'],
        }
        
        hints = []
        matched_concepts = set()
        
        for concept, synonyms in QUERY_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in query_lower and concept not in matched_concepts:
                    # Find actual columns for this concept
                    matching_cols = [col for col, c in mappings.items() if c == concept]
                    if matching_cols:
                        # Only hint if the synonym doesn't exactly match a column name
                        col_names_lower = [col.lower() for col in matching_cols]
                        if synonym not in col_names_lower:
                            hints.append(f"'{synonym}' likely refers to column(s): {', '.join(matching_cols[:2])}")
                            matched_concepts.add(concept)
                    break
        
        return hints
    
    def suggest_operations(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Suggest relevant operations based on query and detected concepts.
        
        Args:
            query: User's natural language query
            df: DataFrame being analyzed
            
        Returns:
            Dict with suggested operations and relevant columns
        """
        query_lower = query.lower()
        mappings = self.infer_column_concepts(df)
        
        suggestions = {
            'detected_concepts': mappings,
            'relevant_columns': [],
            'suggested_operation': None
        }
        
        # Detect operation type from query
        if any(word in query_lower for word in ['total', 'sum', 'aggregate']):
            # Look for revenue/count columns
            relevant = self.get_columns_for_concept(df, 'revenue') + \
                      self.get_columns_for_concept(df, 'count')
            suggestions['relevant_columns'] = relevant
            suggestions['suggested_operation'] = 'sum'
            
        elif any(word in query_lower for word in ['average', 'mean', 'avg']):
            relevant = self.get_columns_for_concept(df, 'revenue') + \
                      self.get_columns_for_concept(df, 'rate') + \
                      self.get_columns_for_concept(df, 'performance')
            suggestions['relevant_columns'] = relevant
            suggestions['suggested_operation'] = 'mean'
            
        elif any(word in query_lower for word in ['trend', 'over time', 'growth']):
            date_cols = self.get_columns_for_concept(df, 'date')
            value_cols = self.get_columns_for_concept(df, 'revenue') + \
                        self.get_columns_for_concept(df, 'count')
            suggestions['relevant_columns'] = {'date': date_cols, 'value': value_cols}
            suggestions['suggested_operation'] = 'time_series'
            
        elif any(word in query_lower for word in ['by category', 'by type', 'breakdown']):
            cat_cols = self.get_columns_for_concept(df, 'category')
            suggestions['relevant_columns'] = cat_cols
            suggestions['suggested_operation'] = 'groupby'
        
        return suggestions


# Singleton accessor
_mapper_instance: Optional[SemanticMapper] = None

def get_semantic_mapper() -> SemanticMapper:
    """
    Get or create the singleton SemanticMapper instance.
    
    Returns:
        SemanticMapper instance
    """
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = SemanticMapper()
        logger.debug("Created new SemanticMapper singleton")
    return _mapper_instance


# ============================================================================
# Enterprise v2.0 — MappingAuditEntry & MappingAuditLog
# ============================================================================

import threading as _threading
from dataclasses import dataclass, field as _field
import datetime as _datetime


@dataclass
class MappingAuditEntry:
    """Record of a single semantic-mapping decision.

    Attributes:
        query: The original user query.
        matched_concept: Concept key that was matched (or ``None``).
        confidence: Mapping confidence score (0.0–1.0).
        timestamp: ISO-8601 creation timestamp.

    .. versionadded:: 2.0
    """

    query: str
    matched_concept: str | None
    confidence: float
    timestamp: str = _field(
        default_factory=lambda: _datetime.datetime.now().isoformat()
    )


class MappingAuditLog:
    """Thread-safe rolling log of semantic-mapping decisions.

    Stores up to *max_entries* audit records and provides summary
    statistics for observability.

    Args:
        max_entries: Maximum entries to retain (FIFO eviction).

    Example::

        audit = MappingAuditLog()
        audit.record(MappingAuditEntry("show sales trend", "trend_analysis", 0.92))
        print(audit.summary())

    .. versionadded:: 2.0
    """

    def __init__(self, max_entries: int = 5_000) -> None:
        self._entries: list[MappingAuditEntry] = []
        self._lock = _threading.Lock()
        self._max = max_entries

    def record(self, entry: MappingAuditEntry) -> None:
        """Append an audit entry, evicting oldest if at capacity."""
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max:
                self._entries = self._entries[-self._max:]

    def summary(self) -> dict:
        """Return aggregate statistics over stored entries.

        Returns:
            Dict with ``total``, ``avg_confidence``, and
            ``unmatched_count`` keys.
        """
        with self._lock:
            total = len(self._entries)
            if total == 0:
                return {"total": 0, "avg_confidence": 0.0, "unmatched_count": 0}
            avg_conf = sum(e.confidence for e in self._entries) / total
            unmatched = sum(1 for e in self._entries if e.matched_concept is None)
            return {
                "total": total,
                "avg_confidence": round(avg_conf, 4),
                "unmatched_count": unmatched,
            }
