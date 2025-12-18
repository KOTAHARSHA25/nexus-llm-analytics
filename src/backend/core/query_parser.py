# Advanced Natural Language Query Parser for Data Analysis
# Converts natural language queries into structured analysis instructions

from typing import Dict, List, Any, Optional, Tuple
import re
import logging
from enum import Enum
from dataclasses import dataclass
from .llm_client import LLMClient

class QueryIntent(Enum):
    """Possible query intents for data analysis"""
    SUMMARIZE = "summarize"
    DESCRIBE = "describe"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    VISUALIZE = "visualize"
    COMPARE = "compare"
    TREND = "trend"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    OUTLIERS = "outliers"
    COUNT = "count"
    SEARCH = "search"
    CUSTOM = "custom"

class DataType(Enum):
    """Data types for columns"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"

@dataclass
class QueryParams:
    """Structured parameters extracted from natural language query"""
    intent: QueryIntent
    columns: List[str]
    conditions: List[Dict[str, Any]]
    aggregation: Optional[str]
    groupby: List[str]
    orderby: Optional[str]
    limit: Optional[int]
    chart_type: Optional[str]
    confidence: float

class IntentClassifier:
    """Classifies the intent of natural language queries"""
    
    # Intent patterns mapping
    INTENT_PATTERNS = {
        QueryIntent.SUMMARIZE: [
            r'\b(summary|summarize|overview|general|basic info|brief)\b',
            r'\b(what.*data|show.*data|tell me about)\b',
            r'\b(first.*rows|head|preview|sample)\b'
        ],
        QueryIntent.DESCRIBE: [
            r'\b(describe|statistics|stats|distribution|summary stats)\b',
            r'\b(mean|average|median|std|variance|min|max)\b',
            r'\b(statistical.*summary|descriptive.*analysis)\b'
        ],
        QueryIntent.FILTER: [
            r'\b(filter|where|show.*only|records.*where|rows.*where)\b',
            r'\b(equal|equals|greater|less|contains|like|match)\b',
            r'\b(find.*records|select.*where|get.*rows)\b'
        ],
        QueryIntent.AGGREGATE: [
            r'\b(group by|aggregate|sum|count|average|total)\b',
            r'\b(by.*category|per.*group|for each)\b',
            r'\b(breakdown|group.*analysis)\b'
        ],
        QueryIntent.VISUALIZE: [
            r'\b(chart|graph|plot|visualization|visualize|show.*chart)\b',
            r'\b(bar.*chart|line.*chart|scatter|histogram|pie.*chart)\b',
            r'\b(create.*chart|generate.*plot|make.*graph)\b'
        ],
        QueryIntent.COMPARE: [
            r'\b(compare|comparison|vs|versus|against|difference)\b',
            r'\b(correlation|relationship|association)\b',
            r'\b(how.*different|what.*difference)\b'
        ],
        QueryIntent.TREND: [
            r'\b(trend|over time|time series|temporal|timeline)\b',
            r'\b(growth|decline|change.*over|pattern.*time)\b',
            r'\b(month|year|day|week|quarter|seasonal)\b'
        ],
        QueryIntent.COUNT: [
            r'\b(count|number of|how many|total.*records)\b',
            r'\b(frequency|occurrences|instances)\b'
        ],
        QueryIntent.OUTLIERS: [
            r'\b(outlier|anomal|unusual|extreme|abnormal)\b',
            r'\b(detect.*outlier|find.*unusual|identify.*anomal)\b'
        ]
    }
    
    @classmethod
    def classify_intent(cls, query: str) -> Tuple[QueryIntent, float]:
        """
        Classify the intent of a natural language query
        
        Returns:
            Tuple of (intent, confidence_score)
        """
        query_lower = query.lower()
        best_intent = QueryIntent.CUSTOM
        best_score = 0.0
        
        for intent, patterns in cls.INTENT_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches * 0.3  # Each match adds to confidence
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        # Normalize confidence score
        confidence = min(best_score, 1.0)
        
        return best_intent, confidence

class ColumnExtractor:
    """Extracts column names and conditions from natural language queries"""
    
    @staticmethod
    def extract_columns(query: str, available_columns: List[str]) -> List[str]:
        """Extract column names mentioned in the query"""
        query_lower = query.lower()
        found_columns = []
        
        for col in available_columns:
            col_lower = col.lower()
            # Check for exact matches and partial matches
            if col_lower in query_lower or any(word in query_lower for word in col_lower.split('_')):
                found_columns.append(col)
        
        return found_columns
    
    @staticmethod
    def extract_conditions(query: str, columns: List[str]) -> List[Dict[str, Any]]:
        """Extract filter conditions from the query"""
        conditions = []
        query_lower = query.lower()
        
        # Updated patterns with consistent grouping
        condition_patterns = [
            (r'(\w+)\s*(=|equals?|is)\s*(["\']?)([^"\']+)\3', 4),  # (column, operator, quote, value)
            (r'(\w+)\s*(>|greater\s+than)\s*(\d+\.?\d*)', 3),     # (column, operator, value)
            (r'(\w+)\s*(<|less\s+than)\s*(\d+\.?\d*)', 3),        # (column, operator, value)
            (r'(\w+)\s*(contains?|includes?|like)\s*(["\']?)([^"\']+)\3', 4)  # (column, operator, quote, value)
        ]
        
        for pattern, expected_groups in condition_patterns:
            try:
                matches = re.finditer(pattern, query_lower)
                for match in matches:
                    if len(match.groups()) >= expected_groups:
                        column = match.group(1)
                        operator = match.group(2)
                        # Get value based on pattern structure
                        if expected_groups == 4:
                            value = match.group(4)  # value is in group 4 for quoted patterns
                        else:
                            value = match.group(3)  # value is in group 3 for numeric patterns
                        
                        # Map natural language operators to SQL-like operators
                        op_mapping = {
                            '=': '==', 'equals': '==', 'equal': '==', 'is': '==',
                            '>': '>', 'greater than': '>',
                            '<': '<', 'less than': '<',
                            'contains': 'contains', 'includes': 'contains', 'like': 'contains'
                        }
                        
                        conditions.append({
                            'column': column,
                            'operator': op_mapping.get(operator, operator),
                            'value': value.strip() if value else ""
                        })
            except Exception as regex_error:
                logging.warning(f"Regex pattern matching error: {regex_error} for pattern: {pattern}")
                continue
        
        return conditions

class AdvancedQueryParser:
    """Advanced natural language query parser using LLM assistance"""
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()
        self.intent_classifier = IntentClassifier()
        self.column_extractor = ColumnExtractor()
    
    def parse_query(self, query: str, available_columns: List[str] = None, 
                   data_sample: Dict[str, Any] = None) -> QueryParams:
        """
        Parse a natural language query into structured parameters
        
        Args:
            query: Natural language query string
            available_columns: List of available column names
            data_sample: Sample of the data for context
            
        Returns:
            QueryParams object with structured query information
        """
        logging.debug(f"Parsing query: {query}")
        
        try:
            # Step 1: Classify intent using pattern matching
            intent, confidence = self.intent_classifier.classify_intent(query)
            
            # Step 2: Use LLM for enhanced parsing if confidence is low
            if confidence < 0.5 or intent == QueryIntent.CUSTOM:
                llm_result = self._llm_enhanced_parsing(query, available_columns, data_sample)
                if llm_result:
                    intent = QueryIntent(llm_result.get('intent', 'custom'))
                    confidence = llm_result.get('confidence', confidence)
            
            # Step 3: Extract columns and conditions
            columns = []
            conditions = []
            
            if available_columns:
                columns = self.column_extractor.extract_columns(query, available_columns)
                conditions = self.column_extractor.extract_conditions(query, available_columns)
            
            # Step 4: Extract additional parameters
            params = self._extract_additional_params(query)
            
            result = QueryParams(
                intent=intent,
                columns=columns,
                conditions=conditions,
                aggregation=params.get('aggregation'),
                groupby=params.get('groupby', []),
                orderby=params.get('orderby'),
                limit=params.get('limit'),
                chart_type=params.get('chart_type'),
                confidence=confidence
            )
            
            logging.debug(f"Parsed query result: intent={intent.value}, confidence={confidence:.2f}")
            return result
            
        except Exception as e:
            logging.error(f"Query parsing failed: {e}")
            # Return default custom query
            return QueryParams(
                intent=QueryIntent.CUSTOM,
                columns=[],
                conditions=[],
                aggregation=None,
                groupby=[],
                orderby=None,
                limit=None,
                chart_type=None,
                confidence=0.0
            )
    
    def _llm_enhanced_parsing(self, query: str, columns: List[str] = None, 
                            data_sample: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Use LLM to enhance query parsing"""
        try:
            prompt = f"""
            You are an expert data analyst. Parse this natural language query into structured parameters.
            
            Query: "{query}"
            Available columns: {columns or 'Unknown'}
            Data sample: {str(data_sample)[:500] if data_sample else 'Not provided'}
            
            Determine:
            1. Primary intent (summarize, describe, filter, aggregate, visualize, compare, trend, count, outliers, or custom)
            2. Relevant columns mentioned
            3. Any filter conditions
            4. Aggregation type if applicable
            5. Chart type if visualization is requested
            6. Confidence score (0.0 to 1.0)
            
            Respond in JSON format:
            {{
                "intent": "intent_name",
                "columns": ["col1", "col2"],
                "conditions": [{{"column": "col", "operator": "==", "value": "val"}}],
                "aggregation": "sum|count|avg|etc or null",
                "chart_type": "bar|line|scatter|etc or null",
                "confidence": 0.8
            }}
            """
            
            response = self.llm_client.generate_primary(prompt)
            
            import json
            result = json.loads(response.get('response', '{}'))
            return result
            
        except Exception as e:
            logging.warning(f"LLM enhanced parsing failed: {e}")
            return None
    
    def _extract_additional_params(self, query: str) -> Dict[str, Any]:
        """Extract additional parameters like limit, order by, etc."""
        params = {}
        query_lower = query.lower()
        
        # Extract limit
        limit_match = re.search(r'\b(top|first|limit)\s+(\d+)', query_lower)
        if limit_match:
            params['limit'] = int(limit_match.group(2))
        
        # Extract aggregation functions
        agg_patterns = {
            'sum': r'\b(sum|total)\b',
            'count': r'\b(count|number)\b',
            'mean': r'\b(average|mean|avg)\b',
            'max': r'\b(maximum|max|highest)\b',
            'min': r'\b(minimum|min|lowest)\b'
        }
        
        for agg, pattern in agg_patterns.items():
            if re.search(pattern, query_lower):
                params['aggregation'] = agg
                break
        
        # Extract chart type
        chart_patterns = {
            'bar': r'\b(bar.*chart|bar.*graph)\b',
            'line': r'\b(line.*chart|line.*graph|time.*series)\b',
            'scatter': r'\b(scatter.*plot|scatter.*chart)\b',
            'pie': r'\b(pie.*chart|pie.*graph)\b',
            'histogram': r'\b(histogram|distribution)\b'
        }
        
        for chart, pattern in chart_patterns.items():
            if re.search(pattern, query_lower):
                params['chart_type'] = chart
                break
        
        return params
    
    def generate_analysis_plan(self, parsed_query: QueryParams) -> Dict[str, Any]:
        """Generate an execution plan based on parsed query"""
        plan = {
            'steps': [],
            'estimated_time': 0,
            'complexity': 'low'
        }
        
        # Determine steps based on intent
        if parsed_query.intent == QueryIntent.SUMMARIZE:
            plan['steps'] = ['load_data', 'basic_summary', 'preview_data']
            plan['estimated_time'] = 5
        
        elif parsed_query.intent == QueryIntent.DESCRIBE:
            plan['steps'] = ['load_data', 'statistical_summary', 'data_types']
            plan['estimated_time'] = 8
        
        elif parsed_query.intent == QueryIntent.FILTER:
            plan['steps'] = ['load_data', 'apply_filters', 'show_results']
            plan['estimated_time'] = 6
        
        elif parsed_query.intent == QueryIntent.VISUALIZE:
            plan['steps'] = ['load_data', 'prepare_visualization', 'generate_chart']
            plan['estimated_time'] = 12
            plan['complexity'] = 'medium'
        
        elif parsed_query.intent == QueryIntent.AGGREGATE:
            plan['steps'] = ['load_data', 'group_data', 'calculate_aggregation', 'format_results']
            plan['estimated_time'] = 10
            plan['complexity'] = 'medium'
        
        else:  # Custom or complex queries
            plan['steps'] = ['load_data', 'analyze_query', 'generate_code', 'execute_analysis']
            plan['estimated_time'] = 15
            plan['complexity'] = 'high'
        
        return plan


# Alias for backward compatibility
EnhancedQueryParser = AdvancedQueryParser