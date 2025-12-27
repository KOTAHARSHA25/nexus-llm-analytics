"""
Automated Validation - Pre-checks before LLM Critic
Catches obvious errors regardless of critic model quality
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ValidationIssue:
    """Single validation issue found"""
    severity: str  # "HIGH", "MEDIUM", "LOW"
    issue_type: str
    description: str
    location: str


@dataclass
class AutomatedValidationResult:
    """Result of automated validation"""
    is_valid: bool
    issues: List[ValidationIssue]
    
    def to_feedback_text(self) -> str:
        """Convert to feedback text for generator"""
        if not self.issues:
            return ""
        
        feedback = "[INVALID]\n<feedback>\n"
        for i, issue in enumerate(self.issues, 1):
            feedback += f"Issue {i}: {issue.description}\n"
            feedback += f"   Location: {issue.location}\n"
            feedback += f"   Severity: {issue.severity}\n"
            feedback += f"   Fix: {issue.issue_type}\n\n"
        feedback += "</feedback>"
        return feedback


class AutomatedValidator:
    """
    Automated validation checks that run BEFORE LLM critic
    Catches obvious errors that even weak models might miss
    """
    
    def __init__(self):
        self.calculation_patterns = {
            r'2\s*\+\s*2\s*=\s*5': "Calculation error: 2+2=4, not 5",
            r'10\s*\+\s*10\s*=\s*25': "Calculation error: 10+10=20, not 25",
            r'5\s*\*\s*5\s*=\s*30': "Calculation error: 5*5=25, not 30",
        }
        
        self.logic_patterns = {
            # Original patterns
            r'age\s*<\s*30.*above\s*30': "Logic error: Should use 'age > 30' to get above 30",
            r'age\s*>\s*30.*below\s*30': "Logic error: Should use 'age < 30' to get below 30",
            r'filter.*!=.*include\s+only': "Logic error: != excludes, should use = to include",
            # NEW: Generic inverted logic patterns
            r'use.*<.*to\s+(get|find|filter).*greater|above|more': "Logic error: Using < for greater/above values",
            r'use.*>.*to\s+(get|find|filter).*less|below|fewer': "Logic error: Using > for less/below values",
            r'condition\s+\w+\s*<\s*\d+.*above|greater': "Logic error: Inverted comparison operator",
            r'condition\s+\w+\s*>\s*\d+.*below|less': "Logic error: Inverted comparison operator",
        }
        
        # NEW: Causation vs Correlation patterns
        self.causation_patterns = [
            (r'correlation.*cause[sd]?', "Correlation does not prove causation"),
            (r'correlat\w+\s+(show|prove|demonstrate)s?\s+(that|cause)', "Correlation doesn't prove causation"),
            (r'since.*correlat\w+.*cause[sd]?', "Correlation doesn't imply causation"),
            (r'high\s+correlation.*therefore.*cause', "High correlation doesn't mean causation"),
        ]
        
        # NEW: Time period errors
        self.quarter_months = {
            'q1': [1, 2, 3],      # Jan, Feb, Mar
            'q2': [4, 5, 6],      # Apr, May, Jun
            'q3': [7, 8, 9],      # Jul, Aug, Sep
            'q4': [10, 11, 12],   # Oct, Nov, Dec
        }
        
        # NEW: Formula errors
        self.formula_patterns = {
            r'profit\s*margin\s*=\s*revenue\s*/\s*cost': "Wrong formula: Profit margin = profit/revenue, not revenue/cost",
            r'profit\s*margin\s*=\s*cost\s*/\s*revenue': "Wrong formula: Profit margin = profit/revenue, not cost/revenue",
            r'growth\s*rate\s*=.*current\s*-\s*previous\s*$': "Missing division: Growth rate = (current-previous)/previous",
        }
        
        # NEW: Percentage expression errors
        self.percentage_patterns = [
            (r'percentage.*=.*0\.\d+\s*$', "Percentage should be multiplied by 100 (e.g., 0.75 → 75%)"),
            (r'\d+\s*/\s*\d+\s*=\s*0\.\d+', "When expressing as percentage, multiply by 100"),
        ]
    
    def validate(self, 
                query: str,
                reasoning: str, 
                output: str,
                data_context: Optional[Dict] = None) -> AutomatedValidationResult:
        """
        Run automated validation checks
        
        Args:
            query: Original user query
            reasoning: LLM's reasoning steps
            output: LLM's final output
            data_context: Optional data context
        
        Returns:
            AutomatedValidationResult with found issues
        """
        issues = []
        
        # Check 1: Calculation errors
        issues.extend(self._check_calculations(reasoning, output))
        
        # Check 2: Logic errors
        issues.extend(self._check_logic_errors(query, reasoning))
        
        # Check 3: Empty or too short reasoning
        issues.extend(self._check_reasoning_quality(reasoning))
        
        # Check 4: Output doesn't match query intent
        issues.extend(self._check_query_alignment(query, output))
        
        # Check 5: Data-specific validations
        if data_context:
            issues.extend(self._check_data_consistency(reasoning, output, data_context))
        
        # NEW Check 6: Causation vs correlation confusion
        issues.extend(self._check_causation_errors(query, reasoning, output))
        
        # NEW Check 7: Formula errors
        issues.extend(self._check_formula_errors(reasoning, output))
        
        # NEW Check 8: Time period errors
        issues.extend(self._check_time_period_errors(query, reasoning))
        
        # NEW Check 9: Missing filter/group operations
        issues.extend(self._check_aggregation_errors(query, reasoning))
        
        # NEW Check 10: Percentage format errors
        issues.extend(self._check_percentage_format(query, reasoning, output))
        
        # Valid if no HIGH severity issues
        high_severity_count = sum(1 for issue in issues if issue.severity == "HIGH")
        is_valid = high_severity_count == 0
        
        return AutomatedValidationResult(
            is_valid=is_valid,
            issues=issues
        )
    
    def _check_calculations(self, reasoning: str, output: str) -> List[ValidationIssue]:
        """Check for obvious calculation errors"""
        issues = []
        text = reasoning + " " + output
        
        for pattern, error_msg in self.calculation_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(ValidationIssue(
                    severity="HIGH",
                    issue_type="calculation_error",
                    description=error_msg,
                    location="reasoning/output"
                ))
        
        # Check for division by zero mentions
        if re.search(r'divid\w*\s+by\s+0', text, re.IGNORECASE):
            issues.append(ValidationIssue(
                severity="HIGH",
                issue_type="division_by_zero",
                description="Cannot divide by zero",
                location="reasoning"
            ))
        
        # NEW: Check explicit addition sequences
        # Pattern: "100 + 200 = 300, 300 + 300 = 600, ..." and verify final total
        addition_sequences = re.findall(
            r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)', 
            text
        )
        for a, b, result in addition_sequences:
            try:
                if int(a) + int(b) != int(result):
                    issues.append(ValidationIssue(
                        severity="HIGH",
                        issue_type="arithmetic_error",
                        description=f"Calculation error: {a}+{b}={int(a)+int(b)}, not {result}",
                        location="reasoning"
                    ))
            except ValueError:
                pass
        
        # Check multiplication errors
        mult_sequences = re.findall(
            r'(\d+)\s*[x×\*]\s*(\d+)\s*=\s*(\d+)',
            text
        )
        for a, b, result in mult_sequences:
            try:
                if int(a) * int(b) != int(result):
                    issues.append(ValidationIssue(
                        severity="HIGH",
                        issue_type="arithmetic_error",
                        description=f"Calculation error: {a}×{b}={int(a)*int(b)}, not {result}",
                        location="reasoning"
                    ))
            except ValueError:
                pass
        
        return issues
    
    def _check_logic_errors(self, query: str, reasoning: str) -> List[ValidationIssue]:
        """Check for logical contradictions"""
        issues = []
        
        for pattern, error_msg in self.logic_patterns.items():
            if re.search(pattern, reasoning, re.IGNORECASE):
                issues.append(ValidationIssue(
                    severity="HIGH",
                    issue_type="logic_error",
                    description=error_msg,
                    location="reasoning"
                ))
        
        # Check for contradictions between query and reasoning
        query_lower = query.lower()
        reasoning_lower = reasoning.lower()
        
        if 'greater than' in query_lower or '>' in query_lower:
            if '<' in reasoning_lower and '>' not in reasoning_lower:
                issues.append(ValidationIssue(
                    severity="MEDIUM",
                    issue_type="inverted_logic",
                    description="Query asks for 'greater than' but reasoning uses '<'",
                    location="reasoning"
                ))
        
        if 'less than' in query_lower or '<' in query_lower:
            if '>' in reasoning_lower and '<' not in reasoning_lower:
                issues.append(ValidationIssue(
                    severity="MEDIUM",
                    issue_type="inverted_logic",
                    description="Query asks for 'less than' but reasoning uses '>'",
                    location="reasoning"
                ))
        
        return issues
    
    def _check_reasoning_quality(self, reasoning: str) -> List[ValidationIssue]:
        """Check if reasoning is substantive"""
        issues = []
        
        # Too short reasoning
        if len(reasoning.strip()) < 20:
            issues.append(ValidationIssue(
                severity="MEDIUM",
                issue_type="insufficient_reasoning",
                description="Reasoning is too brief (< 20 characters)",
                location="reasoning"
            ))
        
        # No step-by-step explanation
        if len(reasoning.split('.')) < 2:
            issues.append(ValidationIssue(
                severity="LOW",
                issue_type="lacks_steps",
                description="Reasoning should break down into multiple steps",
                location="reasoning"
            ))
        
        return issues
    
    def _check_query_alignment(self, query: str, output: str) -> List[ValidationIssue]:
        """Check if output addresses the query"""
        issues = []
        
        query_lower = query.lower()
        output_lower = output.lower()
        
        # Check for key query terms in output
        query_keywords = self._extract_keywords(query_lower)
        output_keywords = self._extract_keywords(output_lower)
        
        overlap = len(query_keywords & output_keywords)
        if overlap == 0 and len(query_keywords) > 0:
            issues.append(ValidationIssue(
                severity="MEDIUM",
                issue_type="misaligned_output",
                description="Output doesn't mention any key terms from the query",
                location="output"
            ))
        
        # Check if query asks for specific format
        if 'average' in query_lower and 'average' not in output_lower:
            issues.append(ValidationIssue(
                severity="MEDIUM",
                issue_type="missing_calculation",
                description="Query asks for average but output doesn't mention it",
                location="output"
            ))
        
        if 'total' in query_lower and 'total' not in output_lower and 'sum' not in output_lower:
            issues.append(ValidationIssue(
                severity="MEDIUM",
                issue_type="missing_calculation",
                description="Query asks for total but output doesn't provide it",
                location="output"
            ))
        
        return issues
    
    def _check_data_consistency(self, 
                                reasoning: str,
                                output: str,
                                data_context: Dict) -> List[ValidationIssue]:
        """Check if reasoning is consistent with data context"""
        issues = []
        
        columns = data_context.get('columns', [])
        text = reasoning + " " + output
        
        # Check if mentioned columns actually exist
        mentioned_columns = re.findall(r'\b[A-Z][a-z_]+\b', text)
        for col in mentioned_columns:
            if col not in columns and len(columns) > 0:
                # Only flag if it looks like a column name
                if '_' in col or col.lower() in ['id', 'name', 'age', 'price', 'revenue']:
                    issues.append(ValidationIssue(
                        severity="LOW",
                        issue_type="nonexistent_column",
                        description=f"Column '{col}' mentioned but not in dataset",
                        location="reasoning"
                    ))
        
        return issues
    
    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text"""
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were'}
        words = re.findall(r'\b\w+\b', text.lower())
        return set(word for word in words if word not in stopwords and len(word) > 3)
    
    # ========== NEW VALIDATION METHODS ==========
    
    def _check_causation_errors(self, query: str, reasoning: str, output: str) -> List[ValidationIssue]:
        """Check for causation vs correlation confusion"""
        issues = []
        text = (reasoning + " " + output).lower()
        
        for pattern, error_msg in self.causation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(ValidationIssue(
                    severity="HIGH",
                    issue_type="causation_confusion",
                    description=error_msg,
                    location="reasoning"
                ))
        
        # Check query asks about causation
        query_lower = query.lower()
        if 'cause' in query_lower or 'causation' in query_lower:
            # If using correlation to prove causation
            if 'correlation' in text and ('therefore' in text or 'so' in text or 'hence' in text):
                if 'cause' in text:
                    issues.append(ValidationIssue(
                        severity="HIGH",
                        issue_type="causation_confusion",
                        description="Cannot conclude causation from correlation alone",
                        location="reasoning"
                    ))
        
        return issues
    
    def _check_formula_errors(self, reasoning: str, output: str) -> List[ValidationIssue]:
        """Check for incorrect formulas"""
        issues = []
        text = (reasoning + " " + output).lower()
        
        for pattern, error_msg in self.formula_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(ValidationIssue(
                    severity="HIGH",
                    issue_type="formula_error",
                    description=error_msg,
                    location="reasoning"
                ))
        
        return issues
    
    def _check_time_period_errors(self, query: str, reasoning: str) -> List[ValidationIssue]:
        """Check for incorrect quarter/time period references"""
        issues = []
        query_lower = query.lower()
        reasoning_lower = reasoning.lower()
        
        # Check for Q4 errors (common mistake)
        if 'q4' in query_lower:
            # Wrong months for Q4
            if any(m in reasoning_lower for m in ['month 9', 'september', 'months 9, 10, 11']):
                if not any(m in reasoning_lower for m in ['month 12', 'december']):
                    issues.append(ValidationIssue(
                        severity="HIGH",
                        issue_type="time_period_error",
                        description="Q4 is Oct/Nov/Dec (months 10-12), not Sep/Oct/Nov (months 9-11)",
                        location="reasoning"
                    ))
        
        # Check for Q1 errors
        if 'q1' in query_lower:
            if any(m in reasoning_lower for m in ['month 4', 'april', 'months 4, 5, 6']):
                issues.append(ValidationIssue(
                    severity="HIGH",
                    issue_type="time_period_error", 
                    description="Q1 is Jan/Feb/Mar (months 1-3), not Apr/May/Jun (months 4-6)",
                    location="reasoning"
                ))
        
        return issues
    
    def _check_aggregation_errors(self, query: str, reasoning: str) -> List[ValidationIssue]:
        """Check for missing filter/group operations"""
        issues = []
        query_lower = query.lower()
        reasoning_lower = reasoning.lower()
        
        # Check: Query asks for specific category but no filter mentioned
        category_keywords = ['category', 'region', 'department', 'type', 'group']
        filter_keywords = ['filter', 'where', 'subset', 'only', 'specific']
        
        # Look for category-specific queries
        for cat in category_keywords:
            if cat in query_lower:
                # Check if there's a specific value mentioned (e.g., "Electronics", "West")
                if "'" in query_lower or '"' in query_lower:
                    # Specific category requested
                    if not any(f in reasoning_lower for f in filter_keywords):
                        # No filtering mentioned
                        if 'sum' in reasoning_lower or 'total' in reasoning_lower or 'average' in reasoning_lower:
                            issues.append(ValidationIssue(
                                severity="MEDIUM",
                                issue_type="missing_filter",
                                description=f"Query asks for specific {cat} but reasoning may be summing all data",
                                location="reasoning"
                            ))
        
        # Check: Query asks "by region/category" but no group by
        group_keywords = ['group', 'grouped', 'per', 'by each', 'for each']
        if ' by ' in query_lower or ' per ' in query_lower:
            if not any(g in reasoning_lower for g in group_keywords):
                issues.append(ValidationIssue(
                    severity="MEDIUM",
                    issue_type="missing_groupby",
                    description="Query asks for breakdown 'by' category/region but grouping not mentioned",
                    location="reasoning"
                ))
        
        return issues
    
    def _check_percentage_format(self, query: str, reasoning: str, output: str) -> List[ValidationIssue]:
        """Check for percentage format errors (forgetting to multiply by 100)"""
        issues = []
        query_lower = query.lower()
        output_lower = output.lower()
        
        # Only check if query asks for percentage
        if 'percent' in query_lower or '%' in query_lower:
            # Check if output has decimal (0.xx) instead of percentage
            # Pattern: "0.75" without "%" following OR without "75%"
            import re
            
            # Find decimals in output
            decimals = re.findall(r'\b0\.\d{1,4}\b', output)
            
            for decimal in decimals:
                # Check if this looks like a forgotten percentage
                decimal_val = float(decimal)
                if 0.0 < decimal_val < 1.0:
                    # Check if the percentage equivalent isn't also mentioned
                    pct_val = int(decimal_val * 100)
                    if f'{pct_val}%' not in output and f'{pct_val} %' not in output:
                        if f'{pct_val} percent' not in output_lower:
                            issues.append(ValidationIssue(
                                severity="HIGH",
                                issue_type="percentage_format_error",
                                description=f"Query asks for percentage but output shows {decimal} instead of {pct_val}%",
                                location="output"
                            ))
                            break  # Only report once
        
        return issues
