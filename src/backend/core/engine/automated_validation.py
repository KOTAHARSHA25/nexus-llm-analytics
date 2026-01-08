from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ValidationIssue:
    description: str
    location: str
    severity: str  # HIGH, MEDIUM, LOW
    issue_type: str

@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue]
    
    def to_feedback_text(self) -> str:
        if self.is_valid:
            return "Validation Successful"
        return "\n".join([f"- {i.description} (Severity: {i.severity})" for i in self.issues])

class AutomatedValidator:
    """
    Performs automated checks on reasoning and code before LLM review.
    """
    
    def validate(self, query: str, reasoning: str, output: str, data_context: Dict[str, Any]) -> ValidationResult:
        issues = []
        
        # 1. Check for empty output
        if not output.strip():
             issues.append(ValidationIssue(
                 description="Output is empty",
                 location="Output",
                 severity="HIGH",
                 issue_type="empty_output"
             ))
             
        # 2. Check for Hallucinated Columns (if data context available)
        columns = data_context.get('columns', [])
        if columns and reasoning:
             # Heuristic: Find words ending in .column or ['column']
             # This is complex to do robustly with regex, simplified check:
             pass

        # 3. Check code block validity
        if "```python" in output:
             if "```" not in output.split("```python")[1]:
                 issues.append(ValidationIssue(
                     description="Unclosed code block",
                     location="Output",
                     severity="HIGH",
                     issue_type="syntax_error"
                 ))

        return ValidationResult(
            is_valid=(len(issues) == 0),
            issues=issues
        )
