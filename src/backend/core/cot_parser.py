"""
Chain-of-Thought Parser
Extracts and validates CoT reasoning from LLM responses
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

class CoTSection(Enum):
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

class CoTParser:
    """Parse and validate CoT-structured responses"""
    
    def __init__(self, reasoning_start="[REASONING]", reasoning_end="[/REASONING]",
                 output_start="[OUTPUT]", output_end="[/OUTPUT]"):
        self.reasoning_start = reasoning_start
        self.reasoning_end = reasoning_end
        self.output_start = output_start
        self.output_end = output_end
    
    def parse(self, response: str) -> ParsedCoT:
        """
        Extract reasoning and output sections from LLM response
        
        Args:
            response: Raw LLM response string
            
        Returns:
            ParsedCoT object with extracted sections
        """
        # Extract reasoning section
        reasoning_pattern = f"{re.escape(self.reasoning_start)}(.*?){re.escape(self.reasoning_end)}"
        reasoning_match = re.search(reasoning_pattern, response, re.DOTALL | re.IGNORECASE)
        
        # Extract output section
        output_pattern = f"{re.escape(self.output_start)}(.*?){re.escape(self.output_end)}"
        output_match = re.search(output_pattern, response, re.DOTALL | re.IGNORECASE)
        
        # Validation
        if not reasoning_match:
            return ParsedCoT(
                reasoning="",
                output=response,  # Fallback to entire response
                is_valid=False,
                error_message="Missing [REASONING] section",
                raw_response=response
            )
        
        if not output_match:
            return ParsedCoT(
                reasoning=reasoning_match.group(1).strip(),
                output="",
                is_valid=False,
                error_message="Missing [OUTPUT] section",
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
    
    def extract_steps(self, reasoning: str) -> List[str]:
        """Extract individual reasoning steps"""
        # Look for "Step N:" patterns
        step_pattern = r"Step\s+\d+:.*?(?=Step\s+\d+:|$)"
        steps = re.findall(step_pattern, reasoning, re.DOTALL | re.IGNORECASE)
        return [step.strip() for step in steps if step.strip()]

class CriticParser:
    """Parse critic model feedback"""
    
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
