from dataclasses import dataclass, field
from typing import List, Optional
import re
import logging

@dataclass
class CriticIssue:
    description: str
    location: str
    severity: str  # HIGH, MEDIUM, LOW
    suggestion: str

@dataclass
class CriticFeedback:
    is_valid: bool
    issues: List[CriticIssue]
    raw_response: str
    feedback: str = ""

@dataclass
class ParsedCoT:
    is_valid: bool
    reasoning: str
    output: str
    error_message: Optional[str] = None

class CoTParser:
    """Parser for Chain-of-Thought responses"""
    
    def __init__(self, reasoning_start: str, reasoning_end: str, output_start: str, output_end: str):
        self.r_start = reasoning_start
        self.r_end = reasoning_end
        self.o_start = output_start
        self.o_end = output_end
        
    def parse(self, text: str) -> ParsedCoT:
        """Parse LLM output into reasoning and final output components"""
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
        """Extract individual steps from reasoning text (heuristic)"""
        steps = []
        for line in reasoning.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line) or line.startswith('- '):
                steps.append(line)
        return steps

class CriticParser:
    """Parser for Critic responses"""
    
    def parse(self, text: str) -> CriticFeedback:
        """Parse critic feedback"""
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
