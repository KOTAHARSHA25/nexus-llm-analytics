"""
Unit tests for Chain-of-Thought Parser
Tests CoTParser and CriticParser classes
"""
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from backend.core.cot_parser import CoTParser, CriticParser, ParsedCoT, CriticFeedback, CriticIssue


class TestCoTParser:
    """Test CoTParser class"""
    
    def test_valid_cot_parsing(self):
        """Test parsing valid CoT response"""
        parser = CoTParser()
        response = """
        [REASONING]
        Step 1: Analyze query
        First, I need to understand what data analysis is required.
        
        Step 2: Calculate total
        Sum all the sales values in the dataset.
        
        Step 3: Verify result
        Check that the calculation is correct.
        [/REASONING]
        
        [OUTPUT]
        Total sales: $10,000
        [/OUTPUT]
        """
        
        result = parser.parse(response)
        assert result.is_valid
        assert "Step 1" in result.reasoning
        assert "Step 2" in result.reasoning
        assert "$10,000" in result.output
        assert result.error_message is None
    
    def test_missing_reasoning_section(self):
        """Test parsing with missing REASONING section"""
        parser = CoTParser()
        response = "[OUTPUT]Some answer[/OUTPUT]"
        
        result = parser.parse(response)
        assert not result.is_valid
        assert "Missing [REASONING]" in result.error_message
    
    def test_missing_output_section(self):
        """Test parsing with missing OUTPUT section"""
        parser = CoTParser()
        response = "[REASONING]Some reasoning[/REASONING]"
        
        result = parser.parse(response)
        assert not result.is_valid
        assert "Missing [OUTPUT]" in result.error_message
    
    def test_reasoning_too_short(self):
        """Test parsing with too short reasoning section"""
        parser = CoTParser()
        response = """
        [REASONING]
        Short
        [/REASONING]
        
        [OUTPUT]
        Answer
        [/OUTPUT]
        """
        
        result = parser.parse(response)
        assert not result.is_valid
        assert "too short" in result.error_message
    
    def test_extract_steps(self):
        """Test extraction of reasoning steps"""
        parser = CoTParser()
        reasoning = """
        Step 1: First step description
        This is the detail for step 1.
        
        Step 2: Second step description
        This is the detail for step 2.
        
        Step 3: Third step description
        This is the detail for step 3.
        """
        
        steps = parser.extract_steps(reasoning)
        assert len(steps) == 3
        assert "Step 1" in steps[0]
        assert "Step 2" in steps[1]
        assert "Step 3" in steps[2]
    
    def test_case_insensitive_tags(self):
        """Test parsing with different case tags"""
        parser = CoTParser()
        response = """
        [reasoning]
        Step 1: First I need to analyze the data to understand what we're working with
        Step 2: Then calculate the total by summing all values
        Step 3: Finally verify the result is correct and makes sense
        [/reasoning]
        
        [output]
        Result: 100
        [/output]
        """
        
        result = parser.parse(response)
        assert result.is_valid
        assert "Step 1" in result.reasoning
        assert "100" in result.output


class TestCriticParser:
    """Test CriticParser class"""
    
    def test_valid_response(self):
        """Test parsing valid response"""
        parser = CriticParser()
        response = "[VALID]"
        
        result = parser.parse(response)
        assert result.is_valid
        assert len(result.issues) == 0
    
    def test_valid_with_explanation(self):
        """Test parsing valid response with explanation"""
        parser = CriticParser()
        response = """
        [VALID]
        
        The reasoning is sound and the calculations are correct.
        All assumptions are clearly stated.
        """
        
        result = parser.parse(response)
        assert result.is_valid
        assert len(result.issues) == 0
    
    def test_single_issue_found(self):
        """Test parsing response with one issue"""
        parser = CriticParser()
        response = """
        [ISSUES]
        Issue 1: Mathematical error in calculation
           Location: Step 3
           Severity: HIGH
           Suggestion: Recalculate the sum using correct values
        [/ISSUES]
        """
        
        result = parser.parse(response)
        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0].severity == "HIGH"
        assert "Step 3" in result.issues[0].location
        assert "Mathematical error" in result.issues[0].description
        assert "Recalculate" in result.issues[0].suggestion
    
    def test_multiple_issues_found(self):
        """Test parsing response with multiple issues"""
        parser = CriticParser()
        response = """
        [ISSUES]
        Issue 1: Logical inconsistency
           Location: Step 2
           Severity: MEDIUM
           Suggestion: Review the logical flow
        
        Issue 2: Missing assumption
           Location: Step 4
           Severity: LOW
           Suggestion: State the assumption explicitly
        
        Issue 3: Wrong method selected
           Location: Step 1
           Severity: HIGH
           Suggestion: Use mean instead of median
        [/ISSUES]
        """
        
        result = parser.parse(response)
        assert not result.is_valid
        assert len(result.issues) == 3
        
        # Check first issue
        assert "Logical inconsistency" in result.issues[0].description
        assert result.issues[0].severity == "MEDIUM"
        
        # Check second issue
        assert "Missing assumption" in result.issues[1].description
        assert result.issues[1].severity == "LOW"
        
        # Check third issue
        assert "Wrong method" in result.issues[2].description
        assert result.issues[2].severity == "HIGH"
    
    def test_missing_severity_defaults_to_medium(self):
        """Test that missing severity defaults to MEDIUM"""
        parser = CriticParser()
        response = """
        [ISSUES]
        Issue 1: Some problem
           Location: Step 2
           Suggestion: Fix it
        [/ISSUES]
        """
        
        result = parser.parse(response)
        assert len(result.issues) == 1
        assert result.issues[0].severity == "MEDIUM"
    
    def test_missing_location_defaults_to_unknown(self):
        """Test that missing location defaults to Unknown"""
        parser = CriticParser()
        response = """
        [ISSUES]
        Issue 1: Some problem
           Severity: LOW
           Suggestion: Fix it
        [/ISSUES]
        """
        
        result = parser.parse(response)
        assert len(result.issues) == 1
        assert result.issues[0].location == "Unknown"
    
    def test_missing_suggestion_has_default(self):
        """Test that missing suggestion has default text"""
        parser = CriticParser()
        response = """
        [ISSUES]
        Issue 1: Some problem
           Location: Step 1
           Severity: HIGH
        [/ISSUES]
        """
        
        result = parser.parse(response)
        assert len(result.issues) == 1
        assert "Review and correct" in result.issues[0].suggestion
    
    def test_case_insensitive_valid(self):
        """Test case insensitive parsing of VALID"""
        parser = CriticParser()
        
        # Test lowercase
        result1 = parser.parse("[valid]")
        assert result1.is_valid
        
        # Test mixed case
        result2 = parser.parse("[VaLiD]")
        assert result2.is_valid
        
        # Test uppercase
        result3 = parser.parse("[VALID]")
        assert result3.is_valid


class TestCoTParserEdgeCases:
    """Test edge cases for CoT Parser"""
    
    def test_empty_response(self):
        """Test parsing empty response"""
        parser = CoTParser()
        result = parser.parse("")
        assert not result.is_valid
    
    def test_malformed_tags(self):
        """Test parsing with malformed tags"""
        parser = CoTParser()
        response = """
        [REASONING
        Step 1: Missing closing bracket
        [/REASONING]
        
        [OUTPUT]
        Answer
        [/OUTPUT
        """
        
        result = parser.parse(response)
        assert not result.is_valid
    
    def test_nested_tags(self):
        """Test parsing with nested tags (should extract outer)"""
        parser = CoTParser()
        response = """
        [REASONING]
        Step 1: Outer reasoning
        [REASONING]
        Nested reasoning (should be ignored)
        [/REASONING]
        Step 2: More reasoning
        [/REASONING]
        
        [OUTPUT]
        Final answer
        [/OUTPUT]
        """
        
        result = parser.parse(response)
        # Should still parse successfully, extracting first match
        assert result.reasoning != ""
        assert result.output == "Final answer"


class TestCriticParserEdgeCases:
    """Test edge cases for Critic Parser"""
    
    def test_empty_response(self):
        """Test parsing empty response"""
        parser = CriticParser()
        result = parser.parse("")
        assert not result.is_valid
        assert len(result.issues) == 0
    
    def test_no_issues_or_valid_marker(self):
        """Test response with neither VALID nor ISSUES"""
        parser = CriticParser()
        response = "This is just some random text without markers"
        
        result = parser.parse(response)
        assert not result.is_valid
        # Should have no issues parsed
        assert len(result.issues) == 0
    
    def test_valid_in_middle_of_text(self):
        """Test VALID marker in middle of text"""
        parser = CriticParser()
        response = """
        After careful analysis, I found the reasoning to be [VALID]
        and all calculations are correct.
        """
        
        result = parser.parse(response)
        # Should still detect [VALID]
        assert result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
