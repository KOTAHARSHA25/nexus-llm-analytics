"""
Test Enhanced CoT Parser with Fallback Strategies
Tests multiple tag variations and content-based parsing
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.cot_parser import CoTParser

def test_exact_match():
    """Test original exact matching strategy"""
    parser = CoTParser()
    
    response = """
[REASONING]
This is my step-by-step reasoning process.
I will analyze the data carefully and consider multiple approaches.
First, I'll examine the dataset structure, then calculate the required metrics.
[/REASONING]

[OUTPUT]
The final answer is 42.
[/OUTPUT]
"""
    
    result = parser.parse(response)
    print(f"✓ Exact Match: Valid={result.is_valid}")
    assert result.is_valid, "Exact match should work"
    assert "step-by-step reasoning" in result.reasoning
    assert "final answer is 42" in result.output
    print(f"  Reasoning: {result.reasoning[:50]}...")
    print(f"  Output: {result.output[:50]}...")


def test_angle_brackets():
    """Test angle bracket tag variation"""
    parser = CoTParser()
    
    response = """
<reasoning>
Let me think through this problem systematically.
I need to understand the requirements, analyze the data, and provide a solution.
This requires careful consideration of all factors involved.
</reasoning>

<output>
The answer is: Calculate the mean of column A.
</output>
"""
    
    result = parser.parse(response)
    print(f"\n✓ Angle Brackets: Valid={result.is_valid}")
    assert result.is_valid, "Angle bracket tags should work"
    assert "systematically" in result.reasoning.lower()
    assert "mean of column A" in result.output
    print(f"  Reasoning: {result.reasoning[:50]}...")
    print(f"  Output: {result.output[:50]}...")


def test_uppercase_tags():
    """Test uppercase tag variation"""
    parser = CoTParser()
    
    response = """
<REASONING>
I will break this down into clear steps for maximum clarity.
Step 1: Load and inspect the data
Step 2: Identify relevant columns
Step 3: Perform the calculation
Step 4: Validate the results
</REASONING>

<OUTPUT>
Result: 156.78
</OUTPUT>
"""
    
    result = parser.parse(response)
    print(f"\n✓ Uppercase Tags: Valid={result.is_valid}")
    assert result.is_valid, "Uppercase tags should work"
    assert "break this down" in result.reasoning.lower()
    assert "156.78" in result.output
    print(f"  Reasoning: {result.reasoning[:50]}...")
    print(f"  Output: {result.output[:50]}...")


def test_colon_format():
    """Test colon-based format"""
    parser = CoTParser()
    
    response = """
REASONING: To solve this query, I need to examine the relationship between variables.
This requires statistical analysis and careful interpretation of the results.
I will use correlation analysis to determine the strength of the relationship.

OUTPUT: Correlation coefficient = 0.85 (strong positive correlation)
"""
    
    result = parser.parse(response)
    print(f"\n✓ Colon Format: Valid={result.is_valid}")
    assert result.is_valid, "Colon format should work"
    assert "relationship between variables" in result.reasoning
    assert "0.85" in result.output
    print(f"  Reasoning: {result.reasoning[:50]}...")
    print(f"  Output: {result.output[:50]}...")


def test_markdown_headers():
    """Test markdown header format"""
    parser = CoTParser()
    
    response = """
## REASONING

This is a complex problem that requires multiple analytical steps.
I will approach this by first understanding the data structure and then applying appropriate methods.
The analysis involves careful consideration of data quality and statistical significance.

## OUTPUT

The recommended action is to focus on customer segments with highest retention rates.
"""
    
    result = parser.parse(response)
    print(f"\n✓ Markdown Headers: Valid={result.is_valid}")
    assert result.is_valid, "Markdown headers should work"
    assert "complex problem" in result.reasoning
    assert "retention rates" in result.output
    print(f"  Reasoning: {result.reasoning[:50]}...")
    print(f"  Output: {result.output[:50]}...")


def test_content_based_no_tags():
    """Test content-based parsing when no tags present"""
    parser = CoTParser()
    
    response = """
To answer this question, I need to analyze the sales data carefully.

First, I'll examine the distribution of sales across different regions.
This will help identify patterns and trends in the data.

Next, I'll calculate summary statistics including mean, median, and standard deviation.
These metrics will provide insight into the central tendency and variability.

Finally, based on this analysis, the total revenue for Q4 is $1,250,000.
"""
    
    result = parser.parse(response)
    print(f"\n✓ Content-Based (No Tags): Valid={result.is_valid}")
    assert result.is_valid, "Content-based parsing should work"
    assert len(result.reasoning) > 50, "Should extract reasoning"
    assert "$1,250,000" in result.output, "Should extract conclusion"
    print(f"  Reasoning: {result.reasoning[:50]}...")
    print(f"  Output: {result.output[:50]}...")


def test_fallback_unstructured():
    """Test fallback for completely unstructured response"""
    parser = CoTParser()
    
    response = "The answer is simply 42. That's all."
    
    result = parser.parse(response)
    print(f"\n✓ Fallback Unstructured: Valid={result.is_valid}")
    # Should return false but provide the raw response as output
    assert not result.is_valid, "Should fail validation for unstructured"
    assert "42" in result.output, "Should still return content"
    assert "Unable to extract" in result.reasoning
    print(f"  Reasoning: {result.reasoning[:50]}...")
    print(f"  Output: {result.output[:50]}...")


def test_short_reasoning():
    """Test handling of too-short reasoning"""
    parser = CoTParser()
    
    response = """
[REASONING]
Quick answer.
[/REASONING]

[OUTPUT]
42
[/OUTPUT]
"""
    
    result = parser.parse(response)
    print(f"\n✓ Short Reasoning: Valid={result.is_valid}")
    # Should fail due to too-short reasoning, but try fuzzy/content-based
    print(f"  Error: {result.error_message}")
    print(f"  Output still available: {len(result.output) > 0}")


def run_all_tests():
    """Run all test cases"""
    print("="*80)
    print("TESTING ENHANCED COT PARSER")
    print("="*80)
    
    tests = [
        ("Exact Match", test_exact_match),
        ("Angle Brackets", test_angle_brackets),
        ("Uppercase Tags", test_uppercase_tags),
        ("Colon Format", test_colon_format),
        ("Markdown Headers", test_markdown_headers),
        ("Content-Based", test_content_based_no_tags),
        ("Fallback", test_fallback_unstructured),
        ("Short Reasoning", test_short_reasoning),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} FAILED: {str(e)}")
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
