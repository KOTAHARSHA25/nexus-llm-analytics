import pytest
from src.backend.core.automated_validation import AutomatedValidator, AutomatedValidationResult, ValidationIssue

@pytest.fixture
def validator():
    return AutomatedValidator()

def test_calculation_errors(validator):
    # Test 2+2=5 pattern
    issues = validator._check_calculations("Reasoning: 2 + 2 = 5", "Output")
    assert len(issues) > 0
    assert issues[0].issue_type == "calculation_error"

    # Test explicit addition sequence
    issues = validator._check_calculations("10 + 20 = 40", "Output")
    assert len(issues) > 0
    assert "10+20=30, not 40" in issues[0].description

def test_logic_errors(validator):
    # Test inverted logic
    issues = validator._check_logic_errors("Show users age greater than 30", "Filter age < 30")
    assert len(issues) > 0
    assert issues[0].issue_type == "inverted_logic"

def test_reasoning_quality(validator):
    # Test short reasoning
    issues = validator._check_reasoning_quality("Too short")
    assert len(issues) > 0
    assert issues[0].issue_type == "insufficient_reasoning"
    
    # Test no steps
    issues = validator._check_reasoning_quality("This is a long sentence but it has no periods or steps in it at all")
    assert len(issues) > 0
    assert issues[0].issue_type == "lacks_steps"

def test_query_alignment(validator):
    issues = validator._check_query_alignment("Show me total sales", "Here is the average price")
    # 'total' in query, not in output 'average'
    # Actually logic checks if 'total' missing from output
    assert len(issues) > 0
    types = [i.issue_type for i in issues]
    assert "missing_calculation" in types # expecting total
    assert "misaligned_output" in types # no keywords overlap likely

def test_data_consistency(validator):
    context = {"columns": ["id", "name", "revenue"]}
    # The regex looks for [A-Z][a-z_]+ so we need capitalized column name
    issues = validator._check_data_consistency("Filter by Age column", "", context)
    assert len(issues) > 0
    assert issues[0].issue_type == "nonexistent_column" 
    assert "Age" in issues[0].description

def test_causation_errors(validator):
    issues = validator._check_causation_errors("Does ad spend cause sales?", "High correlation proves ad spend causes sales", "")
    assert len(issues) > 0
    assert "causation" in issues[0].issue_type

def test_formula_errors(validator):
    issues = validator._check_formula_errors("profit margin = revenue / cost", "")
    assert len(issues) > 0
    assert issues[0].issue_type == "formula_error"

def test_time_period_errors(validator):
    issues = validator._check_time_period_errors("Q4 sales", "Looking at September data")
    assert len(issues) > 0
    assert issues[0].issue_type == "time_period_error"

def test_percentage_format(validator):
    issues = validator._check_percentage_format("Show percentage growth", "", "Growth was 0.75")
    assert len(issues) > 0
    assert issues[0].issue_type == "percentage_format_error"

def test_full_validate(validator):
    result = validator.validate(
        query="Show me total revenue",
        reasoning="Okay. Calculation: 10 + 10 = 25.", # Calc error
        output="The result is 25.",
        data_context={"columns": ["revenue"]}
    )
    assert result.is_valid is False
    assert len(result.issues) > 0
    assert "Calculation error" in result.to_feedback_text()
