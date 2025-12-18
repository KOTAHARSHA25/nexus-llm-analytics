"""
PHASE 1: Chain-of-Thought (CoT) System Testing
Tests self-correction, reasoning extraction, and quality improvement
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backend.core.cot_parser import CoTParser, CriticParser

print("=" * 80)
print("PHASE 1: CHAIN-OF-THOUGHT (CoT) SYSTEM TESTING")
print("=" * 80)

# Test counters
total = 0
passed = 0

# TEST 1: CoT Parser - Valid Response
print("\n[TEST 1] CoT Parser - Valid Response Extraction")
total += 1
parser = CoTParser()
valid_response = """
[REASONING]
Step 1: Understanding the data
First, I need to analyze the sales data structure. The dataset contains revenue, quantity, and date columns.

Step 2: Calculate total revenue
I will sum all revenue values: $1000 + $2000 + $3000 = $6000

Step 3: Verify the calculation
The total makes sense given the data range and no missing values.
[/REASONING]

[OUTPUT]
The total revenue is $6,000 based on summing all sales transactions.
[/OUTPUT]
"""

result = parser.parse(valid_response)
if result.is_valid and "Step 1" in result.reasoning and "$6,000" in result.output:
    print("PASS: Valid CoT response parsed correctly")
    print(f"  - Reasoning length: {len(result.reasoning)} chars")
    print(f"  - Output length: {len(result.output)} chars")
    passed += 1
else:
    print(f"FAIL: Parser returned is_valid={result.is_valid}, error={result.error_message}")

# TEST 2: CoT Parser - Missing Reasoning Section
print("\n[TEST 2] CoT Parser - Missing REASONING Section Detection")
total += 1
invalid_response = "[OUTPUT]Just an answer[/OUTPUT]"
result = parser.parse(invalid_response)
if not result.is_valid and "Missing [REASONING]" in result.error_message:
    print("PASS: Correctly detected missing REASONING section")
    passed += 1
else:
    print("FAIL: Did not detect missing REASONING section")

# TEST 3: CoT Parser - Missing Output Section
print("\n[TEST 3] CoT Parser - Missing OUTPUT Section Detection")
total += 1
invalid_response = "[REASONING]Some reasoning[/REASONING]"
result = parser.parse(invalid_response)
if not result.is_valid and "Missing [OUTPUT]" in result.error_message:
    print("PASS: Correctly detected missing OUTPUT section")
    passed += 1
else:
    print("FAIL: Did not detect missing OUTPUT section")

# TEST 4: CoT Parser - Too Short Reasoning
print("\n[TEST 4] CoT Parser - Short Reasoning Detection")
total += 1
short_response = """
[REASONING]
Too short
[/REASONING]
[OUTPUT]
Answer
[/OUTPUT]
"""
result = parser.parse(short_response)
if not result.is_valid and "too short" in result.error_message.lower():
    print("PASS: Correctly detected reasoning too short (min 50 chars)")
    passed += 1
else:
    print("FAIL: Did not detect short reasoning")

# TEST 5: Step Extraction
print("\n[TEST 5] CoT Parser - Step Extraction")
total += 1
reasoning_with_steps = """
Step 1: First analysis
Analyzing the data structure

Step 2: Calculation phase
Computing the metrics

Step 3: Validation
Checking results
"""
steps = parser.extract_steps(reasoning_with_steps)
if len(steps) == 3 and "Step 1" in steps[0] and "Step 3" in steps[2]:
    print(f"PASS: Extracted {len(steps)} steps correctly")
    passed += 1
else:
    print(f"FAIL: Expected 3 steps, got {len(steps)}")

# TEST 6: Critic Parser - Valid Response
print("\n[TEST 6] Critic Parser - Marks Response as Valid")
total += 1
critic = CriticParser()
valid_critic = "[VALID]"
feedback = critic.parse(valid_critic)
if feedback.is_valid and len(feedback.issues) == 0:
    print("PASS: Critic correctly marked response as valid")
    passed += 1
else:
    print("FAIL: Critic did not recognize [VALID] marker")

# TEST 7: Critic Parser - Issues Detection
print("\n[TEST 7] Critic Parser - Issues Extraction")
total += 1
critic_with_issues = """
[ISSUES]
Issue 1: Missing confidence interval
   Location: Step 2
   Severity: MEDIUM
   Suggestion: Add ±0.5 confidence range

Issue 2: No sample size mentioned
   Location: Step 3
   Severity: LOW
   Suggestion: Include n=100
[/ISSUES]
"""
feedback = critic.parse(critic_with_issues)
if not feedback.is_valid and len(feedback.issues) == 2:
    print(f"PASS: Critic extracted {len(feedback.issues)} issues")
    print(f"  - Issue 1: {feedback.issues[0].description[:50]}...")
    print(f"  - Severity: {feedback.issues[0].severity}")
    passed += 1
else:
    print(f"FAIL: Expected 2 issues, got {len(feedback.issues)}")

# TEST 8: Case Insensitivity
print("\n[TEST 8] CoT Parser - Case Insensitive Tags")
total += 1
lowercase_response = """
[reasoning]
Step 1: Analysis performed
[/reasoning]
[output]
Result: Success
[/output]
"""
result = parser.parse(lowercase_response)
if result.is_valid:
    print("PASS: Parser handles lowercase tags correctly")
    passed += 1
else:
    print("FAIL: Parser did not handle lowercase tags")

# TEST 9: Real-World Example
print("\n[TEST 9] CoT Parser - Real-World Complex Example")
total += 1
complex_response = """
Based on the data analysis:

[REASONING]
Step 1: Data Exploration and Understanding
I examined the customer_data.csv file which contains 100 rows with columns: customer_id, name, age, city, membership_level, and revenue.

Step 2: Calculating Average Age
To find the average age, I need to:
- Extract all age values from the age column
- Sum them up: total = 4,251
- Divide by count: 4,251 / 100 = 42.51 years

Step 3: Verification and Validation
I checked for:
- Missing values: None found
- Outliers: Ages range from 25-60, all reasonable
- Data quality: All numeric values are valid

Step 4: Statistical Significance
With n=100, this sample size is sufficient for a reliable average.
The standard deviation is approximately 10.2 years.
[/REASONING]

[OUTPUT]
The average age of customers is 42.51 years (n=100, SD=10.2).
This represents a middle-aged customer base with good statistical confidence.
[/OUTPUT]
"""
result = parser.parse(complex_response)
if result.is_valid and len(result.reasoning) > 200:
    print("PASS: Complex real-world response parsed successfully")
    print(f"  - Reasoning: {len(result.reasoning)} chars, {len(parser.extract_steps(result.reasoning))} steps")
    print(f"  - Output: {len(result.output)} chars")
    passed += 1
else:
    print(f"FAIL: Complex parsing failed - is_valid={result.is_valid}")

# Summary
print("\n" + "=" * 80)
print("PHASE 1 RESULTS: Chain-of-Thought System")
print("=" * 80)
print(f"Total Tests: {total}")
print(f"Passed: {passed}")
print(f"Failed: {total - passed}")
print(f"Success Rate: {passed/total*100:.1f}%")

if passed == total:
    print("\n✅ ALL CoT SYSTEM TESTS PASSED")
else:
    print(f"\n⚠️ {total - passed} TESTS FAILED - Review above for details")
