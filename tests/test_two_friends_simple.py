"""
Two Friends Model - Simple Diagnostic
Shows exactly what happens in the Generator-Critic loop
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.llm_client import LLMClient
from src.backend.core.cot_parser import CoTParser, CriticParser

def test_simple_loop():
    """Test a simple Generator → Critic interaction"""
    
    llm = LLMClient()
    cot_parser = CoTParser()
    critic_parser = CriticParser()
    
    # Simple query that might have issues
    query = "Calculate 15% of 200"
    
    print("="*80)
    print("SIMPLE TWO FRIENDS TEST")
    print("="*80)
    print(f"\nQuery: {query}\n")
    
    # Step 1: Generator produces reasoning + output
    print("STEP 1: Generator (Primary LLM)")
    print("-" * 80)
    
    generator_prompt = f"""You are a helpful assistant. Think step-by-step and structure your response.

Question: {query}

Provide your response in this format:
[REASONING]
Your step-by-step thinking process here
[/REASONING]

[OUTPUT]
Your final answer here
[/OUTPUT]
"""
    
    generator_response = llm.generate(generator_prompt, model='llama3.1:8b')
    print(f"Raw Response:\n{generator_response}\n")
    
    # Parse the response
    parsed = cot_parser.parse(generator_response)
    print(f"Parsed Successfully: {parsed.is_valid}")
    
    if parsed.is_valid:
        print(f"\nReasoning:\n{parsed.reasoning}\n")
        print(f"Output:\n{parsed.output}\n")
        
        # Step 2: Critic reviews the reasoning
        print("\nSTEP 2: Critic (Review LLM)")
        print("-" * 80)
        
        critic_prompt = f"""Review this reasoning for logical errors or mistakes.

Question: {query}

Reasoning provided:
{parsed.reasoning}

Answer provided:
{parsed.output}

If the reasoning is correct and answer is right, respond with: [VALID]

If there are issues, respond with:
[ISSUES]
Issue 1: Description of the problem
Location: Where in the reasoning
Severity: HIGH/MEDIUM/LOW
Suggestion: How to fix it
"""
        
        critic_response = llm.generate(critic_prompt, model='phi3:mini')
        print(f"Critic Response:\n{critic_response}\n")
        
        # Parse critic feedback
        feedback = critic_parser.parse(critic_response)
        print(f"Validation Result: {'VALID ✓' if feedback.is_valid else 'NEEDS REVISION ✗'}")
        
        if not feedback.is_valid:
            print(f"\nIssues Found: {len(feedback.issues)}")
            for i, issue in enumerate(feedback.issues, 1):
                print(f"\n  Issue {i}:")
                print(f"    Description: {issue.description}")
                print(f"    Severity: {issue.severity}")
                print(f"    Suggestion: {issue.suggestion}")
            
            print("\n" + "="*80)
            print("CONCLUSION: Critic caught issues - would trigger revision")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("CONCLUSION: Critic accepted output - no revision needed")
            print("="*80)
    else:
        print(f"❌ Parser failed: {parsed.error_message}")
        print("\nThis is why the parser fix was important!")
    
    return parsed.is_valid


if __name__ == "__main__":
    success = test_simple_loop()
    sys.exit(0 if success else 1)
