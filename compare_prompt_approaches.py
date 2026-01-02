"""
Compare Template-Based vs Direct LLM Approach
This test evaluates which approach produces better code generation results.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from backend.core.code_generator import CodeGenerator
from backend.core.llm_client import LLMClient

def print_header(title):
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")

def test_both_approaches():
    """
    Compare:
    1. CURRENT: Template-based prompts with structure
    2. PROPOSED: Direct query + data to LLM with minimal guidance
    """
    
    print_header("📊 COMPARING PROMPT APPROACHES")
    
    # Test data
    df = pd.DataFrame({
        'product_name': ['Widget A', 'Widget B', 'Widget C', 'Widget D', 'Widget E'],
        'product_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'sales': [1500, 2300, 1800, 950, 2100],
        'revenue': [15000, 23000, 18000, 9500, 21000],
        'category': ['Electronics', 'Furniture', 'Electronics', 'Clothing', 'Furniture']
    })
    
    test_queries = [
        "What is the highest sales value?",
        "Which product has the highest revenue?",
        "What are the top 3 products by sales?",
        "What is the average revenue?",
        "Show me products in Electronics category"
    ]
    
    try:
        llm_client = LLMClient()
        generator = CodeGenerator(llm_client)
    except Exception as e:
        print(f"❌ Could not initialize: {e}")
        return
    
    # Test with phi3:mini (small model)
    model = "phi3:mini"
    
    print(f"🤖 Testing with model: {model}")
    print(f"📦 Data shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}: {query}")
        print(f"{'─'*80}")
        
        # APPROACH 1: Current template-based
        print("\n1️⃣  TEMPLATE-BASED APPROACH (Current Fix 5):")
        try:
            prompt_template = generator._build_dynamic_prompt(query, df, model)
            print(f"   Prompt length: {len(prompt_template)} chars")
            print(f"   Prompt structure: {'Simple' if len(prompt_template) < 1000 else 'Detailed'}")
            
            result_template = generator.generate_code(query, df, model)
            if result_template.is_valid:
                print(f"   ✅ Generated valid code")
                print(f"   Code preview: {result_template.code[:150]}...")
            else:
                print(f"   ❌ Failed: {result_template.error_message}")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        # APPROACH 2: Direct to LLM (minimal structure)
        print("\n2️⃣  DIRECT LLM APPROACH (Your Proposal):")
        try:
            # Minimal prompt - just query + data info
            columns_list = ", ".join(df.columns.tolist())
            direct_prompt = f"""Generate Python code to answer this question:
"{query}"

The DataFrame 'df' has these columns: {columns_list}
Store the answer in a variable called 'result'.

Code:"""
            
            print(f"   Prompt length: {len(direct_prompt)} chars")
            print(f"   Prompt structure: Minimal/Direct")
            
            # Call LLM directly
            response = llm_client.generate(direct_prompt, model=model)
            response_text = response.get('response', '') if isinstance(response, dict) else str(response)
            
            # Extract code
            code = generator._extract_code(response_text)
            if code:
                is_valid, error = generator._validate_code_syntax(code)
                if is_valid:
                    print(f"   ✅ Generated valid code")
                    print(f"   Code preview: {code[:150]}...")
                else:
                    print(f"   ❌ Syntax error: {error}")
            else:
                print(f"   ❌ No code block found in response")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    # Now provide analysis
    print_header("📈 ANALYSIS: Which Approach is Better?")
    
    print("""
🔍 TEMPLATE-BASED APPROACH (Current Fix 5):
✅ Pros:
   • Provides clear patterns and examples
   • Consistent output format
   • Handles edge cases better (ID vs NAME columns)
   • Model-size aware (simple vs detailed)
   • Better guardrails (reminds about column validation)
   • Less likely to generate irrelevant code
   • Works better with small models (phi3:mini, tinyllama)
   
❌ Cons:
   • More code to maintain (template files)
   • Slightly longer prompts (~700-1400 chars)
   • Fixed structure may limit creativity
   
🔍 DIRECT LLM APPROACH (Your Proposal):
✅ Pros:
   • Simpler implementation (no templates)
   • Less code to maintain
   • More flexible/open-ended
   • Very short prompts (~100-200 chars)
   
❌ Cons:
   • Less consistent results (especially with small models)
   • May generate wrong patterns (e.g., returning full df instead of just name)
   • No guardrails for common mistakes
   • Small models struggle without structure
   • More likely to hallucinate column names
   • May not follow 'result' variable convention
   
💡 RECOMMENDATION:
""")
    
    # Test which actually works better
    print("\nLet me test one query with both approaches and compare results...\n")
    
    query = "What are the top 3 products by sales?"
    
    print(f"Query: {query}\n")
    
    # Template approach
    print("1️⃣  Template approach:")
    try:
        result1 = generator.generate_and_execute(query, df, model, max_retries=0)
        if result1.success:
            print(f"   ✅ SUCCESS")
            print(f"   Result: {result1.result}")
            print(f"   Code: {result1.cleaned_code}")
        else:
            print(f"   ❌ FAILED: {result1.error_message}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Direct approach
    print("\n2️⃣  Direct approach:")
    try:
        columns_list = ", ".join(df.columns.tolist())
        direct_prompt = f"""Generate Python code to answer: "{query}"
DataFrame 'df' columns: {columns_list}
Store answer in 'result'.
Code:"""
        
        response = llm_client.generate(direct_prompt, model=model)
        response_text = response.get('response', '') if isinstance(response, dict) else str(response)
        code = generator._extract_code(response_text)
        
        if code:
            exec_result = generator.execute_code(code, df)
            if exec_result.success:
                print(f"   ✅ SUCCESS")
                print(f"   Result: {exec_result.result}")
                print(f"   Code: {code}")
            else:
                print(f"   ❌ FAILED: {exec_result.error_message}")
        else:
            print(f"   ❌ No code generated")
    except Exception as e:
        print(f"   ❌ Exception: {e}")

if __name__ == "__main__":
    test_both_approaches()
