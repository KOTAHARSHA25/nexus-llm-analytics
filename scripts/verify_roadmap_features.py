
import sys
import os
import asyncio
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from backend.core.database import get_db_manager
from backend.io.code_generator import CodeGenerator

def test_history_db():
    print("\n[1] Testing SQLite History DB...")
    db = get_db_manager()
    
    # 1. Clear
    print("  - Clearing history...")
    db.clear_history()
    
    # 2. Add
    print("  - Adding query...")
    success = db.add_query(
        query="What is the sales trend?", 
        results_summary="Sales are up 10%", 
        files_used=["sales.csv"],
        timestamp="2023-01-01T12:00:00Z"
    )
    if not success:
        print("  ❌ Failed to add query")
        return False
        
    # 3. Get
    print("  - Fetching queries...")
    queries = db.get_recent_queries()
    if len(queries) != 1:
        print(f"  ❌ Expected 1 query, got {len(queries)}")
        return False
        
    if queries[0]['query'] != "What is the sales trend?":
        print(f"  ❌ Content mismatch: {queries[0]['query']}")
        return False
        
    print("  ✅ SQLite DB operations successful")
    return True

def test_codogen_prompt():
    print("\n[2] Testing CodeGenerator Prompt Construction...")
    try:
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        generator = CodeGenerator()
        prompt = generator._build_dynamic_prompt("Plot a chart of A vs B", df)
        
        if "DO NOT generate code to plot charts" not in prompt:
            print("  ❌ Prompt missing anti-plotting constraint")
            return False
            
        print("  ✅ Prompt contains anti-plotting constraint")
        return True
    except Exception as e:
        print(f"  ❌ Prompt construction failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Phase 14 Verification ===")
    history_ok = test_history_db()
    prompt_ok = test_codogen_prompt()
    
    if history_ok and prompt_ok:
        print("\n✅ All features verified successfully!")
        sys.exit(0)
    else:
        print("\n❌ Verification failed")
        sys.exit(1)
