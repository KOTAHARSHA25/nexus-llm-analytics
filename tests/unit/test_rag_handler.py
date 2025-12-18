"""
RAG HANDLER TEST
Purpose: Test document retrieval and RAG capabilities
Date: December 16, 2025
"""

import sys
import os
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

print("="*80)
print("🔍 RAG HANDLER TEST")
print("="*80)

# ============================================================================
# TEST 1: RAG Handler Initialization
# ============================================================================
print("\n[TEST 1] RAG Handler Initialization")
print("-"*80)

try:
    from backend.agents.rag_handler import RAGHandler
    
    rag = RAGHandler()
    print("  ✅ RAGHandler initialized")
    test1_pass = 1
except Exception as e:
    print(f"  ❌ Initialization failed: {e}")
    test1_pass = 0
    rag = None

# ============================================================================
# TEST 2: Document Indexing
# ============================================================================
print("\n[TEST 2] Document Indexing")
print("-"*80)

if rag:
    test_docs = [
        "Sales increased by 25% in Q4 2024",
        "Customer retention rate is 85%",
        "New product launch scheduled for March 2025",
    ]
    
    try:
        # Try to index documents
        result = rag.index_documents(test_docs)
        
        if result:
            print("  ✅ Documents indexed")
            test2_pass = 1
        else:
            print("  ⚠️ Indexing returned None")
            test2_pass = 0.5
    except Exception as e:
        print(f"  ⚠️ Indexing error: {type(e).__name__}")
        test2_pass = 0
else:
    test2_pass = 0

# ============================================================================
# TEST 3: Document Retrieval
# ============================================================================
print("\n[TEST 3] Document Retrieval")
print("-"*80)

if rag:
    queries = [
        "sales growth",
        "customer retention",
        "product launch",
    ]
    
    test3_results = []
    for query in queries:
        try:
            results = rag.retrieve(query, top_k=2)
            
            if results and len(results) > 0:
                print(f"  ✅ '{query}': {len(results)} results")
                test3_results.append(1)
            else:
                print(f"  ⚠️ '{query}': no results")
                test3_results.append(0)
        except Exception as e:
            print(f"  ⚠️ '{query}': {type(e).__name__}")
            test3_results.append(0)
    
    test3_pass = sum(test3_results)
    test3_total = len(queries)
else:
    test3_pass = 0
    test3_total = 3

# ============================================================================
# TEST 4: Similarity Search
# ============================================================================
print("\n[TEST 4] Similarity Search")
print("-"*80)

if rag:
    try:
        # Search for similar content
        similar = rag.find_similar("revenue and sales", top_k=1)
        
        if similar:
            print(f"  ✅ Found similar documents")
            test4_pass = 1
        else:
            print("  ⚠️ No similar documents")
            test4_pass = 0.5
    except Exception as e:
        print(f"  ⚠️ Search error: {type(e).__name__}")
        test4_pass = 0
else:
    test4_pass = 0

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 RAG HANDLER TEST SUMMARY")
print("="*80)

tests = [
    ("Initialization", test1_pass, 1),
    ("Document Indexing", test2_pass, 1),
    ("Document Retrieval", test3_pass, test3_total),
    ("Similarity Search", test4_pass, 1),
]

total_pass = sum(p for _, p, _ in tests)
total_count = sum(t for _, _, t in tests)

for test_name, passed, total in tests:
    pct = (passed/total*100) if total > 0 else 0
    status = "✅" if pct >= 75 else "⚠️" if pct >= 50 else "❌"
    print(f"{status} {test_name}: {passed}/{total} ({pct:.0f}%)")

print("-"*80)
overall_pct = (total_pass/total_count*100) if total_count > 0 else 0
print(f"Overall: {total_pass:.1f}/{total_count} ({overall_pct:.1f}%)")

if overall_pct >= 80:
    print("\n✅ EXCELLENT: RAG handler working well")
elif overall_pct >= 60:
    print("\n⚠️ GOOD: RAG handler functional")
else:
    print("\n❌ CONCERN: RAG handler needs work")

print("="*80)
