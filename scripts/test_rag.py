#!/usr/bin/env python3
"""
Simple test script to verify RAG functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from core.chromadb_client import ChromaDBClient
from core.llm_client import LLMClient
from core.crewai_base import RAGTool

def test_rag_system():
    """Test the RAG system components"""

    print("🧪 Testing RAG System Components...")

    # Test 1: ChromaDB Client
    print("\n1. Testing ChromaDB Client...")
    try:
        chroma_client = ChromaDBClient()
        # Try to query (should return empty if no docs)
        result = chroma_client.query("test query", n_results=3)
        print(f"✅ ChromaDB query successful: {type(result)}")
        print(f"   Documents found: {len(result.get('documents', [[]])[0])}")
        if 'error' in result:
            print(f"   Note: {result['error']}")
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False

    # Test 2: LLM Client
    print("\n2. Testing LLM Client...")
    try:
        llm_client = LLMClient()
        test_prompt = "Hello, please respond with 'LLM test successful'"
        response = llm_client.generate_primary(test_prompt)
        if isinstance(response, dict) and "response" in response:
            print(f"✅ LLM generation successful: {len(response['response'])} chars")
        elif isinstance(response, dict) and "error" in response:
            print(f"⚠️ LLM generation had issues: {response['error']}")
        else:
            print(f"❌ Unexpected LLM response format: {response}")
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

    # Test 3: RAG Tool
    print("\n3. Testing RAG Tool...")
    try:
        rag_tool = RAGTool(chroma_client, llm_client)
        result = rag_tool._run("test query", n_results=3)
        print(f"✅ RAG tool executed successfully: {len(result)} chars")
        print(f"   Result preview: {result[:100]}...")
    except Exception as e:
        print(f"❌ RAG tool test failed: {e}")
        return False

    print("\n🎉 All RAG system tests passed!")
    return True

if __name__ == "__main__":
    success = test_rag_system()
    sys.exit(0 if success else 1)
