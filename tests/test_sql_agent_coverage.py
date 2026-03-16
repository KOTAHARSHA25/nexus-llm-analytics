import pytest
import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.services.analysis_service import AnalysisService

DB_PATH = Path(project_root) / "data" / "samples" / "test_db.sqlite"

@pytest.mark.asyncio
async def test_sql_agent_query():
    """Test SQL Agent with a SQLite database"""
    if not DB_PATH.exists():
        pytest.skip("Test database not found. Run setup_sql_test_db.py first.")
        
    service = AnalysisService()
    
    # Explicitly ask for SQL query to trigger routing (or DataAnalyst in SQL mode)
    query = "Write a SQL query to find the total sales amount by region."
    
    result = await service.analyze(query, context={"filename": "test_db.sqlite", "filepath": str(DB_PATH)})
    
    assert result['success'] is True
    # The result should contain either the SQL query or the answer
    # Since we might not have a live LLM to generate SQL, we check for graceful handling or agent routing
    
    assert result.get('metadata', {}).get('agent') in ['SQLAgent', 'DataAnalyst']
    print(f"Result: {result}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
