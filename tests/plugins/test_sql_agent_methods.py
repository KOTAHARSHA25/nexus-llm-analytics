"""
SQL AGENT - AGENT METHOD TESTS
Testing all 5 implemented SQL analysis methods
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

print("="*80)
print("SQL AGENT - AGENT METHOD TESTS")
print("="*80)
print("Testing SQL query generation, execution, and optimization methods\n")

try:
    from backend.plugins.sql_agent import SQLAgent
    print("✅ SQLAgent imported successfully\n")
except ImportError as e:
    print(f"❌ FAILED to import SQLAgent: {e}")
    sys.exit(1)

# Initialize agent
agent = SQLAgent()
config = {}
agent.config = config
if not agent.initialize():
    print("❌ Agent initialization failed")
    sys.exit(1)

print("✅ Agent initialized successfully\n")

# TEST 1: Schema Analysis
print("="*80)
print("TEST 1: Schema Analysis (_analyze_schema)")
print("="*80)

try:
    result = agent._analyze_schema()
    
    assert result['success'], f"Schema analysis should succeed, got error: {result.get('error', 'unknown')}"
    assert 'result' in result, "Should have result key"
    assert 'schema_analysis' in result['result'], "Should have schema_analysis"
    
    schema = result['result']['schema_analysis']
    print(f"✅ Schema analysis completed")
    print(f"✅ Tables found: {len(schema['tables'])}")
    print(f"✅ Relationships: {len(schema['relationships'])}")
    
    # Validate schema structure
    assert 'tables' in schema, "Should have tables list"
    assert 'relationships' in schema, "Should have relationships list"
    assert len(schema['tables']) > 0, "Should find at least one table"
    
    # Check table details
    for table in schema['tables']:
        print(f"  - {table['name']}: {len(table['columns'])} columns, {table['row_count']} rows")
        assert 'name' in table, "Table should have name"
        assert 'columns' in table, "Table should have columns"
        assert 'row_count' in table, "Table should have row count"
    
    # Check recommendations
    if 'recommendations' in result['result']:
        print(f"✅ Recommendations: {len(result['result']['recommendations'])} suggestions")
    
    print("\n✅ TEST 1 PASSED - Schema analysis working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 1 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 2: SQL Query Generation - Count Users
print("="*80)
print("TEST 2: SQL Query Generation (_generate_sql_query) - Count Users")
print("="*80)

try:
    result = agent._generate_sql_query("count users in database")
    
    assert result['success'], f"Query generation should succeed, got error: {result.get('error', 'unknown')}"
    assert 'result' in result, "Should have result key"
    assert 'generated_sql' in result['result'], "Should have generated SQL"
    
    sql = result['result']['generated_sql']
    print(f"✅ Query generation completed")
    print(f"✅ Generated SQL: {sql}")
    print(f"✅ Explanation: {result['result']['explanation']}")
    
    # Validate SQL generation
    assert "COUNT" in sql.upper(), "Should generate COUNT query"
    assert "users" in sql.lower(), "Should reference users table"
    assert 'explanation' in result['result'], "Should have explanation"
    
    print("\n✅ TEST 2 PASSED - Query generation (count) working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 2 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 3: SQL Query Generation - Average Orders
print("="*80)
print("TEST 3: SQL Query Generation (_generate_sql_query) - Average Orders")
print("="*80)

try:
    result = agent._generate_sql_query("what is the average orders amount")
    
    assert result['success'], "Query generation should succeed"
    
    sql = result['result']['generated_sql']
    print(f"✅ Query generation completed")
    print(f"✅ Generated SQL: {sql}")
    print(f"✅ Explanation: {result['result']['explanation']}")
    
    # Validate SQL generation
    assert "AVG" in sql.upper(), "Should generate AVG query"
    assert "orders" in sql.lower(), "Should reference orders table"
    
    # Check estimated complexity
    if 'estimated_complexity' in result['result']:
        print(f"✅ Estimated complexity: {result['result']['estimated_complexity']}")
    
    print("\n✅ TEST 3 PASSED - Query generation (average) working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 3 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 4: SQL Query Execution
print("="*80)
print("TEST 4: SQL Query Execution (_execute_sql_query)")
print("="*80)

try:
    # Test with a simple SELECT query
    sql_query = "SELECT * FROM users LIMIT 3"
    result = agent._execute_sql_query(sql_query)
    
    assert result['success'], f"Query execution should succeed, got error: {result.get('error', 'unknown')}"
    assert 'result' in result, "Should have result key"
    assert 'results' in result['result'], "Should have results"
    
    print(f"✅ Query execution completed")
    print(f"✅ SQL Query: {result['result']['sql_query']}")
    print(f"✅ Rows returned: {result['result']['row_count']}")
    print(f"✅ Execution time: {result['result']['execution_time_ms']}ms")
    
    # Validate results
    results = result['result']['results']
    assert isinstance(results, list), "Results should be a list"
    assert len(results) > 0, "Should return at least one row"
    
    # Check columns
    if 'columns' in result['result']:
        print(f"✅ Columns: {', '.join(result['result']['columns'])}")
    
    # Display sample data
    print(f"✅ Sample data:")
    for i, row in enumerate(results[:2], 1):
        print(f"  Row {i}: {row}")
    
    print("\n✅ TEST 4 PASSED - Query execution working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 4 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 5: SQL Query Optimization
print("="*80)
print("TEST 5: SQL Query Optimization (_optimize_query)")
print("="*80)

try:
    # Test optimization suggestions for a query
    test_query = "SELECT * FROM orders WHERE user_id = 123"
    result = agent._optimize_query(test_query)
    
    assert result['success'], "Query optimization should succeed"
    assert 'result' in result, "Should have result key"
    assert 'optimization_suggestions' in result['result'], "Should have optimization suggestions"
    
    print(f"✅ Query optimization completed")
    print(f"✅ Original query: {result['result']['original_query']}")
    
    suggestions = result['result']['optimization_suggestions']
    print(f"✅ Optimization suggestions ({len(suggestions)}):")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    # Validate suggestions
    assert len(suggestions) > 0, "Should provide at least one suggestion"
    
    # Check estimated improvement
    if 'estimated_improvement' in result['result']:
        print(f"✅ Estimated improvement: {result['result']['estimated_improvement']}")
    
    print("\n✅ TEST 5 PASSED - Query optimization working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 5 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 6: General Analysis
print("="*80)
print("TEST 6: General Analysis (_general_analysis)")
print("="*80)

try:
    result = agent._general_analysis("What can you do with SQL?")
    
    assert result['success'], "General analysis should succeed"
    assert 'result' in result, "Should have result key"
    
    print(f"✅ General analysis completed")
    print(f"✅ Query: {result['result']['query']}")
    print(f"✅ Analysis type: {result['result']['analysis_type']}")
    
    # Check capabilities
    if 'capabilities' in result['result']:
        capabilities = result['result']['capabilities']
        print(f"✅ Capabilities ({len(capabilities)}):")
        for cap in capabilities:
            print(f"  - {cap}")
    
    # Check next steps
    if 'next_steps' in result['result']:
        next_steps = result['result']['next_steps']
        print(f"✅ Next steps ({len(next_steps)}):")
        for step in next_steps:
            print(f"  - {step}")
    
    print("\n✅ TEST 6 PASSED - General analysis working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 6 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 7: Query Generation - Top Products (Complex Query)
print("="*80)
print("TEST 7: SQL Query Generation - Top Products (JOIN Query)")
print("="*80)

try:
    result = agent._generate_sql_query("show me the top 10 products by order count")
    
    assert result['success'], "Query generation should succeed"
    
    sql = result['result']['generated_sql']
    print(f"✅ Query generation completed")
    print(f"✅ Generated SQL:")
    print(sql)
    print(f"✅ Explanation: {result['result']['explanation']}")
    
    # Validate complex query generation
    sql_upper = sql.upper()
    assert "JOIN" in sql_upper, "Should generate JOIN query"
    assert "GROUP BY" in sql_upper, "Should use GROUP BY"
    assert "ORDER BY" in sql_upper, "Should use ORDER BY"
    assert "LIMIT" in sql_upper, "Should use LIMIT"
    
    print("\n✅ TEST 7 PASSED - Complex query generation working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 7 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# TEST 8: Query Execution - Edge Case (Invalid SQL)
print("="*80)
print("TEST 8: Query Execution - Edge Case (No Valid SQL)")
print("="*80)

try:
    # Test with non-SQL text
    result = agent._execute_sql_query("This is not a SQL query")
    
    # Should fail gracefully
    assert not result['success'], "Should fail with invalid SQL"
    assert 'error' in result, "Should have error message"
    print(f"✅ Error message: {result['error']}")
    print(f"✅ Gracefully handled invalid SQL")
    
    print("\n✅ TEST 8 PASSED - Edge case handling working correctly\n")
    
except Exception as e:
    print(f"❌ TEST 8 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Summary
print("="*80)
print("SQL AGENT METHOD TESTS SUMMARY")
print("="*80)
print("\n✅ All 8/8 tests PASSED!\n")
print("Methods tested:")
print("  ✅ _analyze_schema - Database schema analysis with recommendations")
print("  ✅ _generate_sql_query - Natural language to SQL (count, average, joins)")
print("  ✅ _execute_sql_query - SQL execution with demo results")
print("  ✅ _optimize_query - Query optimization suggestions")
print("  ✅ _general_analysis - General SQL capabilities and guidance")
print("  ✅ Edge case: Complex JOIN query generation")
print("  ✅ Edge case: Invalid SQL handling")
print("\n" + "="*80)
print("✅ SQL AGENT: 5/5 IMPLEMENTED METHODS TESTED (100%)")
print("="*80)
print("\n🎉 All SQL agent methods are NOW TESTED!")
print("📝 Note: Tests use demo/mock data (no actual database required)")
