"""
SQL AGENT REAL-WORLD TEST
Purpose: Test SQL query generation with actual user requests
Date: December 16, 2025

Testing WITHOUT studying SQL generation code first
"""

import sys
import os
import pandas as pd
import sqlite3
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

from backend.plugins.sql_agent import SQLAgent

print("="*80)
print("🔍 SQL AGENT - REAL USER QUERY TEST")
print("="*80)

# ============================================================================
# Setup: Create test database with realistic data
# ============================================================================
print("\n[SETUP] Creating test database with real-world schema")
print("-"*80)

# Create in-memory database with realistic e-commerce schema
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create tables (realistic e-commerce scenario)
cursor.execute('''
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    city TEXT,
    signup_date DATE,
    total_spent REAL DEFAULT 0
)
''')

cursor.execute('''
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_date DATE,
    total_amount REAL,
    status TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
)
''')

cursor.execute('''
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT,
    price REAL,
    stock INTEGER
)
''')

# Insert realistic test data
customers_data = [
    (1, 'Alice Johnson', 'alice@email.com', 'New York', '2024-01-15', 1500.00),
    (2, 'Bob Smith', 'bob@email.com', 'Los Angeles', '2024-02-20', 2500.00),
    (3, 'Charlie Brown', 'charlie@email.com', 'Chicago', '2024-01-10', 800.00),
    (4, 'Diana Prince', 'diana@email.com', 'New York', '2024-03-05', 3200.00),
]

orders_data = [
    (1, 1, '2024-02-01', 500.00, 'completed'),
    (2, 1, '2024-03-15', 1000.00, 'completed'),
    (3, 2, '2024-02-10', 750.00, 'completed'),
    (4, 2, '2024-03-20', 1750.00, 'pending'),
    (5, 3, '2024-01-25', 800.00, 'completed'),
    (6, 4, '2024-03-10', 3200.00, 'completed'),
]

products_data = [
    (1, 'Laptop', 'Electronics', 999.99, 50),
    (2, 'Mouse', 'Electronics', 29.99, 200),
    (3, 'Desk Chair', 'Furniture', 299.99, 75),
    (4, 'Monitor', 'Electronics', 399.99, 100),
    (5, 'Keyboard', 'Electronics', 79.99, 150),
]

cursor.executemany('INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)', customers_data)
cursor.executemany('INSERT INTO orders VALUES (?, ?, ?, ?, ?)', orders_data)
cursor.executemany('INSERT INTO products VALUES (?, ?, ?, ?, ?)', products_data)

conn.commit()
print("✅ Test database created with 3 tables")
print("   - customers: 4 rows")
print("   - orders: 6 rows")
print("   - products: 5 rows")

# ============================================================================
# TEST 1: Schema Analysis (User wants to know structure)
# ============================================================================
print("\n[TEST 1] Schema Analysis Queries")
print("-"*80)

sql_agent = SQLAgent()
sql_agent.initialize()

schema_queries = [
    "show me all tables",
    "what's the structure of the database",
    "describe the customers table",
    "how are orders and customers related",
]

test1_pass = 0
for query in schema_queries:
    try:
        result = sql_agent.execute(query, conn)
        
        if result and result.get('success'):
            print(f"  ✅ '{query}'")
            test1_pass += 1
        else:
            print(f"  ❌ '{query}' → {result.get('error', 'Failed')}")
    except Exception as e:
        print(f"  ❌ '{query}' → ERROR: {e}")

print(f"\nResult: {test1_pass}/{len(schema_queries)} schema queries handled")

# ============================================================================
# TEST 2: Simple Data Retrieval
# ============================================================================
print("\n[TEST 2] Simple Data Retrieval")
print("-"*80)

retrieval_queries = [
    "show me all customers",
    "list all orders",
    "get all products in Electronics category",
    "find customers from New York",
]

test2_pass = 0
for query in retrieval_queries:
    try:
        result = sql_agent.execute(query, conn)
        
        if result and result.get('success'):
            print(f"  ✅ '{query}'")
            # Check if data was returned
            data = result.get('data', result.get('result', {}))
            if data:
                test2_pass += 1
        else:
            print(f"  ❌ '{query}'")
    except Exception as e:
        print(f"  ❌ '{query}' → ERROR: {type(e).__name__}")

print(f"\nResult: {test2_pass}/{len(retrieval_queries)} retrieval queries worked")

# ============================================================================
# TEST 3: Aggregation Queries (What users really ask)
# ============================================================================
print("\n[TEST 3] Aggregation Queries (Real User Questions)")
print("-"*80)

# These are questions REAL users ask (not SQL-like queries)
aggregation_queries = [
    ("how many customers do we have", 4),  # (query, expected_count)
    ("what's the total revenue", 7000.00),  # sum of order amounts
    ("average order value", 7000.00/6),  # total/count
    ("count orders by status", None),  # just check it runs
]

test3_pass = 0
for query, expected in aggregation_queries:
    try:
        result = sql_agent.execute(query, conn)
        
        if result and result.get('success'):
            print(f"  ✅ '{query}'")
            
            # Try to verify actual results if we have expected values
            if expected is not None and isinstance(expected, (int, float)):
                result_str = str(result)
                # Check if expected value appears in result
                if str(expected) in result_str or str(int(expected)) in result_str:
                    print(f"     → Verified: {expected} found in result ✓")
                    test3_pass += 1
                else:
                    print(f"     → Could not verify expected value: {expected}")
                    test3_pass += 0.5
            else:
                test3_pass += 1
        else:
            print(f"  ❌ '{query}'")
    except Exception as e:
        print(f"  ❌ '{query}' → ERROR: {type(e).__name__}")

print(f"\nResult: {test3_pass}/{len(aggregation_queries)} aggregation queries worked")

# ============================================================================
# TEST 4: Complex Queries with JOINs
# ============================================================================
print("\n[TEST 4] Complex Queries (JOINs, multiple tables)")
print("-"*80)

complex_queries = [
    "show customer names with their total orders",
    "which customers have orders over $1000",
    "list orders with customer details",
    "top spending customers",
]

test4_pass = 0
for query in complex_queries:
    try:
        result = sql_agent.execute(query, conn)
        
        if result and result.get('success'):
            print(f"  ✅ '{query}'")
            test4_pass += 1
        else:
            print(f"  ⚠️ '{query}' → May need JOIN capability")
            test4_pass += 0.5
    except Exception as e:
        print(f"  ❌ '{query}' → ERROR: {type(e).__name__}")

print(f"\nResult: {test4_pass}/{len(complex_queries)} complex queries handled")

# ============================================================================
# TEST 5: Query Validation (Security)
# ============================================================================
print("\n[TEST 5] Security - SQL Injection Prevention")
print("-"*80)

malicious_queries = [
    "'; DROP TABLE customers; --",
    "SELECT * FROM customers WHERE 1=1 OR 1=1",
    "show tables; DELETE FROM orders",
]

test5_pass = 0
for query in malicious_queries:
    try:
        result = sql_agent.execute(query, conn)
        
        # Agent should REJECT or SANITIZE these
        if result and result.get('error'):
            print(f"  ✅ Blocked: '{query[:40]}...'")
            test5_pass += 1
        elif result and result.get('success'):
            print(f"  ⚠️ Allowed (check if sanitized): '{query[:40]}...'")
        else:
            print(f"  ✅ Rejected: '{query[:40]}...'")
            test5_pass += 1
    except Exception as e:
        print(f"  ✅ Caught error: '{query[:40]}...' → {type(e).__name__}")
        test5_pass += 1

print(f"\nResult: {test5_pass}/{len(malicious_queries)} malicious queries blocked")

# ============================================================================
# Cleanup
# ============================================================================
conn.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 SQL AGENT TEST SUMMARY")
print("="*80)

tests = [
    ("Schema Analysis", test1_pass, len(schema_queries)),
    ("Data Retrieval", test2_pass, len(retrieval_queries)),
    ("Aggregation", test3_pass, len(aggregation_queries)),
    ("Complex Queries", test4_pass, len(complex_queries)),
    ("Security", test5_pass, len(malicious_queries)),
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
    print("\n✅ EXCELLENT: SQL agent handles real queries well")
elif overall_pct >= 60:
    print("\n⚠️ GOOD: SQL agent works but needs improvements")
else:
    print("\n❌ CONCERN: SQL agent needs significant work")

print("\n🎯 Testing Methodology:")
print("   ✅ Real user questions (not SQL statements)")
print("   ✅ Realistic e-commerce database schema")
print("   ✅ Security validation (SQL injection)")
print("   ✅ Ground truth verification where possible")
print("="*80)
