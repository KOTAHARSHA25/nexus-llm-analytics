"""
AGENT FACTORY TEST
Purpose: Test agent creation and initialization
Date: December 16, 2025
"""

import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

print("="*80)
print("🔍 AGENT FACTORY TEST")
print("="*80)

# ============================================================================
# TEST 1: Agent Factory Initialization
# ============================================================================
print("\n[TEST 1] Agent Factory Initialization")
print("-"*80)

try:
    from backend.agents.agent_factory import AgentFactory
    
    factory = AgentFactory()
    print("  ✅ AgentFactory initialized")
    test1_pass = 1
except Exception as e:
    print(f"  ❌ Initialization failed: {e}")
    test1_pass = 0
    factory = None

# ============================================================================
# TEST 2: Create Plugin Agents
# ============================================================================
print("\n[TEST 2] Create Plugin Agents")
print("-"*80)

if factory:
    agent_types = [
        "statistical",
        "financial", 
        "ml_insights",
        "time_series",
        "sql"
    ]
    
    test2_results = []
    for agent_type in agent_types:
        try:
            # Try to create agent
            agent = factory.create_agent(agent_type)
            
            if agent:
                print(f"  ✅ Created {agent_type} agent")
                test2_results.append(1)
            else:
                print(f"  ⚠️ {agent_type} agent returned None")
                test2_results.append(0.5)
        except Exception as e:
            print(f"  ❌ {agent_type} agent failed: {type(e).__name__}")
            test2_results.append(0)
    
    test2_pass = sum(test2_results)
    test2_total = len(agent_types)
else:
    test2_pass = 0
    test2_total = 5

# ============================================================================
# TEST 3: Agent Metadata
# ============================================================================
print("\n[TEST 3] Agent Metadata Retrieval")
print("-"*80)

if factory:
    try:
        # Get list of available agents
        available = factory.list_available_agents()
        
        if available and len(available) > 0:
            print(f"  ✅ Found {len(available)} available agents")
            test3_pass = 1
        else:
            print("  ⚠️ No agents listed")
            test3_pass = 0.5
    except Exception as e:
        print(f"  ❌ Listing failed: {type(e).__name__}")
        test3_pass = 0
else:
    test3_pass = 0

# ============================================================================
# TEST 4: Agent Capability Matching
# ============================================================================
print("\n[TEST 4] Agent Capability Matching")
print("-"*80)

if factory:
    test_queries = [
        ("calculate average sales", ["statistical", "financial"]),
        ("predict future trends", ["time_series", "ml_insights"]),
        ("run SQL query", ["sql"]),
    ]
    
    test4_results = []
    for query, expected_types in test_queries:
        try:
            # Try to find suitable agent
            suitable = factory.find_suitable_agent(query)
            
            if suitable:
                print(f"  ✅ '{query}' → {suitable}")
                test4_results.append(1)
            else:
                print(f"  ⚠️ '{query}' → No match")
                test4_results.append(0.5)
        except Exception as e:
            print(f"  ⚠️ '{query}' → {type(e).__name__}")
            test4_results.append(0)
    
    test4_pass = sum(test4_results)
    test4_total = len(test_queries)
else:
    test4_pass = 0
    test4_total = 3

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 AGENT FACTORY TEST SUMMARY")
print("="*80)

tests = [
    ("Initialization", test1_pass, 1),
    ("Create Agents", test2_pass, test2_total),
    ("Agent Metadata", test3_pass, 1),
    ("Capability Matching", test4_pass, test4_total),
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
    print("\n✅ EXCELLENT: Agent factory working well")
elif overall_pct >= 60:
    print("\n⚠️ GOOD: Agent factory functional with minor issues")
else:
    print("\n❌ CONCERN: Agent factory needs work")

print("="*80)
