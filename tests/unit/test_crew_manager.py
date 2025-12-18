"""
CREW MANAGER TEST
Purpose: Test multi-agent coordination and task delegation
Date: December 16, 2025
"""

import sys
import os
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

print("="*80)
print("🔍 CREW MANAGER TEST")
print("="*80)

# ============================================================================
# TEST 1: Crew Manager Initialization
# ============================================================================
print("\n[TEST 1] Crew Manager Initialization")
print("-"*80)

try:
    from backend.agents.crew_manager import CrewManager
    
    crew = CrewManager()
    print("  ✅ CrewManager initialized")
    test1_pass = 1
except Exception as e:
    print(f"  ❌ Initialization failed: {type(e).__name__}")
    test1_pass = 0
    crew = None

# ============================================================================
# TEST 2: Task Delegation
# ============================================================================
print("\n[TEST 2] Task Delegation")
print("-"*80)

if crew:
    test_tasks = [
        "Calculate average sales",
        "Predict next quarter revenue",
        "Find products with high ROI",
    ]
    
    test2_results = []
    for task in test_tasks:
        try:
            result = crew.delegate_task(task)
            
            if result:
                print(f"  ✅ Delegated: '{task[:40]}...'")
                test2_results.append(1)
            else:
                print(f"  ⚠️ No delegation: '{task[:40]}...'")
                test2_results.append(0.5)
        except Exception as e:
            print(f"  ⚠️ Error: {type(e).__name__}")
            test2_results.append(0)
    
    test2_pass = sum(test2_results)
    test2_total = len(test_tasks)
else:
    test2_pass = 0
    test2_total = 3

# ============================================================================
# TEST 3: Agent Coordination
# ============================================================================
print("\n[TEST 3] Agent Coordination")
print("-"*80)

if crew:
    complex_query = "Calculate total sales and predict future trends"
    
    try:
        result = crew.coordinate_agents(complex_query)
        
        if result:
            print(f"  ✅ Multi-agent coordination successful")
            test3_pass = 1
        else:
            print("  ⚠️ Coordination unclear")
            test3_pass = 0.5
    except Exception as e:
        print(f"  ⚠️ Coordination error: {type(e).__name__}")
        test3_pass = 0
else:
    test3_pass = 0

# ============================================================================
# TEST 4: Agent Pool Management
# ============================================================================
print("\n[TEST 4] Agent Pool Management")
print("-"*80)

if crew:
    try:
        # Check if can list available agents
        agents = crew.list_agents()
        
        if agents and len(agents) > 0:
            print(f"  ✅ Agent pool: {len(agents)} agents")
            test4_pass = 1
        else:
            print("  ⚠️ No agents in pool")
            test4_pass = 0.5
    except Exception as e:
        print(f"  ⚠️ Pool error: {type(e).__name__}")
        test4_pass = 0
else:
    test4_pass = 0

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 CREW MANAGER TEST SUMMARY")
print("="*80)

tests = [
    ("Initialization", test1_pass, 1),
    ("Task Delegation", test2_pass, test2_total),
    ("Agent Coordination", test3_pass, 1),
    ("Agent Pool Management", test4_pass, 1),
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
    print("\n✅ EXCELLENT: Crew manager working well")
elif overall_pct >= 60:
    print("\n⚠️ GOOD: Crew manager functional")
else:
    print("\n❌ CONCERN: Crew manager needs work")

print("="*80)
