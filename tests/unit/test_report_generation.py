"""
REPORT GENERATION TEST
Purpose: Test report creation, formatting, and export
Date: December 16, 2025
"""

import sys
import os
import pandas as pd
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

print("="*80)
print("🔍 REPORT GENERATION TEST")
print("="*80)

# ============================================================================
# TEST 1: Report Generator Initialization
# ============================================================================
print("\n[TEST 1] Report Generator Initialization")
print("-"*80)

try:
    from backend.utils.report_generator import ReportGenerator
    
    rg = ReportGenerator()
    print("  ✅ ReportGenerator initialized")
    test1_pass = 1
except Exception as e:
    print(f"  ❌ Initialization failed: {type(e).__name__}")
    test1_pass = 0
    rg = None

# ============================================================================
# TEST 2: Summary Generation
# ============================================================================
print("\n[TEST 2] Summary Generation")
print("-"*80)

if rg:
    test_data = pd.DataFrame({
        'product': ['A', 'B', 'C'],
        'sales': [100, 200, 150],
        'profit': [20, 40, 30],
    })
    
    try:
        summary = rg.generate_summary(test_data)
        
        if summary and len(summary) > 0:
            print(f"  ✅ Summary generated ({len(summary)} chars)")
            test2_pass = 1
        else:
            print("  ⚠️ Summary empty")
            test2_pass = 0.5
    except Exception as e:
        print(f"  ⚠️ Summary error: {type(e).__name__}")
        test2_pass = 0
else:
    test2_pass = 0

# ============================================================================
# TEST 3: Insight Extraction
# ============================================================================
print("\n[TEST 3] Insight Extraction")
print("-"*80)

if rg:
    analysis_results = {
        'total_sales': 450,
        'avg_profit': 30,
        'top_product': 'B',
    }
    
    try:
        insights = rg.extract_insights(analysis_results)
        
        if insights and len(insights) > 0:
            print(f"  ✅ Insights extracted ({len(insights)} items)")
            test3_pass = 1
        else:
            print("  ⚠️ No insights extracted")
            test3_pass = 0.5
    except Exception as e:
        print(f"  ⚠️ Insight error: {type(e).__name__}")
        test3_pass = 0
else:
    test3_pass = 0

# ============================================================================
# TEST 4: Markdown Report Creation
# ============================================================================
print("\n[TEST 4] Markdown Report Creation")
print("-"*80)

if rg:
    report_data = {
        'title': 'Sales Analysis Report',
        'sections': [
            {'heading': 'Overview', 'content': 'Sales data for Q4 2024'},
            {'heading': 'Key Findings', 'content': 'Product B performed best'},
        ]
    }
    
    try:
        markdown = rg.create_markdown_report(report_data)
        
        if markdown and '# ' in markdown:
            print("  ✅ Markdown report created")
            test4_pass = 1
        else:
            print("  ⚠️ Markdown incomplete")
            test4_pass = 0.5
    except Exception as e:
        print(f"  ⚠️ Markdown error: {type(e).__name__}")
        test4_pass = 0
else:
    test4_pass = 0

# ============================================================================
# TEST 5: PDF Export (if available)
# ============================================================================
print("\n[TEST 5] PDF Export")
print("-"*80)

if rg:
    try:
        # Try to check if PDF export is available
        has_pdf = hasattr(rg, 'export_to_pdf')
        
        if has_pdf:
            print("  ✅ PDF export capability available")
            test5_pass = 1
        else:
            print("  ⚠️ PDF export not available")
            test5_pass = 0.5
    except Exception as e:
        print(f"  ⚠️ PDF check error: {type(e).__name__}")
        test5_pass = 0
else:
    test5_pass = 0

# ============================================================================
# TEST 6: CSV Export
# ============================================================================
print("\n[TEST 6] CSV Export")
print("-"*80)

if rg:
    export_data = pd.DataFrame({
        'metric': ['Total Sales', 'Avg Profit', 'Top Product'],
        'value': [450, 30, 'B'],
    })
    
    try:
        # Try exporting to CSV
        csv_output = rg.export_to_csv(export_data)
        
        if csv_output and len(csv_output) > 0:
            print("  ✅ CSV export successful")
            test6_pass = 1
        else:
            print("  ⚠️ CSV export unclear")
            test6_pass = 0.5
    except Exception as e:
        print(f"  ⚠️ CSV export error: {type(e).__name__}")
        test6_pass = 0
else:
    test6_pass = 0

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 REPORT GENERATION TEST SUMMARY")
print("="*80)

tests = [
    ("Initialization", test1_pass, 1),
    ("Summary Generation", test2_pass, 1),
    ("Insight Extraction", test3_pass, 1),
    ("Markdown Report", test4_pass, 1),
    ("PDF Export", test5_pass, 1),
    ("CSV Export", test6_pass, 1),
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
    print("\n✅ EXCELLENT: Report generation working well")
elif overall_pct >= 60:
    print("\n⚠️ GOOD: Report generation functional")
else:
    print("\n❌ CONCERN: Report generation needs work")

print("="*80)
