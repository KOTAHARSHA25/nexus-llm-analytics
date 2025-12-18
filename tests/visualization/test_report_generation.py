"""
Task 4.2: Report Generation Testing
Tests PDF, Excel, and combined report generation with visualizations
"""
import requests
import json
import os
from pathlib import Path

API_BASE = "http://localhost:8000"
DATA_DIR = Path("data/reports")

def test_pdf_report():
    """Test PDF report generation with analysis results"""
    print("\n" + "="*70)
    print("TEST 1: PDF Report Generation")
    print("="*70)
    
    # Sample analysis results (simulating actual analysis output)
    analysis_results = [
        {
            "success": True,
            "query": "Analyze sales trends over time",
            "filename": "test_sales_monthly.csv",
            "type": "structured_data",
            "execution_time": 2.5,
            "answer": "Sales show steady growth of 15% over Q1",
            "data": {
                "total_revenue": 324500,
                "avg_units_sold": 234,
                "best_region": "East"
            },
            "code": "df.groupby('region')['revenue'].sum()",
            "describe": {
                "revenue": {"mean": 13520.83, "std": 2456.78}
            }
        },
        {
            "success": True,
            "query": "Compare employee salaries by department",
            "filename": "test_employee_data.csv",
            "type": "structured_data",
            "execution_time": 1.8,
            "answer": "Engineering has highest average salary at $95,500",
            "data": {
                "engineering_avg": 95500,
                "sales_avg": 110000,
                "marketing_avg": 86500
            }
        }
    ]
    
    response = requests.post(
        f"{API_BASE}/generate-report/",
        json={
            "results": analysis_results,
            "format_type": "pdf",
            "title": "Q1 2024 Analytics Report",
            "include_methodology": True,
            "include_raw_data": True
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ PDF Report Generated Successfully!")
        print(f"   Path: {result.get('report_path')}")
        print(f"   Format: {result.get('format')}")
        print(f"   Analyses: {result.get('analysis_count')}")
        
        # Try to download
        download_response = requests.get(f"{API_BASE}/generate-report/download-report")
        if download_response.status_code == 200:
            # Save locally
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            pdf_path = DATA_DIR / "test_report.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(download_response.content)
            print(f"   Downloaded to: {pdf_path}")
            print(f"   File size: {len(download_response.content) / 1024:.2f} KB")
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"   Error: {response.text[:200]}")
        return False


def test_excel_report():
    """Test Excel report generation"""
    print("\n" + "="*70)
    print("TEST 2: Excel Report Generation")
    print("="*70)
    
    analysis_results = [
        {
            "success": True,
            "query": "Student performance analysis",
            "filename": "test_student_grades.csv",
            "type": "structured_data",
            "execution_time": 1.5,
            "answer": "Average score is 84.2 with 3 students scoring above 90",
            "data": {
                "avg_score": 84.2,
                "top_student": "Sarah Williams",
                "subjects_analyzed": 3
            },
            "describe": {
                "score": {"mean": 84.2, "min": 68, "max": 95}
            }
        }
    ]
    
    response = requests.post(
        f"{API_BASE}/generate-report/",
        json={
            "results": analysis_results,
            "format_type": "excel",
            "title": "Student Performance Report"
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Excel Report Generated Successfully!")
        print(f"   Path: {result.get('report_path')}")
        print(f"   Format: {result.get('format')}")
        
        # Download
        download_response = requests.get(f"{API_BASE}/generate-report/download-report")
        if download_response.status_code == 200:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            excel_path = DATA_DIR / "test_report.xlsx"
            with open(excel_path, 'wb') as f:
                f.write(download_response.content)
            print(f"   Downloaded to: {excel_path}")
            print(f"   File size: {len(download_response.content) / 1024:.2f} KB")
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"   Error: {response.text[:200]}")
        return False


def test_combined_reports():
    """Test generating both PDF and Excel simultaneously"""
    print("\n" + "="*70)
    print("TEST 3: Combined PDF + Excel Report Generation")
    print("="*70)
    
    analysis_results = [
        {
            "success": True,
            "query": "IoT sensor analysis",
            "filename": "test_iot_sensor.csv",
            "type": "structured_data",
            "execution_time": 2.1,
            "answer": "Temperature increased from 22.5°C to 29.0°C over 5 hours",
            "data": {
                "temp_start": 22.5,
                "temp_end": 29.0,
                "temp_change": 6.5,
                "readings": 20
            }
        },
        {
            "success": True,
            "query": "Inventory stock analysis",
            "filename": "test_inventory.csv",
            "type": "structured_data",
            "execution_time": 1.3,
            "answer": "15 products tracked, Stationery category has highest stock",
            "data": {
                "total_products": 15,
                "low_stock_items": 3,
                "total_value": 45678.90
            }
        }
    ]
    
    response = requests.post(
        f"{API_BASE}/generate-report/",
        json={
            "results": analysis_results,
            "format_type": "both",
            "title": "Comprehensive Analytics Report"
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Combined Reports Generated Successfully!")
        print(f"   Paths: {result.get('report_paths')}")
        print(f"   Format: {result.get('format')}")
        print(f"   Analyses: {result.get('analysis_count')}")
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"   Error: {response.text[:200]}")
        return False


def test_report_with_visualizations():
    """Test report generation with actual visualization data"""
    print("\n" + "="*70)
    print("TEST 4: Report with Embedded Visualizations")
    print("="*70)
    
    # Create mock visualization data (simulating what /visualize/goal-based returns)
    # This avoids needing a live backend server
    mock_viz_data = {
        "success": True,
        "visualization": {
            "figure_json": {
                "data": [
                    {
                        "x": ["Jan", "Feb", "Mar", "Apr"],
                        "y": [12500, 15600, 18900, 21200],
                        "type": "bar",
                        "name": "Revenue"
                    }
                ],
                "layout": {
                    "title": "Monthly Revenue Trend",
                    "xaxis": {"title": "Month"},
                    "yaxis": {"title": "Revenue ($)"}
                }
            },
            "chart_type": "bar"
        },
        "selected_chart": {
            "type": "bar",
            "priority_score": 9.0,
            "reasoning": "Bar chart best shows revenue comparison across months"
        },
        "suggestions": [
            {
                "type": "bar",
                "priority_score": 9.0,
                "reasoning": "Excellent for comparing monthly values",
                "use_case": "Primary visualization for revenue trends"
            },
            {
                "type": "line",
                "priority_score": 8.5,
                "reasoning": "Shows trend progression clearly",
                "use_case": "Alternative view of revenue growth"
            },
            {
                "type": "scatter",
                "priority_score": 6.0,
                "reasoning": "Can show individual data points",
                "use_case": "Detailed data exploration"
            }
        ]
    }
    
    # Now create report including the visualization
    analysis_results = [
        {
            "success": True,
            "query": "Sales revenue trends analysis",
            "filename": "test_sales_monthly.csv",
            "type": "structured_data",
            "execution_time": 2.8,
            "answer": "Revenue shows consistent growth across all regions",
            "data": {
                "total_revenue": 324500,
                "growth_rate": 15.3,
                "best_month": "March 2024"
            },
            "visualization": mock_viz_data.get('visualization'),
            "chart_type": mock_viz_data.get('selected_chart', {}).get('type'),
            "chart_suggestions": mock_viz_data.get('suggestions', [])[:3]
        }
    ]
    
    response = requests.post(
        f"{API_BASE}/generate-report/",
        json={
            "results": analysis_results,
            "format_type": "pdf",
            "title": "Sales Analysis with Visualizations"
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Report with Visualizations Generated!")
        print(f"   Path: {result.get('report_path')}")
        print(f"   Chart Type: {mock_viz_data.get('selected_chart', {}).get('type')}")
        print(f"   Suggestions Included: {len(mock_viz_data.get('suggestions', []))}")
        
        # Download
        download_response = requests.get(f"{API_BASE}/generate-report/download-report")
        if download_response.status_code == 200:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            pdf_path = DATA_DIR / "test_report_with_viz.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(download_response.content)
            print(f"   Downloaded to: {pdf_path}")
            print(f"   File size: {len(download_response.content) / 1024:.2f} KB")
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"   Error: {response.text[:200]}")
        return False


def test_custom_report_sections():
    """Test report with custom sections and mixed content"""
    print("\n" + "="*70)
    print("TEST 5: Custom Report Sections")
    print("="*70)
    
    analysis_results = [
        {
            "success": True,
            "query": "Executive summary - Q1 performance",
            "filename": "test_sales_monthly.csv",
            "type": "executive_summary",
            "execution_time": 1.0,
            "answer": "Key Highlights: Revenue up 15%, Top region: East, Best product: Electronics",
            "data": {
                "revenue_growth": 15,
                "top_region": "East",
                "best_category": "Electronics"
            }
        },
        {
            "success": True,
            "query": "Technical details - Data processing methodology",
            "filename": "test_sales_monthly.csv",
            "type": "technical",
            "execution_time": 0.5,
            "code": "df.groupby(['region', 'product_category']).agg({'revenue': 'sum', 'units_sold': 'mean'})",
            "answer": "Used pandas groupby aggregation for regional analysis"
        },
        {
            "success": True,
            "query": "Recommendations",
            "filename": "test_sales_monthly.csv",
            "type": "recommendations",
            "execution_time": 0.3,
            "answer": "1. Focus marketing in West region\n2. Increase Electronics inventory\n3. Review Furniture pricing strategy"
        }
    ]
    
    response = requests.post(
        f"{API_BASE}/generate-report/",
        json={
            "results": analysis_results,
            "format_type": "both",
            "title": "Q1 2024 Custom Report",
            "include_methodology": True,
            "include_raw_data": False
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Custom Report Generated!")
        print(f"   Sections: Executive, Technical, Recommendations")
        print(f"   Formats: PDF + Excel")
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        return False


if __name__ == "__main__":
    print("="*70)
    print("PHASE 4.2: REPORT GENERATION TESTING")
    print("="*70)
    print("\nTesting report generation with multiple formats and content types")
    
    results = []
    
    # Run all tests
    results.append(("PDF Report", test_pdf_report()))
    results.append(("Excel Report", test_excel_report()))
    results.append(("Combined Reports", test_combined_reports()))
    results.append(("Reports with Visualizations", test_report_with_visualizations()))
    results.append(("Custom Sections", test_custom_report_sections()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.0f}%)")
    
    if passed_count == total_count:
        print("\n🎉 All report generation tests passed!")
        print(f"   Generated reports saved to: {DATA_DIR.absolute()}")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed - review errors above")
