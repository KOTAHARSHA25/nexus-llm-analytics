# Fix 17: API Endpoint Test
# Test the /api/report/pdf endpoint

import requests
import json
from datetime import datetime

print("="*80)
print("🌐 FIX 17: PDF API ENDPOINT TEST")
print("="*80)
print()

# Sample analysis result
sample_result = {
    "query": "What is the correlation between marketing spend and sales revenue?",
    "success": True,
    "result": "Strong positive correlation detected (r=0.87, p<0.001)",
    "interpretation": """The analysis reveals a robust positive correlation between marketing 
spend and sales revenue with a correlation coefficient of 0.87 (p<0.001), indicating 
statistical significance.

Key findings:
1. For every $1 increase in marketing spend, sales revenue increases by $3.45 on average
2. The relationship is strongest in Q4 (holiday season)
3. Digital marketing channels show higher ROI than traditional channels
4. Diminishing returns observed above $50K monthly spend

Recommendations:
- Optimize marketing budget allocation to maximize ROI
- Focus on high-performing digital channels
- Consider seasonal adjustments to spending patterns""",
    "agent": "StatisticalAgent",
    "model_used": "llama3.1:8b",
    "execution_time": 3.2,
    "timestamp": datetime.now().isoformat(),
    "orchestrator_reasoning": {
        "confidence": 0.92,
        "reasoning": "Correlation analysis requires statistical methods",
        "selected_route": "statistical_analysis"
    },
    "routing_decision": {
        "agent": "StatisticalAgent",
        "confidence": 0.92,
        "reasoning": "Statistical correlation query"
    },
    "insights": [
        "Strong correlation (r=0.87) between variables",
        "Statistical significance confirmed (p<0.001)",
        "Average ROI of $3.45 per $1 spent",
        "Q4 shows peak correlation strength",
        "Diminishing returns above $50K threshold"
    ],
    "key_metrics": {
        "correlation_coefficient": 0.87,
        "p_value": 0.0001,
        "r_squared": 0.7569,
        "average_roi": 3.45,
        "sample_size": 156
    },
    "statistics": {
        "mean_marketing_spend": 28500,
        "median_marketing_spend": 25000,
        "std_dev_spend": 12300,
        "mean_sales": 98400,
        "median_sales": 86250,
        "std_dev_sales": 42500
    },
    "metadata": {
        "rows": 156,
        "columns": 12,
        "agent": "StatisticalAgent",
        "model": "llama3.1:8b",
        "execution_time": 3.2
    },
    "code_generated": """import pandas as pd
import numpy as np
from scipy import stats

# Calculate correlation
correlation, p_value = stats.pearsonr(df['marketing_spend'], df['sales_revenue'])

# Calculate R-squared
r_squared = correlation ** 2

# Linear regression for ROI
slope, intercept, r_val, p_val, std_err = stats.linregress(
    df['marketing_spend'], 
    df['sales_revenue']
)

print(f"Correlation: {correlation:.2f}")
print(f"P-value: {p_value:.4f}")
print(f"R-squared: {r_squared:.4f}")
print(f"ROI: ${slope:.2f} per $1 spent")"
"""
}

# Test 1: Generate PDF via API
print("📡 Test 1: Calling /api/report/pdf endpoint")
print("-" * 80)

try:
    response = requests.post(
        "http://localhost:8000/api/report/pdf",
        json={
            "analysis_result": sample_result,
            "include_raw_data": True,
            "custom_filename": "marketing_correlation_analysis"
        },
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ SUCCESS: API responded with status 200")
        print(f"   - Success: {data.get('success')}")
        print(f"   - Message: {data.get('message')}")
        print(f"   - Report Path: {data.get('report_path')}")
        print(f"   - Download URL: {data.get('download_url')}")
        print(f"   - File Size: {data.get('metadata', {}).get('file_size')}")
        print(f"   - Features: {len(data.get('features', []))} included")
        print()
        print("   Features included:")
        for feature in data.get('features', [])[:10]:
            print(f"      • {feature}")
    else:
        print(f"❌ FAIL: API responded with status {response.status_code}")
        print(f"   Response: {response.text}")
except requests.exceptions.ConnectionError:
    print("⚠️  SKIP: Backend not running (expected for offline test)")
    print("   To test the API endpoint:")
    print("   1. Start backend: python src/backend/main.py")
    print("   2. Run this test again")
except Exception as e:
    print(f"❌ FAIL: {e}")

print()

# Test 2: Generate PDF with minimal data
print("📡 Test 2: Testing with minimal data")
print("-" * 80)

minimal_result = {
    "query": "Simple test",
    "result": "Test result",
    "success": True
}

try:
    response = requests.post(
        "http://localhost:8000/api/report/pdf",
        json={
            "analysis_result": minimal_result,
            "include_raw_data": False
        },
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ SUCCESS: Minimal data PDF generated")
        print(f"   - Report Path: {data.get('report_path')}")
    else:
        print(f"❌ FAIL: Status {response.status_code}")
except requests.exceptions.ConnectionError:
    print("⚠️  SKIP: Backend not running")
except Exception as e:
    print(f"❌ FAIL: {e}")

print()

# Test 3: Download the generated report
print("📡 Test 3: Testing download endpoint")
print("-" * 80)

try:
    response = requests.get(
        "http://localhost:8000/api/report/download-report?filename=marketing_correlation_analysis.pdf",
        timeout=30
    )
    
    if response.status_code == 200 and response.headers.get('content-type') == 'application/pdf':
        print(f"✅ SUCCESS: PDF downloaded successfully")
        print(f"   - Content-Type: {response.headers.get('content-type')}")
        print(f"   - Size: {len(response.content)} bytes")
        
        # Save to test downloads
        with open('downloaded_report_test.pdf', 'wb') as f:
            f.write(response.content)
        print(f"   - Saved to: downloaded_report_test.pdf")
    else:
        print(f"❌ FAIL: Status {response.status_code}")
except requests.exceptions.ConnectionError:
    print("⚠️  SKIP: Backend not running")
except Exception as e:
    print(f"❌ FAIL: {e}")

print()
print("="*80)
print("📊 API TEST SUMMARY")
print("="*80)
print()
print("To run complete API tests:")
print("1. Start backend: python src/backend/main.py")
print("2. Run: python test_fix17_api.py")
print()
print("Manual API test:")
print("""
curl -X POST http://localhost:8000/api/report/pdf \\
  -H "Content-Type: application/json" \\
  -d '{
    "analysis_result": {
      "query": "Test query",
      "result": "Test result",
      "success": true
    },
    "include_raw_data": true
  }'
""")
print()
print("="*80)
