"""
Automated Frontend Verification Tests
Tests all frontend functionality without manual browser interaction
"""
import requests
import json
import time
from pathlib import Path

# Configuration
FRONTEND_URL = "http://localhost:3000"
BACKEND_URL = "http://localhost:8000"
TEST_DATA_DIR = Path("data/samples")

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_test(test_name, status, details=""):
    """Print test result with colors"""
    if status == "PASS":
        print(f"{Colors.GREEN}✅ PASS{Colors.RESET}: {test_name}")
    elif status == "FAIL":
        print(f"{Colors.RED}❌ FAIL{Colors.RESET}: {test_name}")
        if details:
            print(f"   {Colors.YELLOW}Details: {details}{Colors.RESET}")
    elif status == "SKIP":
        print(f"{Colors.YELLOW}⏭️  SKIP{Colors.RESET}: {test_name}")
    else:
        print(f"{Colors.BLUE}⏳ TEST{Colors.RESET}: {test_name}")
    
    if details and status == "PASS":
        print(f"   {Colors.BLUE}{details}{Colors.RESET}")


def test_backend_health():
    """Test 0: Backend health check"""
    try:
        response = requests.get(f"{BACKEND_URL}/health/health", timeout=5)
        if response.status_code == 200:
            print_test("Backend Health Check", "PASS", "Backend is running")
            return True
        else:
            print_test("Backend Health Check", "FAIL", f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_test("Backend Health Check", "FAIL", str(e))
        return False


def test_frontend_loading():
    """Test 1: Frontend loading"""
    try:
        response = requests.get(FRONTEND_URL, timeout=5)
        if response.status_code == 200 and "Nexus LLM Analytics" in response.text:
            print_test("Frontend Loading", "PASS", "Frontend page loads successfully")
            return True
        else:
            print_test("Frontend Loading", "FAIL", "Page content unexpected")
            return False
    except Exception as e:
        print_test("Frontend Loading", "FAIL", str(e))
        return False


def test_file_upload():
    """Test 2: File upload functionality"""
    try:
        # Use test CSV file
        test_file = TEST_DATA_DIR / "test_sales_monthly.csv"
        if not test_file.exists():
            print_test("File Upload", "SKIP", "Test file not found")
            return None
        
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'text/csv')}
            response = requests.post(f"{BACKEND_URL}/upload-documents/", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print_test("File Upload (CSV)", "PASS", f"Uploaded: {data.get('filename')}")
                return data.get('filename')
            else:
                print_test("File Upload (CSV)", "FAIL", data.get('error', 'Unknown error'))
                return False
        else:
            print_test("File Upload (CSV)", "FAIL", f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("File Upload (CSV)", "FAIL", str(e))
        return False


def test_text_input_analysis():
    """Test 3: Text input analysis"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze",
            json={
                "query": "What information can you extract?",
                "text_data": "my name is harsha and i study at mlrit in the course of aiml"
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success') or data.get('result'):
                result = data.get('result', '')
                preview = result[:100] + "..." if len(result) > 100 else result
                print_test("Text Input Analysis", "PASS", f"Result: {preview}")
                return True
            else:
                print_test("Text Input Analysis", "FAIL", data.get('error', 'No result'))
                return False
        else:
            print_test("Text Input Analysis", "FAIL", f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("Text Input Analysis", "FAIL", str(e))
        return False


def test_file_analysis(filename):
    """Test 4: File-based analysis"""
    if not filename:
        print_test("File Analysis", "SKIP", "No uploaded file")
        return None
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze",
            json={
                "query": "What is the total revenue?",
                "filename": filename
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('result') or data.get('success'):
                result = data.get('result', '')
                preview = result[:100] + "..." if len(result) > 100 else result
                print_test("File Analysis", "PASS", f"Result: {preview}")
                return data
            else:
                print_test("File Analysis", "FAIL", data.get('error', 'No result'))
                return False
        else:
            print_test("File Analysis", "FAIL", f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("File Analysis", "FAIL", str(e))
        return False


def test_chart_suggestions(filename):
    """Test 5: Chart suggestions"""
    if not filename:
        print_test("Chart Suggestions", "SKIP", "No uploaded file")
        return None
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/visualize/suggestions",
            json={"filename": filename},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            suggestions = data.get('suggestions', [])
            if suggestions:
                print_test("Chart Suggestions", "PASS", 
                          f"Got {len(suggestions)} suggestions. Top: {suggestions[0].get('type')}")
                return suggestions
            else:
                print_test("Chart Suggestions", "FAIL", "No suggestions returned")
                return False
        else:
            print_test("Chart Suggestions", "FAIL", f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("Chart Suggestions", "FAIL", str(e))
        return False


def test_chart_generation(filename):
    """Test 6: Chart generation"""
    if not filename:
        print_test("Chart Generation", "SKIP", "No uploaded file")
        return None
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/visualize/goal-based",
            json={
                "filename": filename,
                "goal": "Show revenue trends over time",
                "library": "plotly"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            viz = data.get('visualization', {})
            if viz.get('figure_json'):
                chart_type = data.get('selected_chart', {}).get('type', 'unknown')
                print_test("Chart Generation", "PASS", f"Generated {chart_type} chart")
                return data
            else:
                print_test("Chart Generation", "FAIL", "No figure_json in response")
                return False
        else:
            print_test("Chart Generation", "FAIL", f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("Chart Generation", "FAIL", str(e))
        return False


def test_review_insights(analysis_result):
    """Test 7: Review insights generation"""
    if not analysis_result:
        print_test("Review Insights", "SKIP", "No analysis result")
        return None
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze/review-insights",
            json={"analysis_results": [analysis_result]},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            insights = data.get('insights', '')
            if insights:
                preview = insights[:100] + "..." if len(insights) > 100 else insights
                print_test("Review Insights", "PASS", f"Insights: {preview}")
                return insights
            else:
                print_test("Review Insights", "FAIL", "No insights returned")
                return False
        else:
            print_test("Review Insights", "FAIL", f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("Review Insights", "FAIL", str(e))
        return False


def test_report_generation(analysis_result):
    """Test 8: Report generation"""
    if not analysis_result:
        print_test("Report Generation", "SKIP", "No analysis result")
        return None
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/generate-report/",
            json={
                "results": [analysis_result],
                "format_type": "pdf",
                "title": "Frontend Verification Test Report"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success') or data.get('report_path'):
                report_path = data.get('report_path', 'Unknown')
                print_test("Report Generation", "PASS", f"Report: {report_path}")
                return True
            else:
                print_test("Report Generation", "FAIL", data.get('error', 'Unknown error'))
                return False
        else:
            print_test("Report Generation", "FAIL", f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("Report Generation", "FAIL", str(e))
        return False


def test_model_preferences():
    """Test 9: Model preferences API"""
    try:
        response = requests.get(f"{BACKEND_URL}/models/preferences", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            prefs = data.get('preferences', {})
            if prefs:
                primary = prefs.get('primary_model', 'N/A')
                review = prefs.get('review_model', 'N/A')
                print_test("Model Preferences", "PASS", 
                          f"Primary: {primary}, Review: {review}")
                return prefs
            else:
                print_test("Model Preferences", "FAIL", "No preferences returned")
                return False
        else:
            print_test("Model Preferences", "FAIL", f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("Model Preferences", "FAIL", str(e))
        return False


def test_available_chart_types():
    """Test 10: Available chart types endpoint"""
    try:
        response = requests.get(f"{BACKEND_URL}/visualize/types", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            chart_types = data.get('chart_types', {})
            if chart_types:
                type_list = list(chart_types.keys())
                print_test("Chart Types API", "PASS", 
                          f"Available: {', '.join(type_list[:5])}")
                return True
            else:
                print_test("Chart Types API", "FAIL", "No types returned")
                return False
        else:
            print_test("Chart Types API", "FAIL", f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("Chart Types API", "FAIL", str(e))
        return False


def run_all_tests():
    """Run all frontend verification tests"""
    print("="*70)
    print(f"{Colors.BLUE}FRONTEND VERIFICATION TEST SUITE{Colors.RESET}")
    print(f"{Colors.BLUE}Date: November 4, 2025{Colors.RESET}")
    print("="*70)
    print()
    
    results = {}
    
    # Test 0: Backend health
    print(f"\n{Colors.YELLOW}>>> Phase 1: System Health{Colors.RESET}")
    results['backend_health'] = test_backend_health()
    if not results['backend_health']:
        print(f"\n{Colors.RED}❌ Backend not running. Cannot continue tests.{Colors.RESET}")
        return results
    
    results['frontend_loading'] = test_frontend_loading()
    
    # Test file operations
    print(f"\n{Colors.YELLOW}>>> Phase 2: File Upload & Analysis{Colors.RESET}")
    uploaded_filename = test_file_upload()
    results['file_upload'] = uploaded_filename is not False
    
    analysis_result = test_file_analysis(uploaded_filename)
    results['file_analysis'] = analysis_result is not False
    
    # Test text input
    print(f"\n{Colors.YELLOW}>>> Phase 3: Text Input{Colors.RESET}")
    results['text_input'] = test_text_input_analysis()
    
    # Test visualizations
    print(f"\n{Colors.YELLOW}>>> Phase 4: Visualizations{Colors.RESET}")
    chart_suggestions = test_chart_suggestions(uploaded_filename)
    results['chart_suggestions'] = chart_suggestions is not False
    
    chart_data = test_chart_generation(uploaded_filename)
    results['chart_generation'] = chart_data is not False
    
    results['chart_types'] = test_available_chart_types()
    
    # Test review and reports
    print(f"\n{Colors.YELLOW}>>> Phase 5: Review & Reports{Colors.RESET}")
    review = test_review_insights(analysis_result if analysis_result else None)
    results['review_insights'] = review is not False
    
    report = test_report_generation(analysis_result if analysis_result else None)
    results['report_generation'] = report is not False
    
    # Test settings
    print(f"\n{Colors.YELLOW}>>> Phase 6: Settings{Colors.RESET}")
    results['model_preferences'] = test_model_preferences() is not False
    
    # Summary
    print("\n" + "="*70)
    print(f"{Colors.BLUE}TEST SUMMARY{Colors.RESET}")
    print("="*70)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result is True else "❌ FAIL" if result is False else "⏭️  SKIP"
        print(f"  {status}: {test_name.replace('_', ' ').title()}")
    
    print()
    print(f"Total Tests: {total}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
    print(f"{Colors.YELLOW}Skipped: {skipped}{Colors.RESET}")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print(f"\n{Colors.GREEN}🎉 EXCELLENT! Frontend verification passed!{Colors.RESET}")
    elif success_rate >= 70:
        print(f"\n{Colors.YELLOW}⚠️  GOOD, but some tests failed. Review failures.{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}❌ FAILED. Multiple critical issues found.{Colors.RESET}")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
