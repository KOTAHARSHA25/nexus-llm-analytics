"""
API ENDPOINTS TEST
Purpose: Test FastAPI routes and request handling
Date: December 16, 2025
"""

import sys
import os
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

print("="*80)
print("🔍 API ENDPOINTS TEST")
print("="*80)

# ============================================================================
# TEST 1: FastAPI App Initialization
# ============================================================================
print("\n[TEST 1] FastAPI App Initialization")
print("-"*80)

try:
    from fastapi import FastAPI
    from backend.main import app
    
    if app and isinstance(app, FastAPI):
        print("  ✅ FastAPI app initialized")
        test1_pass = 1
    else:
        print("  ❌ App is not FastAPI instance")
        test1_pass = 0
except Exception as e:
    print(f"  ❌ Initialization failed: {type(e).__name__}")
    test1_pass = 0
    app = None

# ============================================================================
# TEST 2: Route Registration
# ============================================================================
print("\n[TEST 2] Route Registration")
print("-"*80)

if app:
    expected_routes = [
        '/upload',
        '/analyze',
        '/visualize',
        '/report',
    ]
    
    # Get all routes from app
    registered_routes = [route.path for route in app.routes]
    
    test2_results = []
    for route in expected_routes:
        if route in registered_routes:
            print(f"  ✅ Route registered: {route}")
            test2_results.append(1)
        else:
            print(f"  ⚠️ Route missing: {route}")
            test2_results.append(0)
    
    test2_pass = sum(test2_results)
    test2_total = len(expected_routes)
else:
    test2_pass = 0
    test2_total = 4

# ============================================================================
# TEST 3: Health Check Endpoint
# ============================================================================
print("\n[TEST 3] Health Check Endpoint")
print("-"*80)

if app:
    health_routes = ['/', '/health', '/ping']
    
    found = False
    for route in health_routes:
        if route in [r.path for r in app.routes]:
            print(f"  ✅ Health endpoint found: {route}")
            found = True
            break
    
    test3_pass = 1 if found else 0
    if not found:
        print("  ⚠️ No health check endpoint found")
else:
    test3_pass = 0

# ============================================================================
# TEST 4: API Router Import
# ============================================================================
print("\n[TEST 4] API Router Import")
print("-"*80)

routers = []
try:
    from backend.api import upload
    routers.append('upload')
    print("  ✅ Upload router imported")
except:
    print("  ⚠️ Upload router not found")

try:
    from backend.api import analyze
    routers.append('analyze')
    print("  ✅ Analyze router imported")
except:
    print("  ⚠️ Analyze router not found")

try:
    from backend.api import visualize
    routers.append('visualize')
    print("  ✅ Visualize router imported")
except:
    print("  ⚠️ Visualize router not found")

try:
    from backend.api import report
    routers.append('report')
    print("  ✅ Report router imported")
except:
    print("  ⚠️ Report router not found")

test4_pass = len(routers)
test4_total = 4

# ============================================================================
# TEST 5: CORS Configuration
# ============================================================================
print("\n[TEST 5] CORS Configuration")
print("-"*80)

if app:
    # Check if CORS middleware is configured
    middleware = [m for m in app.user_middleware if 'CORS' in str(type(m))]
    
    if len(middleware) > 0:
        print(f"  ✅ CORS middleware configured")
        test5_pass = 1
    else:
        print("  ⚠️ CORS middleware not found")
        test5_pass = 0
else:
    test5_pass = 0

# ============================================================================
# TEST 6: Request/Response Models
# ============================================================================
print("\n[TEST 6] Request/Response Models")
print("-"*80)

models_found = []

try:
    from backend.api.models import AnalyzeRequest
    models_found.append('AnalyzeRequest')
    print("  ✅ AnalyzeRequest model found")
except:
    print("  ⚠️ AnalyzeRequest model not found")

try:
    from backend.api.models import AnalyzeResponse
    models_found.append('AnalyzeResponse')
    print("  ✅ AnalyzeResponse model found")
except:
    print("  ⚠️ AnalyzeResponse model not found")

try:
    from backend.api.models import UploadResponse
    models_found.append('UploadResponse')
    print("  ✅ UploadResponse model found")
except:
    print("  ⚠️ UploadResponse model not found")

test6_pass = len(models_found)
test6_total = 3

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 API ENDPOINTS TEST SUMMARY")
print("="*80)

tests = [
    ("App Initialization", test1_pass, 1),
    ("Route Registration", test2_pass, test2_total),
    ("Health Check", test3_pass, 1),
    ("Router Import", test4_pass, test4_total),
    ("CORS Configuration", test5_pass, 1),
    ("Request/Response Models", test6_pass, test6_total),
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
    print("\n✅ EXCELLENT: API endpoints working well")
elif overall_pct >= 60:
    print("\n⚠️ GOOD: API endpoints functional")
else:
    print("\n❌ CONCERN: API endpoints need work")

print("="*80)
