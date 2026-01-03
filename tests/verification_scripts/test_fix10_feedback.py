"""
Test Fix 10: User Feedback Collection

Verifies that the feedback API:
1. Accepts and stores feedback
2. Returns proper response
3. Calculates statistics
4. Exports data
"""

import sys
import requests
import json
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000"
FEEDBACK_ENDPOINT = f"{BASE_URL}/api/feedback/"
STATS_ENDPOINT = f"{BASE_URL}/api/feedback/stats"
EXPORT_ENDPOINT = f"{BASE_URL}/api/feedback/export"


def test_submit_feedback():
    """Test submitting feedback"""
    print("\n📝 Testing Feedback Submission...")
    
    test_cases = [
        {
            "query": "What is the average sales?",
            "result": "The average sales is $1,234.56",
            "rating": 5,
            "thumbs_up": True,
            "comment": "Very accurate!"
        },
        {
            "query": "Show top 10 products",
            "result": "Here are the top 10 products...",
            "rating": 4,
            "thumbs_up": True
        },
        {
            "query": "Calculate profit margin",
            "result": "Profit margin is 23%",
            "rating": 2,
            "thumbs_up": False,
            "comment": "Result seems wrong"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    try:
        for i, feedback in enumerate(test_cases, 1):
            response = requests.post(FEEDBACK_ENDPOINT, json=feedback, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('feedback_id'):
                    print(f"  ✅ Feedback {i} submitted: {data['feedback_id']}")
                    passed += 1
                else:
                    print(f"  ❌ Feedback {i} failed: {data}")
            else:
                print(f"  ❌ Feedback {i} HTTP {response.status_code}: {response.text}")
        
        return passed, total
        
    except requests.exceptions.ConnectionError:
        print("  ⚠️ Backend not running")
        return 0, total
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0, total


def test_feedback_stats():
    """Test fetching feedback statistics"""
    print("\n📊 Testing Feedback Statistics...")
    
    try:
        response = requests.get(STATS_ENDPOINT, timeout=10)
        
        if response.status_code != 200:
            print(f"  ❌ HTTP {response.status_code}: {response.text}")
            return 0, 1
        
        stats = response.json()
        print(f"  Stats: {json.dumps(stats, indent=2)}")
        
        checks = []
        
        # Check required fields
        if 'total' in stats:
            print(f"  ✅ Total feedback: {stats['total']}")
            checks.append(True)
        else:
            print("  ❌ Missing 'total' field")
            checks.append(False)
        
        if 'avg_rating' in stats:
            if stats['avg_rating'] is not None:
                print(f"  ✅ Average rating: {stats['avg_rating']}/5")
            else:
                print("  ℹ️ No average rating (no data)")
            checks.append(True)
        else:
            print("  ❌ Missing 'avg_rating' field")
            checks.append(False)
        
        if 'thumbs_up_rate' in stats:
            if stats['thumbs_up_rate'] is not None:
                print(f"  ✅ Thumbs up rate: {stats['thumbs_up_rate'] * 100:.1f}%")
            else:
                print("  ℹ️ No thumbs up rate (no data)")
            checks.append(True)
        else:
            print("  ❌ Missing 'thumbs_up_rate' field")
            checks.append(False)
        
        passed = sum(checks)
        return passed, len(checks)
        
    except requests.exceptions.ConnectionError:
        print("  ⚠️ Backend not running")
        return 0, 3
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0, 3


def test_feedback_storage():
    """Test that feedback is actually stored to file"""
    print("\n💾 Testing Feedback Storage...")
    
    project_root = Path(__file__).parent
    feedback_file = project_root / 'data' / 'feedback' / 'user_feedback.jsonl'
    
    if not feedback_file.exists():
        print(f"  ⚠️ Feedback file not found: {feedback_file}")
        return 0, 1
    
    try:
        # Count lines in feedback file
        with open(feedback_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            valid_entries = 0
            
            for line in lines:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        if 'id' in entry and 'rating' in entry and 'query' in entry:
                            valid_entries += 1
                    except json.JSONDecodeError:
                        continue
        
        if valid_entries > 0:
            print(f"  ✅ Found {valid_entries} valid feedback entries in file")
            print(f"  ✅ File location: {feedback_file}")
            return 1, 1
        else:
            print("  ❌ No valid entries found in feedback file")
            return 0, 1
            
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return 0, 1


def test_feedback_export():
    """Test feedback export functionality"""
    print("\n📤 Testing Feedback Export...")
    
    try:
        # Test JSONL export
        response = requests.get(f"{EXPORT_ENDPOINT}?format=jsonl", timeout=10)
        
        if response.status_code != 200:
            print(f"  ❌ HTTP {response.status_code}: {response.text}")
            return 0, 1
        
        data = response.json()
        
        if 'count' in data and data['count'] > 0:
            print(f"  ✅ Export successful: {data['count']} entries")
            print(f"  ✅ Format: JSONL")
            return 1, 1
        elif 'count' in data and data['count'] == 0:
            print("  ℹ️ Export returned 0 entries (no feedback yet)")
            return 1, 1
        else:
            print(f"  ❌ Invalid export response: {data}")
            return 0, 1
            
    except requests.exceptions.ConnectionError:
        print("  ⚠️ Backend not running")
        return 0, 1
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0, 1


def test_rating_validation():
    """Test that rating validation works (1-5 range)"""
    print("\n✅ Testing Rating Validation...")
    
    invalid_feedback = {
        "query": "Test query",
        "result": "Test result",
        "rating": 10,  # Invalid - should be 1-5
        "thumbs_up": True
    }
    
    try:
        response = requests.post(FEEDBACK_ENDPOINT, json=invalid_feedback, timeout=10)
        
        # Should get 422 validation error
        if response.status_code == 422:
            print("  ✅ Rating validation working (rejected rating=10)")
            return 1, 1
        else:
            print(f"  ❌ Invalid rating accepted (HTTP {response.status_code})")
            return 0, 1
            
    except requests.exceptions.ConnectionError:
        print("  ⚠️ Backend not running")
        return 0, 1
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0, 1


def main():
    """Run all feedback tests"""
    print("=" * 60)
    print("FIX 10: USER FEEDBACK COLLECTION TEST")
    print("=" * 60)
    print("\nℹ️  Note: Backend server must be running at http://localhost:8000")
    print()
    
    results = []
    
    # Run all tests
    results.append(test_submit_feedback())
    results.append(test_feedback_stats())
    results.append(test_feedback_storage())
    results.append(test_feedback_export())
    results.append(test_rating_validation())
    
    # Calculate totals
    total_passed = sum(r[0] for r in results)
    total_tests = sum(r[1] for r in results)
    percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {total_passed}/{total_tests} tests passed ({percentage:.1f}%)")
    print("=" * 60)
    
    if percentage >= 90:
        print("\n🎯 Fix 10 VERIFIED - Feedback collection working!")
        print("   ✅ Feedback submission")
        print("   ✅ Statistics calculation")
        print("   ✅ File storage (JSONL)")
        print("   ✅ Data export")
        print("   ✅ Input validation")
        print("\n   Feedback flywheel enabled! 🔄")
    elif percentage >= 70:
        print("\n⚠️ Fix 10 PARTIAL - Most features working")
    elif percentage == 0 and total_tests > 0:
        print("\n⚠️ Backend server not running")
        print("   Start server: python -m uvicorn backend.main:app --reload")
    else:
        print("\n❌ Fix 10 NEEDS WORK")
    
    return percentage >= 90


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
