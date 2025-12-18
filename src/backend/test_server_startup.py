#!/usr/bin/env python3
"""
Quick FastAPI Server Test
"""
from main import app
import uvicorn
import sys
import signal
import threading
import time
import requests

def test_server_startup():
    """Test if the FastAPI server can start and respond"""
    print("🧪 Testing FastAPI Server Startup...")
    
    server_ready = threading.Event()
    server_error = None
    
    def run_server():
        try:
            print("🚀 Starting FastAPI server on port 8001 (test mode)...")
            uvicorn.run(app, host="127.0.0.1", port=8001, log_level="error")
        except Exception as e:
            nonlocal server_error
            server_error = e
            print(f"❌ Server failed to start: {e}")
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait a bit for server to start
    time.sleep(3)
    
    # Test if server is responding
    try:
        print("🔍 Testing server response...")
        response = requests.get("http://127.0.0.1:8001/docs", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI server is running and responding!")
            print(f"📊 Status: {response.status_code}")
            print("🌐 API documentation accessible at /docs")
            return True
        else:
            print(f"⚠️ Server responding but with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server - may not have started properly")
        return False
    except requests.exceptions.Timeout:
        print("❌ Server request timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

if __name__ == "__main__":
    success = test_server_startup()
    if success:
        print("\n🎉 SERVER STARTUP TEST PASSED!")
        print("✅ FastAPI backend is functional and ready")
        sys.exit(0)
    else:
        print("\n⚠️ SERVER STARTUP TEST FAILED")
        print("❌ FastAPI backend has startup issues")
        sys.exit(1)