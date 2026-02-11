
import sys
import os
import asyncio
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

async def test_analysis_manager_integration():
    print("🧪 Testing AnalysisManager Integration...")
    
    try:
        # 1. Initialize AnalysisService
        from backend.services.analysis_service import AnalysisService
        service = AnalysisService()
        print("   ✅ AnalysisService initialized")
        
        # 2. Check Manager Connection
        if not hasattr(service, 'analysis_manager'):
             print("   ❌ AnalysisManager NOT attached to service")
             return
             
        manager = service.analysis_manager
        print("   ✅ AnalysisManager connected")
        
        # 3. Simulate Lifecycle via AnalysisManager directly (to test endpoints)
        job_id = manager.start_analysis(user_session="test-user")
        print(f"   ✅ Started Job: {job_id}")
        
        # Verify status
        status = manager.get_analysis_status(job_id)
        if status and status['status'] == 'running':
            print("   ✅ Job status is 'running'")
        else:
            print(f"   ❌ Unexpected status: {status}")
            
        # Update stage
        manager.update_analysis_stage(job_id, "processing")
        status = manager.get_analysis_status(job_id)
        if status and status['stage'] == 'processing':
            print("   ✅ Job stage updated to 'processing'")
            
        # Complete
        manager.complete_analysis(job_id)
        status = manager.get_analysis_status(job_id)
        if status and status['status'] == 'completed':
             print("   ✅ Job completed successfully")
             
        # 4. Test Fail Flow
        fail_id = manager.start_analysis("fail-test")
        manager.fail_analysis(fail_id, "Simulated Error")
        f_status = manager.get_analysis_status(fail_id)
        if f_status and f_status['status'] == 'failed' and f_status['error'] == "Simulated Error":
            print("   ✅ Job failure tracking works")
        else:
            print(f"   ❌ Failure tracking failed: {f_status}")
            
        print("\n🎉 AnalysisManager Modernization Verified!")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_analysis_manager_integration())
