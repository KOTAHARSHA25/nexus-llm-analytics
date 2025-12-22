
import sys
import asyncio
from pathlib import Path

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

import logging
logging.basicConfig(level=logging.DEBUG)

async def test_service():
    print("Initializing AnalysisService...")
    try:
        from backend.services.analysis_service import get_analysis_service
        service = get_analysis_service()
        print("✅ Service Initialized")
        
        # Test 1: Route to DataAnalyst
        print("\nTest 1: Routing 'Analyze sales.csv'")
        topic, confidence, agent = service.registry.route_query("Analyze sales data", file_type=".csv")
        print(f"Routed to: {agent.metadata.name if agent else 'None'} (conf: {confidence})")
        
        if agent and agent.metadata.name == "DataAnalyst":
            print("✅ Correctly routed to DataAnalyst")
        else:
            print("❌ Routing failed")
            
        # Test 2: Route to Reviewer
        print("\nTest 2: Routing 'Review this result'")
        topic, confidence, agent = service.registry.route_query("Review this analysis result")
        print(f"Routed to: {agent.metadata.name if agent else 'None'} (conf: {confidence})")
        
        if agent and agent.metadata.name == "Reviewer":
            print("✅ Correctly routed to Reviewer")
        else:
            print("❌ Routing failed")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_service())
