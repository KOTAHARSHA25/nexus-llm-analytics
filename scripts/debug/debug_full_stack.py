
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

async def main():
    try:
        print("Importing AnalysisService...")
        from backend.services.analysis_service import get_analysis_service
        
        print("Getting service...")
        service = get_analysis_service()
        
        print("Running analyze...")
        # Simulate the exact call that fails
        result = await service.analyze(
            query="test",
            context={
                "text_data": None,
                "filename": None,
                "filenames": None,
                "column": None,
                "value": None,
                "analysis_id": "debug_test",
                "force_refresh": False,
                "review_level": None
            }
        )
        print("Result:", result)
        
    except Exception as e:
        print("CRITICAL CRASH:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
