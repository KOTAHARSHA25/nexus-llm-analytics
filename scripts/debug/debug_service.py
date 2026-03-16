
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import asyncio

async def test():
    try:
        from backend.core.plugin_system import get_agent_registry
        registry = get_agent_registry()
        
        query = "What is 25 * 4?"
        print(f"Routing query: {query}")
        
        # Test Routing
        topic, confidence, agent = registry.route_query(query)
        print(f"Routed to: {agent.metadata.name if agent else 'None'} ({confidence})")
        
        if not agent:
            print("Fallback to DataAnalyst")
            agent = registry.get_agent("DataAnalyst")
            
        print("Executing agent...")
        # Simulate AnalysisService execution
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: agent.execute(query))
        print("Result:", result)

    except Exception as e:
        print("CRASHED:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
