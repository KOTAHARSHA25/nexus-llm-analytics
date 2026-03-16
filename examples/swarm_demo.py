import sys
import os
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from backend.core.plugin_system import AgentRegistry
from backend.core.swarm import SwarmContext
from backend.plugins.data_analyst_agent import DataAnalystAgent

def run_demo():
    print("=== Swarm Architecture Demo ===\n")

    # 1. Initialize Registry & Swarm
    # The registry automatically initializes the SwarmContext
    registry = AgentRegistry()
    registry.discover_agents()
    print(f"Registry initialized.")
    if hasattr(registry, 'agents'):
         print(f"Found {len(registry.agents)} agents.")
    
    # 2. Get Data Analyst
    analyst = registry.get_agent("DataAnalyst")
    if not analyst:
        print("DataAnalyst not found! Ensure plugins are discoverable.")
        return

    print(f"Agent found: {analyst.metadata.name} (v{analyst.metadata.version})")
    
    # 3. Verify Swarm Integration
    if hasattr(analyst, 'swarm_context') and analyst.swarm_context:
        print("✅ Swarm Context injected successfully into agent.")
    else:
        print("❌ Swarm Context NOT injected.")
        return

    # 4. Simulate a Reflective Execution
    print("\nExecuting query: 'Analyze the sales trend in data.csv'")
    
    # Create a dummy file for the test
    dummy_file = "swarm_test_data.csv"
    with open(dummy_file, "w") as f:
        f.write("date,sales\n2023-01-01,100\n2023-02-01,150\n2023-03-01,200")
        
    try:
        # Using reflective_execute
        # passing full filepath to be safe
        full_path = str(Path(dummy_file).absolute())
        result = analyst.reflective_execute("Analyze the sales trend", context={"filepath": full_path, "filename": dummy_file})
        
        print("\nExecution Result:")
        print(f"Success: {result.get('success')}")
        if result.get('success'):
            print(f"Result: {str(result.get('result'))[:200]}...")
        else:
             print(f"Error: {result.get('error')}")
        
        # 5. Check Swarm Blackboard
        print("\nChecking Swarm Blackboard for Insights:")
        # The agent should have published an insight if successful
        # We need to access the swarm context from the agent or registry
        swarm = analyst.swarm_context
        
        # Access message history directly since get_insights() helper is missing in minimal implementation
        if hasattr(swarm, '_message_history'):
             # Filter for insights
             insights = [m for m in swarm._message_history if m.type.value == "insight_found"]
             
             if insights:
                print(f"Found {len(insights)} insights:")
                for i, insight in enumerate(insights):
                    print(f"Insight {i+1} from {insight.source_agent}: {insight.content.get('summary', 'No summary')}")
             else:
                 print("No insights found in message history.")
        else:
            print("Could not access message history.")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Execution failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(dummy_file):
            os.remove(dummy_file)

if __name__ == "__main__":
    run_demo()
