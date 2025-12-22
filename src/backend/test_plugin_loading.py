
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

try:
    from backend.core.plugin_system import get_agent_registry, AgentCapability
    
    print("Initializing Registry...")
    import logging
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    registry = get_agent_registry()
    print(f"Discovered Agents: {len(registry.registered_agents)}")
    
    expected_agents = [
        "DataAnalyst", "RagAgent", "Visualizer", "Reporter", "Reviewer",
        "FinancialAgent", "TimeSeriesAgent", "MLInsightsAgent", "SQLAgent", "StatisticalAgent"
    ]
    
    print("\n--- Checking Agents ---")
    all_found = True
    for name in expected_agents:
        if name in registry.registered_agents:
            print(f"✅ {name} loaded")
        else:
            print(f"❌ {name} NOT loaded")
            all_found = False
            
    if all_found:
        print("\n✅ All Plugin Agents successfully loaded!")
    else:
        print("\n❌ Some agents failed to load.")
        
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()
