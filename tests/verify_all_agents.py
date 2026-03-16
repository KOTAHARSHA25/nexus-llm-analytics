
import sys
import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# --- Setup Paths ---
# Add the project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# --- Imports ---
from dotenv import load_dotenv
from backend.core.plugin_system import AgentRegistry, AgentCapability
from backend.agents.model_manager import get_model_manager

# --- Configuration ---
DATA_DIR = PROJECT_ROOT / "data" / "samples"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configure Root Logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# File Handler
file_handler = logging.FileHandler("verification.log", mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
root_logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
root_logger.addHandler(console_handler)

logger = logging.getLogger("SystemVerifier")
# backend logger configuration is now redundant as root logger covers it

# --- Query Definitions ---
QUERY_TIERS = {
    "Simple": [
        "What are the keys in this JSON file?",
        "Count the number of records.",
        "Show me a sample record."
    ],
    "Intermediate": [
        "Summarize the data in this file.",
        "What are the main fields and their types?",
        "Are there any missing values?"
    ],
    "God Level": [
        "Analyze the data for potential anomalies and outliers.",
        "Perform a deep statistical analysis of the numerical fields.",
        "Identification of patterns or trends in the data."
    ]
}

# --- Verification Logic ---
async def verify_system():
    logger.info("🚀 Starting Comprehensive System Verification...")
    
    # 1. Load Environment
    load_dotenv()
    
    # 2. Initialize Model Manager (for Ollama connection)
    logger.info("Initializing Model Manager...")
    model_manager = get_model_manager()
    try:
        model_manager.ensure_initialized()
        logger.info("✅ Model Manager Initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Model Manager: {e}")
        return

    # 3. Discover Agents
    logger.info("Discovering Agents...")
    plugins_dir = PROJECT_ROOT / "src" / "backend" / "plugins"
    logger.info(f"Plugins Directory: {plugins_dir}")
    if plugins_dir.exists():
        logger.info(f"Directory exists. Contents: {[f.name for f in plugins_dir.glob('*.py')]}")
    else:
        logger.error(f"Plugins directory does not exist!")

    registry = AgentRegistry(plugins_directory=plugins_dir)
    count = registry.discover_agents()
    logger.info(f"✅ Discovered {count} agents.")
    
    # 4. Scan for JSON Files
    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        logger.warning(f"⚠️ No JSON files found in {DATA_DIR}")
        return
        
    logger.info(f"Found {len(json_files)} JSON files to test.")

    # 5. Execution Loop
    results = []
    
    for json_file in json_files:
        logger.info(f"\n--- Processing File: {json_file.name} ---")
        
        # Determine applicable agents
        # For now, we mainly target DataAnalyst, but we check capabilities
        # In a real scenario, we might want to test specific specialist agents if file matches
        
        # Iterate through query tiers
        for tier, queries in QUERY_TIERS.items():
            logger.info(f"\n  [Tier: {tier}]")
            
            for query in queries:
                logger.info(f"    ❓ Query: {query}")
                
                # Find best agent
                best_capability, score, agent = registry.route_query(query, file_type=".json")
                
                if not agent:
                    logger.warning(f"      ⚠️ No suitable agent found for query: {query}")
                    results.append({
                        "file": json_file.name,
                        "tier": tier,
                        "query": query,
                        "status": "Skipped (No Agent)",
                        "agent": "None"
                    })
                    continue
                
                logger.info(f"      🤖 Selected Agent: {agent.metadata.name} (Score: {score:.2f})")
                
                # Execute
                try:
                    # Provide filepath in context
                    context = {"filepath": str(json_file), "filename": json_file.name}
                    
                    # Run async execution if available, else sync
                    if hasattr(agent, "execute_async"):
                        result = await agent.execute_async(query, **context)
                    else:
                        result = agent.execute(query, **context)
                    
                    success = result.get("success", False)
                    status_icon = "✅" if success else "❌"
                    logger.info(f"      {status_icon} Result: {success}")
                    
                    if not success:
                         logger.error(f"        Error: {result.get('error')}")
                    
                    results.append({
                        "file": json_file.name,
                        "tier": tier,
                        "query": query,
                        "status": "Success" if success else "Failed",
                        "agent": agent.metadata.name,
                        "error": result.get("error") if not success else None
                    })

                except Exception as e:
                    logger.error(f"      ❌ Execution Exception: {e}")
                    results.append({
                        "file": json_file.name,
                        "tier": tier,
                        "query": query,
                        "status": "Exception",
                        "agent": agent.metadata.name,
                        "error": str(e)
                    })

    # 6. Summary Report
    logger.info("\n" + "="*50)
    logger.info("📊 VERIFICATION SUMMARY")
    logger.info("="*50)
    
    success_count = sum(1 for r in results if r["status"] == "Success")
    total_count = len(results)
    
    logger.info(f"Total Tests: {total_count}")
    logger.info(f"Passed:      {success_count}")
    logger.info(f"Failed:      {total_count - success_count}")
    
    if total_count > 0:
        success_rate = (success_count / total_count) * 100
        logger.info(f"Success Rate: {success_rate:.2f}%")
    
    logger.info("="*50)
    
    # Save results to file
    output_file = PROJECT_ROOT / "verification_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to {output_file}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(verify_system())
