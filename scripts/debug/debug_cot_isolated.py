import logging
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# Setup basic logging
logging.basicConfig(level=logging.DEBUG)

def test_cot_engine():
    try:
        from backend.core.engine.self_correction_engine import SelfCorrectionEngine
        from backend.core.llm_client import LLMClient
        from backend.core.config import get_settings
        
        # Load environment
        from dotenv import load_dotenv
        load_dotenv()
        
        # Load config manually
        try:
            with open("config/cot_review_config.json", "r") as f:
                full_config = json.load(f)
                config = full_config["cot_review"]
        except Exception as e:
            print(f"Failed to load config: {e}")
            return

        print("Initializing LLM Client...")
        llm_client = LLMClient()
        print("LLM Client Initialized.")

        print("Initializing Engine...")
        engine = SelfCorrectionEngine(config, llm_client)
        print("Engine Initialized.")

        data_context = {
            "rows": 100,
            "columns": ["id", "val"],
            "data_types": {"id": "int", "val": "float"},
            "stats_summary": "Summary..."
        }

        print("Running Loop (Real LLM)...")
        # Use a simple prompt that guarantees CoT structure to avoid infinite loops if model is dumb
        result = engine.run_correction_loop(
            query="Write a Python script to reverse a string.",
            data_context=data_context,
            generator_model="tinyllama:latest", 
            critic_model="tinyllama:latest"
        )
        print("Loop Finished.")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.total_iterations}")
        print(f"Output: {result.final_output[:100]}...")

    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cot_engine()
