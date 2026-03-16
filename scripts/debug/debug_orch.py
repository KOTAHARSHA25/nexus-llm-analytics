
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

try:
    from backend.core.engine.query_orchestrator import QueryOrchestrator
    
    print("Initializing Orchestrator...")
    orch = QueryOrchestrator()
    
    print("Running create_execution_plan...")
    plan = orch.create_execution_plan(
        query="What is 25 * 4?",
        data=None,
        context={}
    )
    print("Plan created successfully:")
    print(plan)

except Exception as e:
    print("CRASHED:")
    print(e)
    import traceback
    traceback.print_exc()
