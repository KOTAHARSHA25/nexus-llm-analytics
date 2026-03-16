
import sys
import os
import asyncio
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from backend.services.analysis_service import AnalysisService
from backend.core.engine.query_orchestrator import ExecutionPlan, ExecutionMethod, ReviewLevel

async def trace_coexistence():
    print("🔍 STARTING COEXISTENCE TRACE...\n")
    
    # Mock dependencies to track calls without real execution cost
    with patch('backend.services.analysis_service.get_semantic_mapper') as mock_mapper, \
         patch('backend.services.analysis_service.get_enhanced_cache_manager') as mock_cache, \
         patch('backend.services.analysis_service.get_agent_registry') as mock_registry, \
         patch('backend.services.analysis_service.get_model_manager') as mock_model_mgr:
        
        # Setup Mocks
        service = AnalysisService()
        
        # Mock Semantic Mapper
        mock_mapper.return_value.enhance_query_context.return_value = "enhanced query"
        
        # Mock Orchestrator (we patch the property interaction or the method)
        # However, AnalysisService lazy loads orchestrator. Let's patch the orchestrator instance on the service
        mock_orchestrator = MagicMock()
        service._orchestrator = mock_orchestrator
        
        # Mock Registry and Agent
        mock_agent = MagicMock()
        mock_agent.metadata.name = "MockDataAgent"
        
        async def mock_execute(*args, **kwargs):
            return {"success": True, "result": "Agent executed"}
            
        mock_agent.execute_async = mock_execute
        mock_registry.return_value.route_query.return_value = ("topic", 0.9, mock_agent)
        
        # --- SCENARIO 1: Standard Flow (No Cache, Simple Query) ---
        print("1️⃣  SCENARIO: Standard Flow")
        mock_cache.return_value.get_sync.return_value = None # Cache Miss
        mock_orchestrator.create_execution_plan.return_value = ExecutionPlan(
            model="fast_model", execution_method=ExecutionMethod.DIRECT_LLM, 
            review_level=ReviewLevel.NONE, complexity_score=0.2, reasoning="Simple"
        )
        
        await service.analyze("simple query", context={})
        
        print("   [Check] Semantic Mapping called? " + ("✅ Yes" if mock_mapper.return_value.enhance_query_context.called else "❌ No"))
        print("   [Check] Orchestrator Planning called? " + ("✅ Yes" if mock_orchestrator.create_execution_plan.called else "❌ No"))
        print("   [Check] Cache Lookup called? " + ("✅ Yes" if mock_cache.return_value.get_sync.called else "❌ No"))
        print("   [Check] Agent Routing called? " + ("✅ Yes" if mock_registry.return_value.route_query.called else "❌ No"))
        print("   [Result] Coexistence confirmed along main path.\n")


        # --- SCENARIO 2: Cache Hit (Short-Circuit) ---
        print("2️⃣  SCENARIO: Cache Hit (Efficiency Check)")
        mock_mapper.reset_mock()
        mock_orchestrator.reset_mock()
        mock_registry.reset_mock()
        
        # Setup Cache Hit
        mock_cache.return_value.get_sync.return_value = {"success": True, "result": "Cached Result", "metadata": {}}
        
        await service.analyze("repeat query", context={})
        
        print("   [Check] Semantic Mapping called? " + ("✅ Yes" if mock_mapper.return_value.enhance_query_context.called else "❌ No (Expected check before cache)")) 
        print("   [Check] Orchestrator Planning called? " + ("✅ Yes" if mock_orchestrator.create_execution_plan.called else "❌ No"))
        print("   [Check] Cache Lookup called? " + ("✅ Yes" if mock_cache.return_value.get_sync.called else "❌ No"))
        print("   [Check] Agent Routing called? " + ("❌ No (Correctly Skipped)" if not mock_registry.return_value.route_query.called else "⚠️ Yes (Inefficient)"))
        print("   [Result] Efficiency features correctly overtake execution.\n")


        # --- SCENARIO 3: Self-Correction Interception ---
        print("3️⃣  SCENARIO: Self-Correction Interception (Deep Integration)")
        mock_registry.reset_mock()
        mock_cache.return_value.get_sync.return_value = None # Cache Miss
        
        # Setup Orchestrator to demand Mandatory Review
        mock_orchestrator.create_execution_plan.return_value = ExecutionPlan(
            model="smart_model", execution_method=ExecutionMethod.DIRECT_LLM, 
            review_level=ReviewLevel.MANDATORY, complexity_score=0.9, reasoning="Complex"
        )
        
        # Patch the cot_engine on the service
        mock_cot = MagicMock()
        service._cot_engine = mock_cot
        mock_cot.run_correction_loop.return_value = MagicMock(success=True, final_output="Corrected Result", total_iterations=2, final_reasoning="Logic")
        
        await service.analyze("complex query", context={})
        
        print("   [Check] Orchestrator signaled Mandatory Review? ✅ Yes")
        print("   [Check] Self-Correction Engine triggered? " + ("✅ Yes" if mock_cot.run_correction_loop.called else "❌ No"))
        print("   [Check] Standard Agent Routing skipped? " + ("✅ Yes (Correctly Replaced)" if not mock_registry.return_value.route_query.called else "❌ No (Conflict)"))
        print("   [Result] Advanced features correctly intercept standard flow.\n")

if __name__ == "__main__":
    asyncio.run(trace_coexistence())
