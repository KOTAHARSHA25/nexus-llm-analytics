# Analysis Service
# High-level orchestrator for analysis requests (Service Layer)

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add src to path if needed (for direct execution)
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.plugin_system import get_agent_registry
from backend.core.user_preferences import get_preferences_manager
from backend.core.result_interpreter import interpret_result, ResultInterpreter


class AnalysisService:
    """
    Service layer for handling analysis requests.
    Routes requests to appropriate agents via Plugin Registry.
    Supports intelligent model routing when enabled.
    Replaces legacy CrewManager and AnalysisExecutor.
    """
    
    def __init__(self):
        self.registry = get_agent_registry()
        self._intelligent_router = None
        # Ensure plugins are loaded
        # self.registry.discover_agents() # This is usually done on import or app startup
        logging.info("AnalysisService initialized")
    
    @property
    def intelligent_router(self):
        """Get intelligent router (lazy loaded)"""
        if self._intelligent_router is None:
            from backend.core.intelligent_router import get_intelligent_router
            self._intelligent_router = get_intelligent_router()
        return self._intelligent_router
    
    def _get_model_for_query(self, query: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Get optimal model for query using intelligent routing.
        Returns None if routing is disabled or unavailable.
        """
        try:
            prefs = get_preferences_manager().load_preferences()
            if not prefs.enable_intelligent_routing:
                return None
            
            # Build data info for routing
            data_info = {}
            if context.get('data_info'):
                data_info = context['data_info']
            elif context.get('rows') and context.get('columns'):
                data_info = {
                    'rows': context['rows'],
                    'columns': context['columns']
                }
            
            # Route the query
            decision = self.intelligent_router.route(query, data_info)
            logging.info(f"🧠 Intelligent routing: {decision.selected_tier.value} -> {decision.selected_model}")
            
            return decision.selected_model
            
        except Exception as e:
            logging.warning(f"Intelligent routing failed, using default: {e}")
            return None

    async def analyze(self, 
                     query: str, 
                     context: Dict[str, Any] = None, 
                     user_id: str = None) -> Dict[str, Any]:
        """
        Main entry point for analysis.
        
        Args:
            query: The user's question or command
            context: Additional context (files, conversation history, etc.)
            user_id: User identifier for personalization
            
        Returns:
            Analysis result dictionary
        """
        context = context or {}
        filename = context.get('filename')
        file_type = Path(filename).suffix if filename else None
        
        # Resolve filepath if not already provided
        if filename and not context.get('filepath'):
            from backend.utils.data_utils import DataPathResolver
            resolved_path = DataPathResolver.resolve_data_file(filename)
            if resolved_path:
                context['filepath'] = str(resolved_path)
        
        logging.info(f"AnalysisService received query: {query} (file: {filename})")
        
        # 0. Intelligent model routing (if enabled)
        selected_model = self._get_model_for_query(query, context)
        if selected_model:
            context['selected_model'] = selected_model
        
        # 1. Route the query to the best agent
        topic, confidence, agent = self.registry.route_query(query, file_type=file_type)
        
        if not agent:
            logging.warning("No specific agent found, falling back to DataAnalyst")
            agent = self.registry.get_agent("DataAnalyst")
            
        if not agent:
            return {
                "success": False,
                "error": "No capable agent available for this request",
                "type": "error"
            }
            
        logging.info(f"Routed to agent: {agent.metadata.name} (confidence: {confidence})")
        
        # 2. Execute analysis
        try:
            # We pass the query and full context to the agent
            # The agent is responsible for its own execution logic (optimization, etc.)
            result = agent.execute(query, **context)
            
            # 3. Generate human-readable interpretation if not provided
            interpretation = result.get("interpretation")
            if not interpretation or interpretation == "None" or len(str(interpretation).strip()) < 20:
                # Auto-generate interpretation from result
                raw_result = result.get("result", {})
                if isinstance(raw_result, dict):
                    interpretation = interpret_result(
                        result=raw_result,
                        query=query,
                        agent_name=agent.metadata.name
                    )
                elif isinstance(raw_result, str) and len(raw_result) > 50:
                    # Result is already a readable string
                    interpretation = raw_result
                else:
                    # Fallback interpretation
                    interpretation = f"Analysis completed by {agent.metadata.name}.\n\n{str(raw_result)}"
            
            # 4. Standardize response - always include interpretation
            response = {
                "success": result.get("success", False),
                "result": result.get("result"),
                "agent": agent.metadata.name,
                "metadata": result.get("metadata", {}),
                "interpretation": interpretation,  # Human-readable text for display
                "type": "analysis_result"
            }
            
            # Include routing info if used
            if selected_model:
                response["metadata"]["routed_model"] = selected_model
                
            if "error" in result:
                response["error"] = result["error"]
                
            return response
            
        except Exception as e:
            logging.error(f"Analysis execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent": agent.metadata.name,
                "type": "error"
            }

# Singleton
_service_instance = None

def get_analysis_service():
    global _service_instance
    if not _service_instance:
        _service_instance = AnalysisService()
    return _service_instance
