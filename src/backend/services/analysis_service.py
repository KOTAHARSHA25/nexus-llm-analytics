# Analysis Service
# High-level orchestrator for analysis requests (Service Layer)

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add src to path if needed (for direct execution)
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.plugin_system import get_agent_registry


class AnalysisService:
    """
    Service layer for handling analysis requests.
    Routes requests to appropriate agents via Plugin Registry.
    Replaces legacy CrewManager and AnalysisExecutor.
    """
    
    def __init__(self):
        self.registry = get_agent_registry()
        # Ensure plugins are loaded
        # self.registry.discover_agents() # This is usually done on import or app startup
        logging.info("AnalysisService initialized")

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
        
        logging.info(f"AnalysisService received query: {query} (file: {filename})")
        
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
            
            # 3. Standardize response
            response = {
                "success": result.get("success", False),
                "result": result.get("result"),
                "agent": agent.metadata.name,
                "metadata": result.get("metadata", {}),
                "type": "analysis_result"
            }
            
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
