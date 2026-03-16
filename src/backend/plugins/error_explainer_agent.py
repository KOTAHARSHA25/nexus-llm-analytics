"""
Error Explainer Agent Plugin
============================
Specialized agent for translating technical errors into user-friendly
explanations and troubleshooting steps.

Author: Nexus Team
Since: v2.1.0
"""

from typing import Dict, Any, Optional
import logging
from backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability

class ErrorExplainerAgent(BasePluginAgent):
    """
    Translates stack traces and error messages into actionable advice.
    """

    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="ErrorExplainer",
            version="1.0.0",
            description="Explains errors in plain English",
            author="Nexus Team",
            capabilities=[AgentCapability.REPORTING],
            priority=50
        )

    def initialize(self, **kwargs: Any) -> bool:
        self.registry = kwargs.get("registry")
        from backend.agents.model_manager import get_model_manager
        self.initializer = get_model_manager()
        return True

    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs: Any) -> float:
        """
        Only handles queries explicitly routed for error explanation.
        """
        if "explain_error" in kwargs.get("mode", ""):
            return 1.0
        if "error" in query.lower() and "explain" in query.lower():
            return 0.8
        return 0.0

    def execute(self, query: str, data: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Analyze the error and provide a user-friendly explanation.
        """
        try:
            self.initializer.ensure_initialized()
            
            error_msg = query
            stack_trace = kwargs.get("stack_trace", "")
            context_info = kwargs.get("context", "")
            
            prompt = f"""You are a helpful Technical Support Agent.
Translate this technical error into a friendly, actionable explanation for a non-technical user.

Error Message: "{error_msg}"

Stack Trace / Context:
{stack_trace}
{context_info}

Format your response as:
**What went wrong:** [One sentence summary]
**Why it happened:** [Brief non-technical reason]
**How to fix it:** [Bulleted actionable steps]

Keep it concise. Avoid jargon where possible.
"""
            
            system_prompt = "You are an expert at debugging and explaining software errors."
            
            response = self.initializer.llm_client.generate(
                prompt=prompt,
                system=system_prompt,
                model=self.initializer.primary_llm.model
            )
            
            explanation = response.get('response', str(response)) if isinstance(response, dict) else str(response)
            
            return {
                "success": True,
                "result": explanation,
                "metadata": {"agent": "ErrorExplainer", "original_error": error_msg}
            }

        except Exception as e:
            logging.error(f"ErrorExplainer failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": "Could not explain the error due to an internal failure."
            }

    def reflective_execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Error explainer is simple enough not to need complex reflection loop yet
        return self.execute(query, **(context or {}))
