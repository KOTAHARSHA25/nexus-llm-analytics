# Reviewer Agent Plugin
# Handles quality assurance and validation

import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability
from backend.agents.model_initializer import get_model_initializer

class ReviewerAgent(BasePluginAgent):
    """
    Reviewer Agent Plugin.
    Validates analysis results and provides feedback.
    """
    
    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="Reviewer",
            version="2.0.0",
            description="Reviews analysis for quality and accuracy",
            author="Nexus Team",
            capabilities=[AgentCapability.DATA_ANALYSIS],
            file_types=[],
            dependencies=[],
            priority=20
        )
    
    def initialize(self, **kwargs) -> bool:
        self.initializer = get_model_initializer()
        return True
    
    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs) -> float:
        keywords = ["review", "validate", "check", "verify", "audit"]
        if any(k in query.lower() for k in keywords):
            return 0.8
        return 0.0

    def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Execute review task"""
        try:
            self.initializer.ensure_initialized()
            
            # Using review_llm which is usually smaller/faster (e.g. phi3)
            # but getting the clean model name from the initialized object
            
            results_to_review = f"Results:\n{str(data)[:8000]}"
            
            system_prompt = """You are a meticulous reviewer with expertise in statistical validation and quality assurance. 
You verify calculations, check for errors, and ensure analysis conclusions are well-supported by the evidence."""

            user_prompt = f"""
Review the following analysis results for quality and accuracy.

Original Query: "{query}"

RESULTS TO REVIEW:
{results_to_review}

YOUR TASK:
Provide a structured review covering:
1. Accuracy Check: Are the numbers consistent?
2. Logical Validity: Do the conclusions follow from the data?
3. Missing Information: What else should have been analyzed?
4. Quality Score: Rate from 1-10.

If the analysis is good, keep the review brief and positive.
If there are errors, explain them clearly.
"""
            
            response = self.initializer.llm_client.generate(
                prompt=user_prompt,
                system=system_prompt,
                model=self.initializer.review_llm.model # Using clean model name from review_llm object
            )
            
            review_text = response.get('response', str(response)) if isinstance(response, dict) else str(response)
            
            return {
                "success": True,
                "result": review_text,
                "metadata": {"agent": "Reviewer", "mode": "direct_generation"}
            }
            
        except Exception as e:
            logging.error(f"Reviewer execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
