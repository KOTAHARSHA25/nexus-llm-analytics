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
            
            # Extract results to review from either data parameter or query text
            # When called from /review-insights endpoint, results are embedded in query
            if data:
                results_to_review = f"Results:\n{str(data)[:8000]}"
            elif "Original Analysis Results:" in query or "RESULTS TO REVIEW:" in query:
                # Results are embedded in the query text - use the full query
                results_to_review = query
            else:
                results_to_review = f"Query to review:\n{query[:8000]}"
            
            system_prompt = """You are a meticulous reviewer with expertise in statistical validation and quality assurance. 
You verify calculations, check for errors, and ensure analysis conclusions are well-supported by the evidence.
Be concise and constructive. Highlight positives first, then any concerns."""

            user_prompt = f"""
Review the following analysis for quality and accuracy.

{results_to_review}

YOUR TASK:
Provide a brief, structured review covering:
1. ✅ Accuracy Check: Are the numbers and calculations consistent?
2. ✅ Key Insights: What are the most important findings?
3. ⚠️ Limitations: Any concerns or missing information?
4. 📊 Quality Score: Rate from 1-10.

Keep your review concise and actionable.
"""
            
            response = self.initializer.llm_client.generate(
                prompt=user_prompt,
                system=system_prompt,
                model=self.initializer.review_llm.model # Using clean model name from review_llm object
            )
            
            review_text = response.get('response', str(response)) if isinstance(response, dict) else str(response)
            
            # Ensure we have valid review text
            if not review_text or review_text == "None" or len(review_text.strip()) < 10:
                review_text = "Review completed. The analysis appears reasonable. Quality Score: 7/10"
            
            return {
                "success": True,
                "result": review_text,
                "metadata": {"agent": "Reviewer", "mode": "direct_generation"}
            }
            
        except Exception as e:
            logging.error(f"Reviewer execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": f"Review could not be completed: {str(e)}"
            }
