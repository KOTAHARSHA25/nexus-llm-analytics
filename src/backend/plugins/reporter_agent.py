# Reporter Agent Plugin
# Handles report generation

import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability
from backend.agents.model_initializer import get_model_initializer

class ReporterAgent(BasePluginAgent):
    """
    Reporter Agent Plugin.
    Compiles analysis results into professional reports.
    """
    
    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="Reporter",
            version="2.0.0",
            description="Compiles comprehensive business reports",
            author="Nexus Team",
            capabilities=[AgentCapability.REPORTING],
            file_types=[],
            dependencies=[],
            priority=20
        )
    
    def initialize(self, **kwargs) -> bool:
        self.initializer = get_model_initializer()
        return True
    
    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs) -> float:
        query_lower = query.lower()
        # Only claim actual report generation, not summary statistics
        if "summary statistics" in query_lower or "summary stats" in query_lower:
            return 0.0  # Let DataAnalyst handle
        keywords = ["report", "writeup", "document results", "generate report"]
        if any(k in query_lower for k in keywords):
            return 0.8
        return 0.0

    def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Execute reporting task"""
        try:
            self.initializer.ensure_initialized()
            
            # Construct the prompt for the LLM
            # Incorporating the persona and task directly
            
            results_context = f"Analysis Results:\n{str(data)[:8000]}" # Increased context limit as direct LLM calls handle token limits better than recursive agent loops
            
            system_prompt = """You are a skilled business analyst and technical writer. 
You excel at transforming complex data analysis into clear, actionable reports for stakeholders at all levels.
Your goal is to create comprehensive and professional analysis reports."""

            user_prompt = f"""
Create a professional report based on the request: "{query}"

DATA/ANALYSIS TO REPORT ON:
{results_context}

REPORT STRUCTURE:
1. Executive Summary
   - Brief overview of the goal and main outcome.
2. Key Findings
   - Bullet points of the most important metrics or discoveries.
3. Detailed Analysis
   - In-depth explanation of the data evidence.
4. Recommendations
   - Actionable next steps based on the data.

FORMATTING INSTRUCTIONS:
- Use Professional Markdown.
- Use bolding for key terms.
- Keep sections clear and distinct.
- Do NOT include any 'Thought/Action' trace. Just write the report.
"""
            
            # Use the primary LLM directly
            # ModelInitializer provides the clean model name automatically
            response = self.initializer.llm_client.generate(
                prompt=user_prompt,
                system=system_prompt,
                model=self.initializer.primary_llm.model # Using clean model name (no ollama/ prefix needed)
            )
            
            # Extract text from response (handle dict or str return)
            report_text = response.get('response', str(response)) if isinstance(response, dict) else str(response)
            
            return {
                "success": True,
                "result": report_text,
                "metadata": {"agent": "Reporter", "mode": "direct_generation"}
            }
            
        except Exception as e:
            logging.error(f"Reporter execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
