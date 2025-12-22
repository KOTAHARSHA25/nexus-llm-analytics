"""
Dynamic Analysis Planner
========================
Generates domain-specific analysis plans based on data content.
This allows the system to handle any data domain (Biology, Sports, Retail, etc.) 
without hardcoded agents.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

from backend.agents.model_initializer import get_model_initializer

@dataclass
class AnalysisStep:
    """A single step in the dynamic analysis plan"""
    step_id: int
    description: str
    tool: str  # e.g., "python_pandas", "visualization", "rag"
    reasoning: str

@dataclass
class AnalysisPlan:
    """Complete analysis plan"""
    domain: str
    summary: str
    steps: List[AnalysisStep]
    confidence: float

class DynamicPlanner:
    """
    Planner that uses LLM to inspect data structure and generic meaningful analysis steps.
    """
    
    def __init__(self):
        self._initializer = get_model_initializer()
        
    @property
    def llm(self):
        return self._initializer.llm_client

    def create_plan(self, query: str, data_preview: str, model: str = None) -> AnalysisPlan:
        """
        Generate a dynamic analysis plan.
        
        Args:
            query: User's intent
            data_preview: String summary of the data (from DataOptimizer)
            model: Model to use for planning
            
        Returns:
            AnalysisPlan object
        """
        try:
            prompt = self._build_planning_prompt(query, data_preview)
            
            # Use smart model for planning
            planning_model = model or self._initializer.primary_llm
            
            # Extract string name if it's an object (OllamaLLM)
            if hasattr(planning_model, 'model'):
                planning_model = planning_model.model
            
            response = self.llm.generate(
                prompt=prompt,
                model=str(planning_model),
                adaptive_timeout=True
            )
            
            if not response or not response.get('success'):
                raise ValueError("Failed to generate plan from LLM")
                
            return self._parse_plan(response.get('response', ''))
            
        except Exception as e:
            logging.error(f"Dynamic planning failed: {e}")
            # Fallback plan
            return AnalysisPlan(
                domain="General",
                summary="Fallback analysis due to planning error",
                steps=[
                    AnalysisStep(1, f"Analyze the data to answer: {query}", "generic_analysis", "Fallback")
                ],
                confidence=0.5
            )

    def _build_planning_prompt(self, query: str, data_preview: str) -> str:
        return f"""
You are an Expert Data Science Planner. 
Your goal is to create a step-by-step analysis plan to answer the user's query based on the provided dataset.

USER QUERY: "{query}"

DATASET PREVIEW:
{data_preview}

INSTRUCTIONS:
1. Identify the DOMAIN of the data (e.g., Finance, Genomics, Sports, Web Logs).
2. Formulate a logical 2-4 step plan to answer the query.
3. If the query is vague (e.g., "analyze this"), propose 3 interesting angles found in the data.
4. Output MUST be valid JSON in the following format:

{{
  "domain": "Detected Domain",
  "summary": "Brief summary of what we will do",
  "confidence": 0.9,
  "steps": [
    {{
      "id": 1,
      "description": "Calculate correlation between X and Y...",
      "tool": "python_pandas",
      "reasoning": "To see if there is a relationship..."
    }},
    {{
      "id": 2,
      "description": "Visualize the trend of Z over time...",
      "tool": "visualization",
      "reasoning": "To spot anomalies..."
    }}
  ]
}}

Provide ONLY the JSON output.
"""

    def _parse_plan(self, llm_output: str) -> AnalysisPlan:
        """Parse LLM JSON output into AnalysisPlan object"""
        try:
            # Clean generic markdown if present
            cleaned = llm_output.replace("```json", "").replace("```", "").strip()
            plan_dict = json.loads(cleaned)
            
            steps = [
                AnalysisStep(
                    step_id=s.get('id', i),
                    description=s.get('description', 'Unknown step'),
                    tool=s.get('tool', 'python_pandas'),
                    reasoning=s.get('reasoning', '')
                )
                for i, s in enumerate(plan_dict.get('steps', []), 1)
            ]
            
            return AnalysisPlan(
                domain=plan_dict.get('domain', 'General'),
                summary=plan_dict.get('summary', 'Analysis Plan'),
                steps=steps,
                confidence=float(plan_dict.get('confidence', 0.5))
            )
            
        except json.JSONDecodeError:
            logging.warning("Failed to parse planning JSON. Using raw output as single step.")
            return AnalysisPlan(
                domain="General",
                summary="Direct Analysis",
                steps=[AnalysisStep(1, llm_output[:200], "general_analysis", "Parsing failed")],
                confidence=0.1
            )
             
def get_dynamic_planner():
    return DynamicPlanner()
