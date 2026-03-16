"""
Conflict Resolution & Meta-Routing Module
=========================================
Provides the `MetaRouter` class to resolve overlaps between specialized agents.
For example, deciding between StatisticalAgent, MLInsightsAgent, and TimeSeriesAgent
for queries that could technically be handled by multiple.

Author: Nexus Team
Since: v2.1.0
"""

import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MetaRouter:
    """
    Arbitrates between agents when multiple capabilities overlap.
    Uses an LLM to assign a confidence score to each candidate agent
    and picks the best fit.
    """

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
        self.agent_profiles = {
            "StatisticalAgent": "Hypothesis testing (t-tests, ANOVA), correlation matrices, regression analysis, p-values, descriptive statistics.",
            "MLInsightsAgent": "Pattern recognition, clustering (K-Means), anomaly detection, dimensionality reduction (PCA), predictive modeling (classification/regression).",
            "TimeSeriesAgent": "Temporal data analysis, forecasting (ARIMA, Prophet), trend detection, seasonality analysis, date-based resampling.",
            "DataAnalystAgent": "General data cleaning, pandas operations, simple aggregations, or fallback for unknown tasks.",
            "SQLAgent": "Database interactions, extracting data from SQL, schema analysis.",
            "RAGAgent": "Document search, policy lookup, unstructured text analysis.",
            "VisualizerAgent": "Creating charts, plots, and graphs.",
            "ReporterAgent": "Summarizing results into reports, PDF generation."
        }

    def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Decide which agent(s) should handle the query.
        Returns a list of selected agents with reasoning and confidence.
        """
        if not self.llm_client:
            logger.warning("MetaRouter: No LLM client available.")
            return [{"agent": "DataAnalystAgent", "confidence": 1.0, "reasoning": "Fallback due to no LLM"}]

        prompt = f"""You are a Meta-Decision Router. Choose the BEST agent(s) for this query.
Avoid generic 'DataAnalystAgent' if a specialist is more appropriate.

Query: "{query}"

Agent Capabilities:
{json.dumps(self.agent_profiles, indent=2)}

Task:
1. Analyze the query intent.
2. Assign a confidence score (0.0-1.0) to relevant agents.
3. Select the primary agent (highest score).
4. If the query strictly requires multiple specialists (e.g., "Forecast sales AND check for anomalies"), select top 2.

Return JSON ONLY:
[
  {{
    "agent": "AgentName",
    "confidence": 0.95,
    "reasoning": "Query asks for forecasting..."
  }}
]
"""
        try:
            # Use a fast model for routing
            model = getattr(self.llm_client, "primary_model", "phi3:mini")
            
            response = self.llm_client.generate(prompt, model=model)
            if isinstance(response, dict):
                response = response.get('response', '')
            
            # Clean generic markdown
            if "```" in response:
                response = response.split("```")[1]
                if response.strip().startswith("json"):
                    response = response.strip()[4:]
            
            selections = json.loads(response.strip())
            
            # Sort by confidence
            selections.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            logger.info(f"MetaRouter selections: {selections}")
            return selections

        except Exception as e:
            logger.error(f"MetaRouter failed: {e}")
            return [{"agent": "DataAnalystAgent", "confidence": 0.5, "reasoning": "Router Error"}]
