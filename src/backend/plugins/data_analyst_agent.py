# Data Analyst Agent Plugin
# Standardizes the core Data Analyst as a system plugin
# incorporating advanced DataOptimizer and CoT logic.

import sys
import logging
import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability
from backend.agents.model_initializer import get_model_initializer
from backend.core.dynamic_planner import get_dynamic_planner

class DataAnalystAgent(BasePluginAgent):
    """
    Advanced Data Analyst Agent (Plugin Version)
    Handles general-purpose structured data analysis using DataOptimizer and CoT.
    """
    
    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="DataAnalyst",
            version="2.0.0",
            description="General-purpose data analyst for structured data (CSV, JSON, Excel) with optimization and self-correction",
            author="Nexus Team",
            capabilities=[AgentCapability.DATA_ANALYSIS],
            file_types=[".csv", ".json", ".xlsx", ".xls"],
            dependencies=["pandas"],
            priority=10
        )
    
    def initialize(self, **kwargs) -> bool:
        self.initializer = get_model_initializer()
        self._cot_engine = None
        self._cot_config = None
        return True
    
    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs) -> float:
        # High confidence for structured files if no specialized agent picks it up
        if file_type in [".csv", ".json", ".xlsx", ".xls"]:
            return 0.9  # High priority for structured data
        return 0.1

    def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Execute analysis using DataOptimizer and LLM"""
        filename = kwargs.get('filename')
        filepath = kwargs.get('filepath')
        
        # If no filepath but filename exists, resolve it
        if filename and not filepath:
             filepath = self._resolve_filepath(filename)
             
        if not filepath and not data:
            return {"success": False, "error": "No file provided"}
            
        try:
            self.initializer.ensure_initialized()
            
            # 1. Optimize Data
            data_info, optimized_data, available_columns = self._optimize_data(filepath, filename)
            
            # 2. Select Model & Complexity
            selected_model, routing_decision = self._select_model(
                query, optimized_data, available_columns, filepath, kwargs.get('force_model')
            )
            
            complexity_score = self._get_complexity_score(
                query, optimized_data, available_columns, filepath, routing_decision
            )
            
            # 3. Dynamic Planning
            try:
                planner = get_dynamic_planner()
                analysis_plan = planner.create_plan(query, data_info)
            except Exception:
                analysis_plan = None
                
            # 4. Decide on CoT
            cot_config = self._load_cot_config()
            use_cot = (
                cot_config.get('enabled', False) and
                cot_config.get('auto_enable_on_routing', True) and
                complexity_score >= cot_config.get('complexity_threshold', 0.4)
            )
            
            # 5. Execute
            if use_cot:
                result, metadata = self._execute_with_cot(
                    query, data_info, optimized_data, available_columns,
                    selected_model, complexity_score, filepath, analysis_plan
                )
            else:
                result = self._execute_direct(
                    query, data_info, filename, selected_model, analysis_plan
                )
                metadata = {}
                
            metadata.update({
                "agent": "DataAnalyst",
                "model": selected_model,
                "complexity": complexity_score,
                "method": "CoT" if use_cot else "Direct"
            })
            
            return {
                "success": True,
                "result": result,
                "metadata": metadata
            }
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logging.error(f"DataAnalyst execution failed: {e}\n{tb}")
            return {"success": False, "error": str(e), "traceback": tb}

    # --- Helper Methods ported from AnalysisExecutor ---

    def _load_cot_config(self) -> Dict[str, Any]:
        if self._cot_config is None:
            config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'cot_review_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self._cot_config = json.load(f)
            else:
                self._cot_config = {'enabled': False}
        return self._cot_config

    def _ensure_cot_engine(self):
        config = self._load_cot_config()
        if config.get('enabled') and self._cot_engine is None:
            try:
                from backend.core.self_correction_engine import SelfCorrectionEngine
                self._cot_engine = SelfCorrectionEngine(config)
            except ImportError:
                pass
        return self._cot_engine

    def _resolve_filepath(self, filename: str) -> Optional[str]:
        from backend.utils.data_utils import DataPathResolver
        
        # Use centralized path resolver which handles various locations (uploads, samples, etc.)
        resolved_path = DataPathResolver.resolve_data_file(filename)
        
        if resolved_path:
            return str(resolved_path)
            
        return None

    def _optimize_data(self, filepath: str, filename: str) -> tuple:
        try:
            from backend.utils.data_optimizer import DataOptimizer 
            
            optimizer = DataOptimizer(max_rows=100, max_chars=8000)
            optimized_data = optimizer.optimize_for_llm(filepath)
            
            if optimized_data['is_optimized']:
                data_info = optimized_data['preview']
                available_columns = optimized_data.get('stats', {}).get('columns', [])
            else:
                data_info = optimized_data['preview']
                available_columns = []
            
            return data_info, optimized_data, available_columns
        except Exception as e:
            logging.warning(f"Optimization failed: {e}")
            return str(e), {'is_optimized': False}, []

    def _select_model(self, query, optimized_data, available_columns, filepath, force_model=None):
        if force_model: return force_model, None
        return self.initializer.primary_llm.model, None

    def _get_complexity_score(self, query, optimized_data, available_columns, filepath, routing_decision):
        if routing_decision: return routing_decision.complexity_score
        return 0.5

    def _execute_direct(self, query, data_info, filename, selected_model, analysis_plan=None):
        strategy_context = ""
        if analysis_plan:
            steps_text = "\n".join([f"{s.step_id}. {s.description}" for s in analysis_plan.steps])
            strategy_context = f"STRATEGY:\n{steps_text}\n"
            
        prompt = f"""
        Analyze the following dataset stats/preview strictly.
        QUERY: {query}
        FILE: {filename}
        
        {strategy_context}
        
        DATA CONTEXT:
        {data_info}
        
        STRICT ANALYSIS RULES & INTEGRITY CHECKS:
        1. 🧪 DATA QUALITY CHECK (Mandatory):
           - Report total rows, missing values (if any), and duplicate rows (if any).
           - If 'duplicate_rows' is > 0 in stats, explicitly flag it as a data quality issue.
           - If 'null_count' is high in relevant columns, warn the user.
           
        2. 📊 INSIGHT GENERATION:
           - Answer the query using ONLY the provided data context (PRE-CALCULATED STATISTICS).
           - Do NOT calculate totals/averages yourself from the sample rows – use the "OVERALL COLUMN STATISTICS" section.
           - Identify key insights based on meaningful fields, not arbitrary limits.
           - If multiple interpretations are possible, state them clearly.

        3. 🚫 STRICT PROHIBITIONS:
           - Do NOT analyze only the 3-row sample as if it were the whole dataset.
           - Do NOT introduce fake metrics or assume values not present.
           - Do NOT generalize from insufficient data.
           - If the answer is not in the data, state: "The dataset provided does not contain sufficient information to answer this."

        FORMAT:
        - **Data Quality Status**: [Safe/Warning/Critical]
        - **Direct Answer**: [Clear, number-backed answer]
        - **Key Insights**: [Bullet points with evidence]
        - **Validation**: [Why this conclusion is statistically valid based on the logic]
        
        Response:
        """
        response = self.initializer.llm_client.generate(prompt, model=selected_model)
        if isinstance(response, dict): return response.get('response', str(response))
        return str(response)

    def _execute_with_cot(self, query, data_info, optimized_data, available_columns, selected_model, complexity, filepath, plan):
        cot_engine = self._ensure_cot_engine()
        if not cot_engine:
            return self._execute_direct(query, data_info, os.path.basename(filepath), selected_model, plan), {}
            
        data_context = {
            'stats_summary': data_info,
            'columns': available_columns
        }
        
        result = cot_engine.run_correction_loop(
            query=query,
            data_context=data_context,
            generator_model=selected_model,
            critic_model="gpt-4" # or similar
        )
        return result.final_output, {"cot_iterations": result.total_iterations}
