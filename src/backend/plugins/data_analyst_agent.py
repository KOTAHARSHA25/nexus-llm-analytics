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
        """
        Enhanced confidence scoring for DataAnalyst
        High confidence for simple data queries on structured files
        CONSERVATIVE: Should defer to specialized agents when they apply
        """
        confidence = 0.0
        query_lower = query.lower()
        
        # Normalize file_type (handle both "csv" and ".csv")
        if file_type and not file_type.startswith('.'):
            file_type = '.' + file_type
        
        # Base confidence for structured files - START LOW as fallback agent
        if file_type in [".csv", ".json", ".xlsx", ".xls"]:
            confidence = 0.3  # Low base - we're a fallback, not first choice
            
            # Defer to specialized agents based on analytical operations (not domain vocabulary)
            specialized_operations = {
                "statistical_tests": ["t-test", "correlation", "anova", "chi-square", "regression", 
                                     "hypothesis", "p-value", "significance", "statistical test"],
                "time_series_analysis": ["forecast", "arima", "predict", "trend", "seasonality", 
                                        "seasonal decomposition", "time series"],
                "ratio_calculations": ["ratio", "percentage", "proportion", "rate", "margin",
                                      "per", "relative to"],
                "ml_operations": ["clustering", "k-means", "anomaly detection", "pca", "machine learning",
                                 "dimensionality", "segment", "pattern detection"]
            }
            
            for operation, patterns in specialized_operations.items():
                if any(pattern in query_lower for pattern in patterns):
                    logging.debug(f"DataAnalyst: Deferring to specialized agent for {operation}")
                    return 0.1  # Return very low - let specialist handle it
            
            # HIGH PRIORITY: Summary statistics is DataAnalyst domain
            if "summary statistics" in query_lower or "summary stats" in query_lower:
                confidence += 0.5  # Very strong boost
            
            # Only boost if query is VERY simple and general
            very_simple_patterns = [
                "what is the average", "show me the top", "display summary",
                "what is the name", "get the data", "calculate total"
            ]
            if any(pattern in query_lower for pattern in very_simple_patterns):
                confidence += 0.4  # Boost for truly simple queries
            
            # Moderate boost for general analysis keywords (only if not specialized)
            simple_keywords = ["average", "total", "sum", "count", "maximum", "minimum"]
            keyword_matches = sum(1 for keyword in simple_keywords if keyword in query_lower)
            confidence += min(keyword_matches * 0.05, 0.15)
                
            # Reduce confidence if query explicitly mentions SQL/database
            sql_specific = ["sql", "query database", "database", "table join", "select from"]
            if any(keyword in query_lower for keyword in sql_specific):
                confidence *= 0.2  # Significantly reduce if SQL-specific
                
        return min(confidence, 0.85)  # Cap below specialists

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
                
                # Log metadata only - NEVER log data content
                total_rows = optimized_data.get('stats', {}).get('total_rows', 0)
                total_cols = optimized_data.get('stats', {}).get('total_columns', 0)
                logging.info(f"Data optimized: {filename} ({total_rows} rows, {total_cols} cols)")
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
        # Intelligent context based on data characteristics (not query keywords)
        # Data optimizer already analyzed the data - respect its decisions
        hint = ""
        if 'PRE-CALCULATED STATISTICS' in data_info:
            # Large dataset detected by optimizer - add minimal guidance
            hint = "Note: Use the pre-calculated statistics provided below.\n\n"
        
        # Clean, direct prompt - query unchanged
        prompt = f"""{hint}Question: {query}

Data from: {filename}

{data_info}

Answer:"""
        
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
        
        # Use the review model from initializer instead of hardcoded model
        critic_model = self.initializer.review_llm.model if hasattr(self.initializer, 'review_llm') else selected_model
        
        result = cot_engine.run_correction_loop(
            query=query,
            data_context=data_context,
            generator_model=selected_model,
            critic_model=critic_model
        )
        return result.final_output, {"cot_iterations": result.total_iterations}
