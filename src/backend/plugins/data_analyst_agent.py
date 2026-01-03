# Data Analyst Agent Plugin (Phase 1 Enhanced)
# Standardizes the core Data Analyst as a system plugin
# incorporating advanced DataOptimizer, QueryOrchestrator, and CoT logic.
# Phase 1: Smart Fallback integration ensures process never stops

import sys
import logging
import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability
from backend.agents.model_manager import get_model_manager
from backend.core.dynamic_planner import get_dynamic_planner
from backend.core.engine.query_orchestrator import QueryOrchestrator, ExecutionMethod, ReviewLevel

# Phase 1 imports
try:
    from backend.core.phase1_integration import (
        get_phase1_coordinator,
        resilient_llm_call,
        GracefulDegradation
    )
    from backend.infra.circuit_breaker import get_circuit_breaker, CircuitState
    PHASE1_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Phase 1 components not available: {e}")
    PHASE1_AVAILABLE = False

class DataAnalystAgent(BasePluginAgent):
    """
    Advanced Data Analyst Agent (Plugin Version, Phase 1 Enhanced)
    
    Handles general-purpose structured data analysis using DataOptimizer and CoT.
    
    Phase 1 Enhancements:
    - Smart fallback: Process never stops completely
    - Circuit breaker: Resilient LLM calls  
    - RAM-aware: Adapts to system resources
    - Dynamic model discovery: Works with any available models
    """
    
    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="DataAnalyst",
            version="2.1.0",  # Phase 1 version bump
            description="General-purpose data analyst for structured data with smart fallback and self-correction",
            author="Nexus Team",
            capabilities=[AgentCapability.DATA_ANALYSIS],
            file_types=[".csv", ".json", ".xlsx", ".xls"],
            dependencies=["pandas"],
            priority=10
        )
    
    def initialize(self, **kwargs) -> bool:
        self.initializer = get_model_manager()
        self._cot_engine = None
        self._cot_config = None
        self._orchestrator = None  # Lazy loaded QueryOrchestrator
        self._circuit_name = "data_analyst"  # Phase 1: Circuit breaker name
        return True
    
    def _get_orchestrator(self) -> QueryOrchestrator:
        """
        Lazy load the QueryOrchestrator for unified decision making.
        
        STREAMLINED: No config needed - orchestrator loads from cot_review_config.json automatically
        """
        if self._orchestrator is None:
            self._orchestrator = QueryOrchestrator()  # Auto-loads config
        return self._orchestrator
    
    def _get_planner_config(self) -> Dict[str, Any]:
        """
        Load DynamicPlanner configuration from cot_review_config.json.
        Returns planner settings with safe defaults.
        """
        try:
            import json
            from pathlib import Path
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "cot_review_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('dynamic_planner', {'enabled': True})
            else:
                logging.debug("cot_review_config.json not found, using planner defaults")
        except Exception as e:
            logging.warning(f"Failed to load planner config: {e}")
        
        # Safe defaults
        return {
            'enabled': True,
            'inject_into_prompts': True,
            'max_steps': 10,
            'skip_fallback_plans': True
        }
    
    def _get_circuit_breaker_config(self, circuit_name: str) -> Dict[str, Any]:
        """
        Load circuit breaker configuration from cot_review_config.json.
        Returns circuit-specific settings with safe defaults.
        
        Enterprise Enhancement: Configuration-driven circuit breaker parameters
        """
        try:
            import json
            from pathlib import Path
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "cot_review_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    cb_config = config.get('circuit_breaker', {})
                    
                    if not cb_config.get('enabled', True):
                        logging.info("Circuit breaker disabled in configuration")
                        return None
                    
                    # Get circuit-specific config
                    circuit_settings = cb_config.get('circuits', {}).get(circuit_name, {})
                    
                    if circuit_settings:
                        logging.debug(f"Loaded circuit breaker config for {circuit_name}: {circuit_settings}")
                        return circuit_settings
            else:
                logging.debug("cot_review_config.json not found, using circuit breaker defaults")
        except Exception as e:
            logging.warning(f"Failed to load circuit breaker config: {e}")
        
        # Safe defaults if config not found
        return {
            'failure_threshold': 3,
            'recovery_timeout': 60,
            'success_threshold': 2,
            'timeout': 30
        }
    
    def _load_orchestrator_config(self) -> Dict[str, Any]:
        """
        DEPRECATED: Config now loaded directly by orchestrator from cot_review_config.json
        Kept for backward compatibility but not used anymore.
        """
        cot_config = self._load_cot_config()
        return cot_config
    
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
            
            # HIGH PRIORITY: Queries asking for specific values/records/names
            # These MUST go to DataAnalyst for code generation, NOT StatisticalAgent
            specific_value_patterns = [
                "what is the", "which is the", "find the", "show me the",
                "who is the", "where is the", "get the",
                "most", "least", "highest", "lowest", "top", "bottom",
                "best", "worst", "largest", "smallest", "maximum", "minimum",
                "with highest", "with lowest", "with most", "with least"
            ]
            if any(pattern in query_lower for pattern in specific_value_patterns):
                confidence += 0.5  # Strong boost - we handle specific value lookups
            
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
        """
        Execute analysis using QueryOrchestrator for unified decision making.
        
        Phase 1 Enhanced: Uses execute_with_fallback for resilient operation.
        """
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
            
            # 2. Use QueryOrchestrator for UNIFIED decision making (3-track system)
            orchestrator = self._get_orchestrator()
            execution_plan = orchestrator.create_execution_plan(
                query=query,
                data=optimized_data,
                context={'columns': available_columns, 'filepath': filepath}
            )
            
            # Extract decisions from plan
            selected_model = kwargs.get('force_model') or execution_plan.model
            complexity_score = execution_plan.complexity_score
            execution_method = execution_plan.execution_method
            review_level = execution_plan.review_level
            
            logging.info(f"QueryOrchestrator decision: {execution_plan.reasoning}")
            
            # 3. Dynamic Planning (optional enhancement - respects config)
            analysis_plan = None
            try:
                # Load config to check if dynamic planning is enabled
                planner_config = self._get_planner_config()
                if planner_config.get('enabled', True):  # Enabled by default
                    planner = get_dynamic_planner()
                    analysis_plan = planner.create_plan(query, data_info)
                    logging.debug("DynamicPlanner invoked for analysis strategy")
                else:
                    logging.debug("DynamicPlanner disabled by configuration")
            except Exception as plan_error:
                logging.warning(f"DynamicPlanner failed (continuing without plan): {plan_error}")
                analysis_plan = None
            
            # 4. Determine if Two Friends Model (CoT) should be used based on orchestrator
            cot_config = self._load_cot_config()
            use_cot = self._should_use_cot(review_level, cot_config)
            
            # 5. Execute with Phase 1 fallback protection
            if PHASE1_AVAILABLE:
                result, metadata = self._execute_with_phase1_fallback(
                    query=query,
                    data_info=data_info,
                    optimized_data=optimized_data,
                    available_columns=available_columns,
                    selected_model=selected_model,
                    complexity_score=complexity_score,
                    filepath=filepath,
                    analysis_plan=analysis_plan,
                    execution_method=execution_method,
                    use_cot=use_cot,
                    execution_plan=execution_plan
                )
            else:
                # Legacy execution path
                if execution_method == ExecutionMethod.CODE_GENERATION:
                    result, metadata = self._execute_with_code_gen(
                        query, data_info, optimized_data, available_columns,
                        selected_model, filepath, analysis_plan
                    )
                elif use_cot:
                    result, metadata = self._execute_with_cot(
                        query, data_info, optimized_data, available_columns,
                        selected_model, complexity_score, filepath, analysis_plan
                    )
                else:
                    result = self._execute_direct(
                        query, data_info, filename, selected_model, analysis_plan
                    )
                    metadata = {}
            
            # Return successful result
            metadata.update({
                "agent": "DataAnalyst",
                "model": selected_model,
                "complexity": complexity_score,
                "execution_method": execution_method.value if hasattr(execution_method, 'value') else str(execution_method),
                "review_level": review_level.value if hasattr(review_level, 'value') else str(review_level)
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

    async def execute_async(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Async version of execute for non-blocking operation.
        Mirrors execute() but uses async LLM calls where possible.
        """
        filename = kwargs.get('filename')
        filepath = kwargs.get('filepath')
        
        if filename and not filepath:
             filepath = self._resolve_filepath(filename)
             
        if not filepath and not data:
            return {"success": False, "error": "No file provided"}
            
        try:
            self.initializer.ensure_initialized()
            
            # 1. Optimize Data (sync - file I/O)
            data_info, optimized_data, available_columns = self._optimize_data(filepath, filename)
            
            # 2. Use QueryOrchestrator
            orchestrator = self._get_orchestrator()
            execution_plan = orchestrator.create_execution_plan(
                query=query,
                data=optimized_data,
                context={'columns': available_columns, 'filepath': filepath}
            )
            
            selected_model = kwargs.get('force_model') or execution_plan.model
            complexity_score = execution_plan.complexity_score
            execution_method = execution_plan.execution_method
            review_level = execution_plan.review_level
            
            logging.info(f"QueryOrchestrator decision (async): {execution_plan.reasoning}")
            
            # 3. Dynamic Planning
            try:
                planner = get_dynamic_planner()
                analysis_plan = planner.create_plan(query, data_info)
            except Exception:
                analysis_plan = None
            
            # 4. Determine if CoT should be used
            cot_config = self._load_cot_config()
            use_cot = self._should_use_cot(review_level, cot_config)
            
            # 5. Execute - use async LLM calls for direct execution
            # Note: Code generation and CoT are still sync for now (future enhancement)
            if execution_method == ExecutionMethod.CODE_GENERATION:
                result, metadata = self._execute_with_code_gen(
                    query, data_info, optimized_data, available_columns,
                    selected_model, filepath
                )
            elif use_cot:
                result, metadata = self._execute_with_cot(
                    query, data_info, optimized_data, available_columns,
                    selected_model, complexity_score, filepath, analysis_plan
                )
            else:
                # Use async LLM call for direct execution
                result = await self._execute_direct_async(
                    query, data_info, filename, selected_model, analysis_plan
                )
                metadata = {}
                
            metadata.update({
                "agent": "DataAnalyst",
                "model": selected_model,
                "complexity": complexity_score,
                "execution_method": execution_method.value,
                "review_level": review_level.value,
                "method": "CoT" if use_cot else "Direct",
                "phase1_enabled": PHASE1_AVAILABLE
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
            
            # Phase 1: Provide graceful degradation instead of failure
            if PHASE1_AVAILABLE:
                degraded = GracefulDegradation.generate_degraded_response(
                    query=query,
                    context={'filepath': filepath},
                    error=str(e)
                )
                degraded['traceback'] = tb
                return degraded
            
            return {"success": False, "error": str(e), "traceback": tb}
    
    def _execute_with_phase1_fallback(self, 
                                      query: str,
                                      data_info: str,
                                      optimized_data: Dict,
                                      available_columns: List,
                                      selected_model: str,
                                      complexity_score: float,
                                      filepath: str,
                                      analysis_plan: Any,
                                      execution_method: ExecutionMethod,
                                      use_cot: bool,
                                      execution_plan) -> tuple:
        """
        Phase 1: Execute with smart fallback through model chain.
        Process never stops completely.
        """
        filename = os.path.basename(filepath) if filepath else "data"
        
        def execute_with_model(model: str):
            """Inner execution function for fallback chain - MUST return dict"""
            if execution_method == ExecutionMethod.CODE_GENERATION:
                result, metadata = self._execute_with_code_gen(
                    query, data_info, optimized_data, available_columns,
                    model, filepath
                )
                return {'success': True, 'result': result, 'metadata': metadata}
            elif use_cot:
                result, metadata = self._execute_with_cot(
                    query, data_info, optimized_data, available_columns,
                    model, complexity_score, filepath, analysis_plan
                )
                return {'success': True, 'result': result, 'metadata': metadata}
            else:
                result = self._execute_direct(
                    query, data_info, filename, model, analysis_plan
                )
                return {'success': True, 'result': result, 'metadata': {}}
        
        # Use orchestrator's execute_with_fallback
        orchestrator = self._get_orchestrator()
        result = orchestrator.execute_with_fallback(
            plan=execution_plan,
            execute_func=execute_with_model,
            max_retries=2
        )
        
        # Check if result is a dict (fallback/degraded response)
        if isinstance(result, dict):
            if result.get('degraded'):
                return result.get('message', str(result)), {
                    'degraded': True,
                    'fallback_activated': True
                }
            return result.get('result', str(result)), result.get('_execution_metadata', {})
        
        return result, {'fallback_activated': False}
    
    def _should_use_cot(self, review_level: ReviewLevel, cot_config: Dict[str, Any]) -> bool:
        """Determine if CoT (Two Friends Model) should be used based on review level"""
        if not cot_config.get('cot_review', {}).get('enabled', False):
            return False
        
        if review_level == ReviewLevel.MANDATORY:
            return True
        elif review_level == ReviewLevel.OPTIONAL:
            return cot_config.get('cot_review', {}).get('auto_enable_on_routing', True)
        else:  # NONE
            return False
    
    def _execute_with_code_gen(self, query, data_info, optimized_data, available_columns, model, filepath, analysis_plan=None) -> Dict[str, Any]:
        """
        Execute using LLM code generation (Phase 2 implementation).
        Returns a dict with result, code, and execution metadata.
        """
        try:
            import pandas as pd
            from backend.io.code_generator import get_code_generator
            
            # Validate filepath
            if not filepath:
                logging.warning("Code generation: No filepath provided - falling back to direct LLM")
                fallback_result = self._execute_direct(query, data_info, "data", model, None)
                return fallback_result, {"execution_method": "direct_llm_fallback", "code_gen_error": "No file path"}
            
            # Load the actual data based on file type
            file_ext = os.path.splitext(filepath)[1].lower()
            try:
                if file_ext == '.json':
                    df = pd.read_json(filepath)
                elif file_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(filepath)
                elif file_ext == '.csv':
                    df = pd.read_csv(filepath)
                else:
                    # Try CSV as fallback for unknown types
                    df = pd.read_csv(filepath)
            except Exception as read_error:
                logging.warning(f"Failed to read file as {file_ext}: {read_error} - falling back to direct LLM")
                fallback_result = self._execute_direct(query, data_info, os.path.basename(filepath), model, None)
                return fallback_result, {"execution_method": "direct_llm_fallback", "code_gen_error": str(read_error)}
            
            data_file = os.path.basename(filepath)
            
            # Get code generator
            code_gen = get_code_generator()
            
            # Build analysis context from plan if available (with validation)
            analysis_context = None
            if analysis_plan:
                try:
                    if hasattr(analysis_plan, 'summary'):
                        summary = str(analysis_plan.summary).strip()
                        # Skip fallback plans (they don't add value)
                        if summary and summary != "Fallback analysis due to planning error":
                            analysis_context = {'strategy': summary}
                            logging.info(f"✅ DynamicPlanner strategy for code gen: {summary[:80]}...")
                            
                            # Add steps with validation
                            if hasattr(analysis_plan, 'steps') and analysis_plan.steps:
                                try:
                                    validated_steps = []
                                    for step in analysis_plan.steps:
                                        if hasattr(step, 'description'):
                                            step_text = str(step.description).strip()
                                        else:
                                            step_text = str(step).strip()
                                        if step_text and len(step_text) < 500:
                                            validated_steps.append(step_text)
                                    if validated_steps:
                                        analysis_context['steps'] = validated_steps[:10]  # Max 10 steps
                                        logging.debug(f"Added {len(validated_steps)} validated steps")
                                except Exception as step_error:
                                    logging.warning(f"Step validation failed: {step_error}")
                        else:
                            logging.debug("Skipping fallback plan for code generation")
                except Exception as context_error:
                    logging.warning(f"Failed to build analysis_context: {context_error}")
                    analysis_context = None  # Fail gracefully
            
            # Generate and execute code with history tracking
            result = code_gen.generate_and_execute(
                query=query,
                df=df,
                model=model,
                max_retries=2,
                data_file=data_file,
                save_history=True,
                analysis_context=analysis_context
            )
            
            if result.success:
                logging.info(f"Code generation succeeded: {result.execution_time_ms:.1f}ms")
                
                # Build structured response with code included
                # Build metadata with generated code for API response
                metadata = {
                    "execution_method": "code_generation",
                    "generated_code": result.generated_code,  # Original LLM code
                    "executed_code": result.code,             # Cleaned/executed code
                    "code": result.code,                      # For API compatibility
                    "execution_id": result.execution_id,
                    "execution_time_ms": result.execution_time_ms,
                    "code_gen_model": model,
                    "attempt_count": result.attempt_count,
                    "retry_errors": result.retry_errors
                }
                
                # Format the actual result clearly - THIS IS THE ANSWER, not the code
                if result.result is not None:
                    if isinstance(result.result, pd.DataFrame):
                        # For DataFrame results, format nicely
                        if len(result.result) <= 20:
                            display_text = f"## Analysis Result\n\n{result.result.to_markdown()}"
                        else:
                            display_text = f"## Analysis Result (Top 20 of {len(result.result)} rows)\n\n{result.result.head(20).to_markdown()}"
                    elif isinstance(result.result, dict):
                        # Extract and format dict result
                        actual_result = result.result.get('result', result.result)
                        if isinstance(actual_result, (int, float)):
                            display_text = f"## Answer\n\n**{actual_result:,}**"
                        else:
                            display_text = f"## Answer\n\n{actual_result}"
                    elif isinstance(result.result, (int, float)):
                        display_text = f"## Answer\n\n**{result.result:,}**"
                    elif isinstance(result.result, str):
                        display_text = f"## Answer\n\n{result.result}"
                    else:
                        display_text = f"## Answer\n\n{str(result.result)}"
                else:
                    display_text = "Analysis completed but no result was returned."
                
                # Note: Code is available in metadata for Details tab, not shown in main result
                # to keep the answer clean and readable
                
                return display_text, metadata
            else:
                # Fallback to direct LLM on failure
                logging.warning(f"Code generation failed: {result.error} - falling back to direct LLM")
                fallback_result = self._execute_direct(query, data_info, os.path.basename(filepath), model, None)
                return fallback_result, {"execution_method": "direct_llm_fallback", "code_gen_error": result.error}
                
        except Exception as e:
            logging.error(f"Code generation error: {e} - falling back to direct LLM")
            fallback_result = self._execute_direct(query, data_info, os.path.basename(filepath), model, None)
            return fallback_result, {"execution_method": "direct_llm_fallback", "code_gen_error": str(e)}

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
            # Guard against None filepath
            if not filepath:
                logging.warning(f"No filepath provided for optimization, filename: {filename}")
                return f"No file path available for {filename}", {'is_optimized': False}, []
            
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

    # NOTE: _select_model and _get_complexity_score REMOVED
    # Model selection and complexity analysis now handled by QueryOrchestrator
    # See _get_orchestrator() and execute() for unified 3-track decision system

    def _execute_direct(self, query, data_info, filename, selected_model, analysis_plan=None):
        # Intelligent context based on data characteristics (not query keywords)
        # Data optimizer already analyzed the data - respect its decisions
        hint = ""
        if 'PRE-CALCULATED STATISTICS' in data_info:
            # Large dataset detected by optimizer - add minimal guidance
            hint = "Note: Use the pre-calculated statistics provided below.\n\n"
        
        # Add analysis plan if available (DynamicPlanner's strategy)
        plan_context = ""
        if analysis_plan:
            try:
                # Validate and sanitize analysis_plan structure
                if hasattr(analysis_plan, 'summary'):
                    summary = str(analysis_plan.summary).strip()
                    if summary and summary != "Fallback analysis due to planning error":
                        plan_context = f"\n\n📋 ANALYSIS STRATEGY:\n{summary}\n"
                        logging.info(f"✅ DynamicPlanner strategy injected (direct execution): {summary[:80]}...")
                        
                        # Add steps if available and valid
                        if hasattr(analysis_plan, 'steps') and analysis_plan.steps:
                            try:
                                steps_list = []
                                for i, step in enumerate(analysis_plan.steps):
                                    # Handle both string and AnalysisStep objects
                                    if hasattr(step, 'description'):
                                        step_text = str(step.description).strip()
                                    else:
                                        step_text = str(step).strip()
                                    
                                    if step_text and len(step_text) < 500:  # Safety: max 500 chars per step
                                        steps_list.append(f"{i+1}. {step_text}")
                                
                                if steps_list:
                                    steps_text = "\n".join(steps_list[:10])  # Safety: max 10 steps
                                    plan_context += f"\nSTEPS:\n{steps_text}\n"
                                    logging.debug(f"Added {len(steps_list)} steps to prompt")
                            except Exception as step_error:
                                logging.warning(f"Failed to process steps: {step_error}")
                    else:
                        logging.debug("DynamicPlanner returned fallback plan, skipping injection")
            except Exception as plan_error:
                logging.warning(f"Failed to inject analysis plan: {plan_error}")
                plan_context = ""  # Fail gracefully
        
        # Clean, direct prompt - query unchanged
        prompt = f"""{hint}Question: {query}

Data from: {filename}

{data_info}{plan_context}

Answer:"""
        
        # FIX 12 (ENTERPRISE): Circuit Breaker Protection with Configuration
        try:
            if PHASE1_AVAILABLE:
                # Load circuit-specific configuration
                cb_config = self._get_circuit_breaker_config(self._circuit_name)
                
                if cb_config is None:
                    # Circuit breaker disabled in config
                    response = self.initializer.llm_client.generate(prompt, model=selected_model)
                    if isinstance(response, dict): return response.get('response', str(response))
                    return str(response)
                
                # Create circuit breaker with config
                from backend.infra.circuit_breaker import CircuitBreakerConfig
                config = CircuitBreakerConfig(
                    failure_threshold=cb_config.get('failure_threshold', 3),
                    recovery_timeout=cb_config.get('recovery_timeout', 60.0),
                    success_threshold=cb_config.get('success_threshold', 2),
                    timeout=cb_config.get('timeout', 30.0)
                )
                circuit = get_circuit_breaker(self._circuit_name, config)
                
                # Wrap LLM call in circuit breaker for graceful degradation
                def llm_call():
                    response = self.initializer.llm_client.generate(prompt, model=selected_model)
                    # Format response to circuit breaker expected format
                    if isinstance(response, dict):
                        return {"success": True, "response": response.get('response', str(response))}
                    return {"success": True, "response": str(response)}
                
                result = circuit.call(llm_call)
                
                if result.get("fallback_used"):
                    logging.warning(f"⚠️ Circuit breaker fallback used for {self._circuit_name}")
                else:
                    logging.debug(f"✅ Circuit breaker call successful for {self._circuit_name}")
                
                return result.get("response", result.get("result", str(result)))
            else:
                # Fallback if Phase 1 not available
                response = self.initializer.llm_client.generate(prompt, model=selected_model)
                if isinstance(response, dict): return response.get('response', str(response))
                return str(response)
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            return f"Analysis failed: {str(e)}. Please check if Ollama is running and models are available."

    async def _execute_direct_async(self, query, data_info, filename, selected_model, analysis_plan=None):
        """Async version of _execute_direct for non-blocking LLM calls."""
        hint = ""
        if 'PRE-CALCULATED STATISTICS' in data_info:
            hint = "Note: Use the pre-calculated statistics provided below.\n\n"
        
        # Add analysis plan if available (DynamicPlanner's strategy)
        plan_context = ""
        if analysis_plan:
            try:
                # Validate and sanitize analysis_plan structure
                if hasattr(analysis_plan, 'summary'):
                    summary = str(analysis_plan.summary).strip()
                    if summary and summary != "Fallback analysis due to planning error":
                        plan_context = f"\n\n📋 ANALYSIS STRATEGY:\n{summary}\n"
                        logging.info(f"✅ DynamicPlanner strategy injected (async direct): {summary[:80]}...")
                        
                        if hasattr(analysis_plan, 'steps') and analysis_plan.steps:
                            try:
                                steps_list = []
                                for i, step in enumerate(analysis_plan.steps):
                                    if hasattr(step, 'description'):
                                        step_text = str(step.description).strip()
                                    else:
                                        step_text = str(step).strip()
                                    if step_text and len(step_text) < 500:
                                        steps_list.append(f"{i+1}. {step_text}")
                                if steps_list:
                                    steps_text = "\n".join(steps_list[:10])
                                    plan_context += f"\nSTEPS:\n{steps_text}\n"
                            except Exception as step_error:
                                logging.warning(f"Failed to process steps: {step_error}")
            except Exception as plan_error:
                logging.warning(f"Failed to inject analysis plan: {plan_error}")
                plan_context = ""
        
        prompt = f"""{hint}Question: {query}

Data from: {filename}

{data_info}{plan_context}

Answer:"""
        
        # FIX 12 (ENTERPRISE): Circuit Breaker Protection for async LLM calls
        try:
            if PHASE1_AVAILABLE:
                # Load circuit-specific configuration
                cb_config = self._get_circuit_breaker_config(self._circuit_name)
                
                if cb_config is None:
                    # Circuit breaker disabled in config
                    response = await self.initializer.llm_client.generate_async(prompt, model=selected_model)
                    if isinstance(response, dict): return response.get('response', str(response))
                    return str(response)
                
                # Create circuit breaker with config
                from backend.infra.circuit_breaker import CircuitBreakerConfig
                config = CircuitBreakerConfig(
                    failure_threshold=cb_config.get('failure_threshold', 3),
                    recovery_timeout=cb_config.get('recovery_timeout', 60.0),
                    success_threshold=cb_config.get('success_threshold', 2),
                    timeout=cb_config.get('timeout', 30.0)
                )
                circuit = get_circuit_breaker(self._circuit_name, config)
                
                # Wrap async LLM call in circuit breaker
                async def async_llm_call():
                    response = await self.initializer.llm_client.generate_async(prompt, model=selected_model)
                    if isinstance(response, dict):
                        return {"success": True, "response": response.get('response', str(response))}
                    return {"success": True, "response": str(response)}
                
                # Note: Circuit breaker.call() is sync, but we can wrap it
                result = circuit.call(lambda: asyncio.run(async_llm_call()))
                
                if result.get("fallback_used"):
                    logging.warning(f"⚠️ Circuit breaker fallback used for async {self._circuit_name}")
                else:
                    logging.debug(f"✅ Circuit breaker async call successful for {self._circuit_name}")
                
                return result.get("response", result.get("result", str(result)))
            else:
                # Fallback if Phase 1 not available
                response = await self.initializer.llm_client.generate_async(prompt, model=selected_model)
                if isinstance(response, dict): return response.get('response', str(response))
                return str(response)
        except Exception as e:
            logging.error(f"Async LLM call failed: {e}")
            return f"Analysis failed: {str(e)}. Please check if Ollama is running and models are available."

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
            critic_model=critic_model,
            analysis_plan=plan
        )
        return result.final_output, {"cot_iterations": result.total_iterations}
