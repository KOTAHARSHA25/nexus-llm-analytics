"""
Analysis Executor Module
========================
Handles structured data analysis execution with intelligent routing.
Extracted from crew_manager.py for better maintainability.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from backend.agents.model_initializer import get_model_initializer
from backend.agents.agent_factory import get_agent_factory


def friendly_error(error_msg: str, suggestion: str = "") -> Dict[str, Any]:
    """Create a user-friendly error response."""
    return {
        "success": False,
        "error": error_msg,
        "suggestion": suggestion,
        "type": "error"
    }


class AnalysisExecutor:
    """
    Executes structured data analysis with intelligent routing and CoT support.
    """
    
    def __init__(self):
        self._initializer = get_model_initializer()
        self._factory = get_agent_factory()
        self._query_parser = None
        self._cot_engine = None
        self._cot_config = None
        
        logging.info("📊 AnalysisExecutor created (lazy loading enabled)")
    
    @property
    def query_parser(self):
        """Lazy load query parser."""
        if self._query_parser is None:
            from backend.core.query_parser import EnhancedQueryParser
            self._query_parser = EnhancedQueryParser()
        return self._query_parser
    
    def _load_cot_config(self) -> Dict[str, Any]:
        """Load Chain-of-Thought configuration."""
        if self._cot_config is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', '..',
                'config', 'cot_review_config.json'
            )
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    self._cot_config = json.load(f)
            else:
                self._cot_config = {'enabled': False}
        return self._cot_config
    
    def _ensure_cot_engine(self):
        """Ensure CoT engine is initialized if enabled."""
        config = self._load_cot_config()
        if config.get('enabled') and self._cot_engine is None:
            try:
                from backend.core.self_correction_engine import SelfCorrectionEngine
                self._cot_engine = SelfCorrectionEngine(config)
            except ImportError:
                logging.warning("SelfCorrectionEngine not available")
        return self._cot_engine
    
    def analyze_structured(
        self,
        query: str,
        filename: str,
        analysis_id: Optional[str] = None,
        max_retries: int = 2,
        enable_review: bool = False,
        user_model: Optional[str] = None,
        force_model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze structured data (CSV, JSON) with intelligent routing.
        
        Args:
            query: User's analysis query
            filename: Name of the data file
            analysis_id: Optional ID for cancellation tracking
            max_retries: Maximum retry attempts
            enable_review: Whether to enable reviewer agent
            user_model: User's preferred model
            force_model: Force specific model (highest priority)
            
        Returns:
            Analysis results dictionary
        """
        current_retry = 0
        
        while current_retry <= max_retries:
            try:
                # Check for cancellation
                if analysis_id:
                    try:
                        from backend.core.analysis_manager import check_cancellation
                        check_cancellation(analysis_id)
                    except Exception:
                        return {"error": "Analysis was cancelled", "status": "cancelled"}
                
                logging.info(f"📊 Starting structured analysis (attempt {current_retry + 1}/{max_retries + 1})")
                
                # Ensure models are initialized
                self._initializer.ensure_initialized()
                
                # Resolve file path
                filepath = self._resolve_filepath(filename)
                if not filepath:
                    return friendly_error(
                        f"File not found: {filename}",
                        "Please upload the file first"
                    )
                
                # Optimize data for LLM
                data_info, optimized_data, available_columns = self._optimize_data(filepath, filename)
                
                # Parse the query
                parsed_query = self.query_parser.parse_query(
                    query,
                    available_columns=available_columns,
                    data_sample=optimized_data.get('sample', [])[:3] if optimized_data.get('sample') else None
                )
                
                logging.info(f"   Intent: {parsed_query.intent.value}, confidence: {parsed_query.confidence:.2f}")
                
                # Select model (priority: force_model > user preference > intelligent routing)
                selected_model, routing_decision = self._select_model(
                    query=query,
                    optimized_data=optimized_data,
                    available_columns=available_columns,
                    filepath=filepath,
                    force_model=force_model or user_model
                )
                
                # Calculate complexity for CoT decision
                complexity_score = self._get_complexity_score(
                    query, optimized_data, available_columns, filepath, routing_decision
                )
                
                # Determine if CoT self-correction should be used (Hybrid Routing)
                cot_config = self._load_cot_config()
                # Roadmap recommends 0.3-0.4 threshold for hybrid routing
                threshold = cot_config.get('complexity_threshold', 0.4)
                
                use_cot = (
                    cot_config.get('enabled', False) and
                    cot_config.get('auto_enable_on_routing', True) and
                    complexity_score >= threshold
                )
                
                logging.info(f"🛣️ Hybrid Routing: {'SLOW LANE (CoT)' if use_cot else 'FAST LANE (Direct)'} (Score: {complexity_score:.2f} vs Threshold: {threshold})")
                
                # Execute analysis
                if use_cot:
                    analysis_answer, cot_metadata = self._execute_with_cot(
                        query, data_info, optimized_data, available_columns,
                        selected_model, complexity_score, filepath
                    )
                else:
                    analysis_answer = self._execute_direct(
                        query, data_info, filename, selected_model, routing_decision
                    )
                    cot_metadata = {}
                
                # Check cancellation again
                if analysis_id:
                    try:
                        from backend.core.analysis_manager import check_cancellation
                        check_cancellation(analysis_id)
                    except Exception:
                        return {"error": "Analysis was cancelled", "status": "cancelled"}
                
                # Optional review step
                if enable_review:
                    analysis_answer = self._review_answer(query, analysis_answer)
                
                # Get preferences for response
                from backend.core.user_preferences import get_preferences_manager
                prefs = get_preferences_manager().load_preferences()
                
                return {
                    "success": True,
                    "result": analysis_answer,
                    "filename": filename,
                    "query": query,
                    "type": "structured_analysis",
                    "execution_time": 0,
                    "retry_attempt": current_retry,
                    "metadata": cot_metadata,
                    "routing_info": {
                        "selected_model": selected_model,
                        "selected_tier": routing_decision.selected_tier.value if routing_decision else "manual",
                        "complexity_score": round(routing_decision.complexity_score, 3) if routing_decision else round(complexity_score, 3),
                        "routing_time_ms": round(routing_decision.routing_time_ms, 2) if routing_decision else 0,
                        "intelligent_routing_enabled": prefs.enable_intelligent_routing,
                        "using_force_model": bool(force_model or user_model),
                        "reason": self._get_routing_reason(force_model or user_model, prefs, routing_decision)
                    }
                }
                
            except Exception as e:
                current_retry += 1
                logging.error(f"❌ Analysis failed (attempt {current_retry}/{max_retries + 1}): {e}")
                
                if current_retry <= max_retries:
                    logging.info(f"🔄 Retrying analysis...")
                    continue
                else:
                    return friendly_error(
                        f"Analysis failed after {max_retries + 1} attempts: {str(e)}",
                        "Check your query and data file format. Try simplifying the query."
                    )
        
        return friendly_error("Unexpected error in analysis", "Please try again")
    
    def _resolve_filepath(self, filename: str) -> Optional[str]:
        """Resolve the full path to a data file."""
        base_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
        
        # Check uploads directory first
        uploads_path = os.path.join(base_data_dir, 'uploads', filename)
        if os.path.exists(uploads_path):
            return uploads_path
        
        # Check samples directory
        samples_path = os.path.join(base_data_dir, 'samples', filename)
        if os.path.exists(samples_path):
            return samples_path
        
        # Check samples subdirectories
        file_ext = os.path.splitext(filename)[1].lower()
        subdir = 'csv' if file_ext == '.csv' else 'json' if file_ext == '.json' else None
        
        if subdir:
            subdir_path = os.path.join(base_data_dir, 'samples', subdir, filename)
            if os.path.exists(subdir_path):
                return subdir_path
        
        # Fallback to direct path
        direct_path = os.path.join(base_data_dir, filename)
        if os.path.exists(direct_path):
            return direct_path
        
        return None
    
    def _optimize_data(self, filepath: str, filename: str) -> tuple:
        """Optimize data for LLM consumption."""
        try:
            from utils.data_optimizer import DataOptimizer
            
            # CRITICAL FIX: Increased max_chars from 3000 to 8000 to include 
            # all pre-calculated statistics (sums, averages, grouped aggregations)
            # Without these stats, LLM hallucinates answers instead of using real data
            optimizer = DataOptimizer(max_rows=100, max_chars=8000)
            optimized_data = optimizer.optimize_for_llm(filepath)
            
            if optimized_data['is_optimized']:
                data_info = optimized_data['preview']
                available_columns = optimized_data.get('stats', {}).get('columns', [])
                
                if optimized_data.get('was_nested'):
                    logging.info("   Data was nested, flattened for LLM processing")
                logging.info(f"   Data optimized: {optimized_data['total_rows']} rows -> {len(optimized_data.get('sample', []))} sample rows")
            else:
                data_info = optimized_data['preview']
                available_columns = []
            
            return data_info, optimized_data, available_columns
            
        except Exception as e:
            logging.warning(f"⚠️ Could not optimize data: {e}")
            return (
                f"File: {filename} (data loading will be done during analysis)",
                {'is_optimized': False, 'preview': '', 'total_rows': 0, 'stats': {}, 'sample': None},
                []
            )
    
    def _select_model(
        self,
        query: str,
        optimized_data: Dict[str, Any],
        available_columns: List[str],
        filepath: str,
        force_model: Optional[str] = None
    ) -> tuple:
        """Select the appropriate model for analysis."""
        from backend.core.user_preferences import get_preferences_manager
        
        prefs = get_preferences_manager().load_preferences()
        user_primary_model = self._initializer.cached_models.get('primary', '').replace("ollama/", "")
        
        if force_model:
            logging.info(f"🎯 [FORCE MODEL] Using explicitly requested model: {force_model}")
            return force_model, None
        
        if not prefs.enable_intelligent_routing:
            logging.info(f"👤 [USER CHOICE] Using user's primary model: {user_primary_model}")
            return user_primary_model, None
        
        # Intelligent routing enabled
        data_complexity_info = {
            'rows': optimized_data.get('total_rows', 0),
            'columns': len(available_columns),
            'data_types': optimized_data.get('stats', {}).get('column_types', {}),
            'file_size_mb': os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0
        }
        
        routing_decision = self._initializer.intelligent_router.route(query, data_complexity_info)
        
        # Validate tier for complex queries
        if routing_decision.complexity_score > 0.5 and routing_decision.selected_tier.value == 'fast':
            logging.warning(f"⚠️ Query complexity ({routing_decision.complexity_score:.3f}) too high for FAST tier")
            selected_model = routing_decision.fallback_model or user_primary_model
        else:
            selected_model = routing_decision.selected_model
        
        logging.info(f"🧠 [INTELLIGENT ROUTING] Complexity: {routing_decision.complexity_score:.3f}, Model: {selected_model}")
        
        return selected_model, routing_decision
    
    def _get_complexity_score(
        self,
        query: str,
        optimized_data: Dict[str, Any],
        available_columns: List[str],
        filepath: str,
        routing_decision: Any
    ) -> float:
        """Get complexity score for CoT decision."""
        if routing_decision:
            return routing_decision.complexity_score
        
        # Calculate complexity without full routing
        data_complexity_info = {
            'rows': optimized_data.get('total_rows', 0),
            'columns': len(available_columns),
            'data_types': optimized_data.get('stats', {}).get('column_types', {}),
            'file_size_mb': os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0
        }
        
        try:
            from backend.core.query_complexity_analyzer import QueryComplexityAnalyzer
            analyzer = QueryComplexityAnalyzer()
            analysis = analyzer.analyze(query, data_complexity_info)
            logging.info(f"   Complexity analyzed for CoT: {analysis.total_score:.3f}")
            return analysis.total_score
        except Exception as e:
            logging.warning(f"Could not analyze complexity: {e}")
            return 0.5  # Default medium complexity
    
    def _execute_with_cot(
        self,
        query: str,
        data_info: str,
        optimized_data: Dict[str, Any],
        available_columns: List[str],
        selected_model: str,
        complexity_score: float,
        filepath: str
    ) -> tuple:
        """Execute analysis with Chain-of-Thought self-correction."""
        cot_config = self._load_cot_config()
        cot_engine = self._ensure_cot_engine()
        
        if not cot_engine:
            logging.warning("CoT engine not available, falling back to direct execution")
            return self._execute_direct(query, data_info, os.path.basename(filepath), selected_model, None), {}
        
        logging.info(f"🧠 CoT SELF-CORRECTION ACTIVATED (complexity: {complexity_score:.3f})")
        
        data_context = {
            'rows': optimized_data.get('total_rows', 0),
            'columns': available_columns,
            'data_types': optimized_data.get('stats', {}).get('column_types', {}),
            'stats_summary': data_info,
            'file_size_mb': os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0
        }
        
        critic_model = self._initializer.cached_models.get('review', '').replace("ollama/", "")
        
        correction_result = cot_engine.run_correction_loop(
            query=query,
            data_context=data_context,
            generator_model=selected_model,
            critic_model=critic_model
        )
        
        cot_metadata = {
            'cot_iterations': correction_result.total_iterations,
            'cot_validated': correction_result.success,
            'cot_termination': correction_result.termination_reason,
            'cot_time_seconds': correction_result.total_time_seconds,
            'cot_reasoning': correction_result.final_reasoning,
            'cot_enabled_reason': f"Auto-enabled (complexity {complexity_score:.3f})"
        }
        
        logging.info(f"✅ CoT Result: {correction_result.total_iterations} iterations, validated={correction_result.success}")
        
        return correction_result.final_output, cot_metadata
    
    def _execute_direct(
        self,
        query: str,
        data_info: str,
        filename: str,
        selected_model: str,
        routing_decision: Any
    ) -> str:
        """Execute direct LLM analysis without CoT."""
        direct_prompt = f"""You are a precise data analyst. Answer ONLY using the pre-calculated statistics provided below.

QUESTION: {query}

DATA FILE: {filename}

=== PRE-CALCULATED DATA (USE THESE EXACT NUMBERS) ===
{data_info}
=== END OF DATA ===

ANSWER LOOKUP GUIDE:
1. "total X?" → Find "OVERALL COLUMN STATISTICS" → Look for the column named X → Use "TOTAL (sum of all values)"
2. "average X?" → Find "OVERALL COLUMN STATISTICS" → Look for X column → Use "AVERAGE (mean)"
3. "maximum X?" → Find "OVERALL COLUMN STATISTICS" → Look for X column → Use "MAXIMUM (largest single value)"
4. "minimum X?" → Find "OVERALL COLUMN STATISTICS" → Look for X column → Use "MINIMUM (smallest single value)"
5. "how many rows?" → Use "Total Rows:" from Dataset Overview
6. "highest/best region by REVENUE?" → Find "QUICK RANKINGS" → Use "HIGHEST REVENUE by REGION" line
7. "highest/best region by SALES?" → Find "QUICK RANKINGS" → Use "HIGHEST SALES by REGION" line
8. "X by region/product?" → Look in "GROUPED AGGREGATIONS" for breakdown

⚠️ CRITICAL - READ CAREFULLY:
- Match the EXACT column name from the question. REVENUE ≠ SALES. They are different columns!
- If question asks about "REVENUE", look for lines containing "REVENUE" (not SALES)
- If question asks about "SALES", look for lines containing "SALES" (not REVENUE)
- For "highest region by REVENUE": Find "HIGHEST REVENUE by REGION" in QUICK RANKINGS
- Copy the value EXACTLY as shown
- State your answer in one clear sentence with the name and exact number

YOUR ANSWER:"""
        
        try:
            response = self._initializer.llm_client.generate(
                prompt=direct_prompt,
                model=selected_model,
                adaptive_timeout=True
            )
            
            if isinstance(response, dict) and 'response' in response:
                return response['response']
            return str(response)
            
        except Exception as e:
            logging.error(f"❌ Direct LLM call failed with {selected_model}: {e}")
            
            # Try fallback
            if routing_decision and hasattr(routing_decision, 'fallback_model') and routing_decision.fallback_model:
                try:
                    logging.info(f"🔄 Trying fallback model: {routing_decision.fallback_model}")
                    fallback_response = self._initializer.llm_client.generate(
                        prompt=direct_prompt,
                        model=routing_decision.fallback_model,
                        adaptive_timeout=True
                    )
                    
                    if isinstance(fallback_response, dict) and 'response' in fallback_response:
                        return fallback_response['response']
                    return str(fallback_response)
                except Exception as fallback_error:
                    logging.error(f"❌ Fallback also failed: {fallback_error}")
            
            return f"Error getting analysis: {str(e)}"
    
    def _review_answer(self, query: str, analysis_answer: str) -> str:
        """Review the analysis answer using the review LLM."""
        review_prompt = f"""QUESTION: {query}
ANSWER: {analysis_answer}

Please verify using the PRE-CALCULATED STATISTICS included in the data information.
If everything matches, reply only with the single word "Approved".
If you believe the PRE-CALCULATED STATISTICS are incorrect, provide a one-sentence correction."""

        try:
            review_llm = self._initializer.review_llm
            if not review_llm:
                return analysis_answer
                
            review_response = review_llm.call([{"role": "user", "content": review_prompt}])
            review_answer = str(review_response)
            
            if "approved" in review_answer.lower():
                return analysis_answer
            else:
                return f"{analysis_answer}\n\nReviewer's correction: {review_answer}"
        except Exception as e:
            logging.error(f"⚠️ Review LLM call failed: {e}")
            return analysis_answer
    
    def _get_routing_reason(self, force_model: Optional[str], prefs: Any, routing_decision: Any) -> str:
        """Get human-readable routing reason."""
        if force_model:
            return "Force model parameter"
        if not prefs.enable_intelligent_routing:
            return "User's primary model from settings (intelligent routing disabled)"
        if routing_decision:
            return f"Intelligent routing (complexity: {routing_decision.complexity_score:.3f})"
        return "Default model"


# Singleton instance
_analysis_executor: Optional[AnalysisExecutor] = None


def get_analysis_executor() -> AnalysisExecutor:
    """Get the singleton AnalysisExecutor instance."""
    global _analysis_executor
    if _analysis_executor is None:
        _analysis_executor = AnalysisExecutor()
    return _analysis_executor
