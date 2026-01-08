# Analysis Service
# High-level orchestrator for analysis requests (Service Layer)

import logging
import threading
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add src to path if needed (for direct execution)
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.plugin_system import get_agent_registry
from backend.core.engine.user_preferences import get_preferences_manager
from backend.io.result_interpreter import interpret_result, ResultInterpreter
from backend.core.semantic_mapper import get_semantic_mapper
from backend.core.engine.query_orchestrator import ReviewLevel
from backend.agents.model_manager import get_model_manager
from backend.utils.data_utils import DataPathResolver, read_dataframe
from backend.core.enhanced_cache_integration import get_enhanced_cache_manager


class AnalysisService:
    """
    Service layer for handling analysis requests.
    Routes requests to appropriate agents via Plugin Registry.
    Supports intelligent model routing when enabled.
    Replaces legacy CrewManager and AnalysisExecutor.
    """
    
    def __init__(self):
        self.registry = get_agent_registry()
        self._orchestrator = None
        # Ensure plugins are loaded
        # self.registry.discover_agents() # This is usually done on import or app startup
        logging.info("AnalysisService initialized")
    
    @property
    def orchestrator(self):
        """Get query orchestrator (lazy loaded)"""
        if self._orchestrator is None:
            from backend.core.engine.query_orchestrator import QueryOrchestrator
            self._orchestrator = QueryOrchestrator()
        return self._orchestrator

    @property
    def cot_engine(self):
        """Get self-correction engine (lazy loaded)"""
        if not hasattr(self, '_cot_engine') or self._cot_engine is None:
            try:
                from backend.core.engine.self_correction_engine import SelfCorrectionEngine
                
                # Get config from orchestrator to ensure consistency
                config = self.orchestrator.config
                if not config:
                    # Fallback if orchestrator fails to load config
                    config = {'max_iterations': 2, 'timeout_per_iteration_seconds': 30, 
                              'tags': {'reasoning_start': '[REASONING]', 'reasoning_end': '[/REASONING]', 
                                      'output_start': '[OUTPUT]', 'output_end': '[/OUTPUT]'},
                              'generator': {'system_prompt_template': 'generator_prompt.j2'},
                              'critic': {'system_prompt_template': 'critic_prompt.j2'}}
                
                manager = get_model_manager()
                self._cot_engine = SelfCorrectionEngine(config, manager.llm_client)
                logging.info("SelfCorrectionEngine initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize SelfCorrectionEngine: {e}")
                self._cot_engine = None
        return self._cot_engine
        
    @property
    def cache_manager(self):
        """Get enhanced cache manager (lazy loaded)"""
        if not hasattr(self, '_cache_manager'):
            try:
                self._cache_manager = get_enhanced_cache_manager()
            except Exception:
                self._cache_manager = None
        return self._cache_manager
    
    def _get_model_for_query(self, query: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Get optimal model for query using QueryOrchestrator.
        Returns None if routing is disabled or unavailable.
        """
        try:
            # Create execution plan using QueryOrchestrator
            plan = self.orchestrator.create_execution_plan(
                query=query,
                data=context.get('dataframe'),
                context=context
            )
            
            # Log the brain's reasoning
            logging.info(f"🧠 QueryOrchestrator Decision: {plan.model}")
            logging.info(f"   Reasoning: {plan.reasoning}")
            if plan.user_override:
                logging.info(f"   ⚠️ USER OVERRIDE: User explicitly chose this model")
            
            return plan.model
            
        except Exception as e:
            logging.warning(f"QueryOrchestrator failed, using default: {e}")
            return None

    async def analyze(self, 
                     query: str, 
                     context: Dict[str, Any] = None, 
                     user_id: str = None) -> Dict[str, Any]:
        """
        Main entry point for analysis.
        
        Args:
            query: The user's question or command
            context: Additional context (files, conversation history, etc.)
            user_id: User identifier for personalization
            
        Returns:
            Analysis result dictionary
        """
        context = context or {}
        filename = context.get('filename')
        file_type = Path(filename).suffix if filename else None
        
        # Resolve filepath if not already provided
        if filename and not context.get('filepath'):
            from backend.utils.data_utils import DataPathResolver
            resolved_path = DataPathResolver.resolve_data_file(filename)
            if resolved_path:
                context['filepath'] = str(resolved_path)
        
        logging.info(f"AnalysisService received query: {query} (file: {filename})")
        
        # 0a. Apply semantic layer to enhance query (Fix 8)
        # 0a. Apply semantic layer to enhance query (Fix 8)
        df_for_mapping = context.get('dataframe')
        if df_for_mapping is None and context.get('filepath'):
             try:
                 df_for_mapping = read_dataframe(context['filepath'], sample_size=100)
             except Exception:
                 pass # Ignore if load fails, skip mapping

        if df_for_mapping is not None:
            try:
                mapper = get_semantic_mapper()
                enhanced_query = mapper.enhance_query_context(query, df_for_mapping)
                if enhanced_query != query:
                    logging.info(f"🔍 Query enhanced with semantic concepts")
                    query = enhanced_query  # Use enhanced version
            except Exception as e:
                logging.warning(f"Semantic enhancement failed, using original query: {e}")
        
        # 0b. Intelligent model routing (if enabled)
        selected_model = self._get_model_for_query(query, context)
        if selected_model:
            context['selected_model'] = selected_model
        
        # 0. Check Cache (INTEGRATION POINT 2: Advanced Caching)
        cache_key = None
        if self.cache_manager:
            try:
                # Robust cache key generation
                # We use filename if available, otherwise just query (assuming same session/data)
                # Ideally, we should hash the dataframe content, but that's expensive.
                data_id = context.get('filename') or str(id(context.get('dataframe')))
                key_components = f"{query}_{data_id}_{selected_model or 'auto'}"
                cache_key = f"analysis_result_{hash(key_components)}"
                
                # Check cache unless forced refresh
                if not context.get('force_refresh', False):
                    cached_result = self.cache_manager.get_sync(cache_key)
                    if cached_result:
                        logging.info(f"⚡ Cache HIT for query: {query[:50]}...")
                        if 'metadata' in cached_result:
                            cached_result['metadata']['cached'] = True
                            cached_result['metadata']['cache_retrieval_time'] = 0.01 # Mock time
                        return cached_result
                else:
                    logging.info("🔄 Force refresh requested - skipping cache lookup")
            except Exception as e:
                logging.warning(f"Cache lookup failed: {e}")
                # Ensure we don't crash on cache failure
                pass

        # 0c. Self-Correction / CoT Review (INTEGRATION POINT 1)
        # Check if we should apply the "Two Friends" model before routing to a specific agent
        try:
            # Re-evaluate plan to get review level (since _get_model_for_query only returned model)
            # This is fast/heuristic so calling it twice is acceptable for safety
            plan = self.orchestrator.create_execution_plan(
                query=query,
                data=None, # AnalysisService usually doesn't have the DF loaded yet
                context=context
            )
            
            # Override review level if requested (UPGRADE FEATURE)
            if context.get('review_level'):
                from backend.core.engine.query_orchestrator import ReviewLevel
                try:
                    requested_level = str(context['review_level']).lower()
                    # Map simplified strings to enum values if needed
                    if requested_level == 'mandatory':
                        plan.review_level = ReviewLevel.MANDATORY
                        logging.info(f"✨ Forcing Review Level: MANDATORY (User Request)")
                    elif requested_level == 'optional':
                        plan.review_level = ReviewLevel.OPTIONAL
                        logging.info(f"✨ Forcing Review Level: OPTIONAL (User Request)")
                    elif requested_level == 'none':
                        plan.review_level = ReviewLevel.NONE
                        logging.info(f"✨ Forcing Review Level: NONE (User Request)")
                except Exception as e:
                    logging.warning(f"Failed to apply review_level override: {e}")
            
            # INTERCEPTION CRITERIA:
            # 1. Method is compatible (DIRECT_LLM, CODE_GENERATION, or AGENT_EXECUTION if forced)
            # 2. Review level is MANDATORY or OPTIONAL
            # 3. We have a valid cot_engine
            allowed_methods = ['direct_llm', 'code_generation', 'agent_execution', 'agent_tool_use']
            if (plan.execution_method.value in allowed_methods and 
                plan.review_level.value in ['mandatory', 'optional'] and
                self.cot_engine):
                
                logging.info(f"⚡ Intercepting with Self-Correction Engine (Level: {plan.review_level.value})")
                
                # We need to load data context for the engine
                df = None
                if context.get('filepath'):
                    try:
                        df = read_dataframe(context['filepath'], sample_size=1000)
                    except Exception as e:
                        logging.warning(f"Could not load DF for self-correction: {e}")

                # Prepare data context
                data_ctx = {
                    "rows": len(df) if df is not None else 0,
                    "columns": list(df.columns) if df is not None else [],
                    "data_types": str(df.dtypes) if df is not None else {},
                    "stats_summary": str(df.describe()) if df is not None else "No data loaded"
                }
                
                # Execute 'Two Friends' Loop
                result = self.cot_engine.run_correction_loop(
                    query=query,
                    data_context=data_ctx,
                    generator_model=plan.model,
                    critic_model="phi3:mini", # Critic is optimized for speed/reasoning
                    analysis_plan=plan
                )
                
                if result.success or len(result.final_output) > 50:
                    # Return result directly, masquerading as an agent result
                    # This ensures the API response structure is maintained
                    return {
                        "success": result.success,
                        "result": result.final_output,
                        "agent": "SelfCorrectionEngine",
                        "metadata": {
                            "iterations": result.total_iterations,
                            "reasoning": result.final_reasoning,
                            "review_level": plan.review_level.value,
                            "model": plan.model,
                            "execution_method": "direct_llm_with_review"
                        },
                        "interpretation": result.final_output, # CoT output is usually readable
                        "type": "analysis_result"
                    }
                    
                    # Cache the result if successful (INTEGRATION POINT 2)
                    if self.cache_manager and cache_key and final_result.get('success'):
                        try:
                            # Cache for 1 hour by default
                            self.cache_manager.put_sync(cache_key, final_result, ttl=3600, level='l2_optimized')
                        except Exception as e:
                            logging.warning(f"Failed to cache result: {e}")
                            
                    return final_result
                else:
                    logging.warning("Self-correction loop failed/empty, falling back to standard routing")

        except Exception as e:
            logging.error(f"Self-correction interception error: {e}")
            # Fallback to standard flow (Coexistence check)
            pass

        # 1. Route the query to the best agent
        topic, confidence, agent = self.registry.route_query(query, file_type=file_type)
        
        if not agent:
            # Fallback logic:
            # If we have a file, use DataAnalyst.
            # If not, execution fails unless we handle it here.
            if context.get('filename') or context.get('text_data'):
                logging.warning("No specific agent found, falling back to DataAnalyst")
                agent = self.registry.get_agent("DataAnalyst")
            else:
                # No agent + no data = General chat query?
                # We can't easily execute "Direct LLM" here as we need an Agent object.
                # But we can try to find a 'GeneralAgent' or return a specific error.
                logging.warning("No agent found and no data provided - attempting generic response")
                # Attempt to use a Generic/Chat agent if available, otherwise specific error
                agent = self.registry.get_agent("ChatAgent")
                
                if not agent:
                    # Final fallback: Create an ad-hoc DirectLLM execution
                    # For now, let's error gracefully instead of crashing DataAnalyst
                    return {
                        "success": True, # Technically managed
                        "result": "I'm not sure which agent to use for this. Please provide a file for analysis or ask a specific data question.",
                        "error": "No capable agent and no data provided",
                        "type": "response"
                    }
            
        if not agent:
            return {
                "success": False,
                "error": "No capable agent available for this request",
                "type": "error"
            }
            
        logging.info(f"Routed to agent: {agent.metadata.name} (confidence: {confidence})")
        
        # 2. Execute analysis
        try:
            # Check if agent has async execute method
            if hasattr(agent, 'execute_async'):
                logging.debug(f"Using async execution for {agent.metadata.name}")
                result = await agent.execute_async(query, **context)
            else:
                # Fallback to sync execution (run in thread pool to avoid blocking)
                import asyncio
                logging.debug(f"Using sync execution for {agent.metadata.name}")
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: agent.execute(query, **context))
            
            # 3. Generate human-readable interpretation if not provided
            interpretation = result.get("interpretation")
            if not interpretation or interpretation == "None" or len(str(interpretation).strip()) < 20:
                # Auto-generate interpretation from result
                raw_result = result.get("result", {})
                if isinstance(raw_result, dict):
                    interpretation = interpret_result(
                        result=raw_result,
                        query=query,
                        agent_name=agent.metadata.name
                    )
                elif isinstance(raw_result, str) and len(raw_result) > 50:
                    # Result is already a readable string
                    interpretation = raw_result
                else:
                    # Fallback interpretation
                    interpretation = f"Analysis completed by {agent.metadata.name}.\n\n{str(raw_result)}"
            
            # 4. Standardize response - always include interpretation
            response = {
                "success": result.get("success", False),
                "result": result.get("result"),
                "agent": agent.metadata.name,
                "metadata": result.get("metadata", {}),
                "interpretation": interpretation,  # Human-readable text for display
                "type": "analysis_result"
            }
            
            # Include routing info if used
            if selected_model:
                response["metadata"]["routed_model"] = selected_model
                
            if "error" in result:
                response["error"] = result["error"]
            
            # Cache the result if successful (INTEGRATION POINT 2 - Standard Path)
            if self.cache_manager and cache_key and response.get('success'):
                try:
                    # Cache for 1 hour by default
                    self.cache_manager.put_sync(cache_key, response, ttl=3600, level='l2_optimized')
                except Exception as e:
                    logging.warning(f"Failed to cache standard result: {e}")
                
            return response
            
        except Exception as e:
            logging.error(f"Analysis execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent": agent.metadata.name,
                "type": "error"
            }

# Thread-safe Singleton
_service_instance = None
_service_lock = threading.Lock()

def get_analysis_service():
    """Get or create the singleton AnalysisService instance (thread-safe)."""
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            # Double-check pattern for thread safety
            if _service_instance is None:
                _service_instance = AnalysisService()
    return _service_instance
