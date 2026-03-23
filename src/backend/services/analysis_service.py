"""Analysis Service — Nexus LLM Analytics
==========================================

High-level orchestrator (Service Layer) for analysis requests.
Routes queries to specialised agents via the Plugin Registry,
with intelligent model routing, CoT self-correction, caching,
and comprehensive result interpretation.

Classes
-------
AnalysisService
    Entry-point for all analysis operations; delegates to
    :class:`QueryOrchestrator`, agent plugins, and the
    :class:`EnhancedCacheManager`.

v2.0 Enterprise Additions
-------------------------
* :class:`AnalysisServiceMetrics` — tracks request counts,
  cache-hit rates, and per-agent routing statistics.
* :func:`get_analysis_service` already exists (documented).
"""
# Analysis Service
# High-level orchestrator for analysis requests (Service Layer)

from __future__ import annotations

import asyncio
import gc
import logging
import threading
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add src to path if needed (for direct execution)
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.plugin_system import get_agent_registry
from backend.core.engine.user_preferences import get_preferences_manager
from backend.io.result_interpreter import interpret_result
from backend.core.semantic_mapper import get_semantic_mapper
from backend.core.engine.query_orchestrator import ReviewLevel
from backend.agents.model_manager import get_model_manager
from backend.utils.data_utils import DataPathResolver, read_dataframe
from backend.core.dataframe_store import get_dataframe_store
from backend.core.enhanced_cache_integration import get_enhanced_cache_manager
from backend.core.dynamic_planner import get_dynamic_planner, AnalysisPlan, AnalysisStep

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service layer for handling analysis requests.

    Routes requests to appropriate agents via the Plugin Registry.
    Supports intelligent model routing, caching, and CoT
    self-correction when enabled.  Replaces legacy CrewManager
    and AnalysisExecutor.

    Attributes:
        registry: Plugin agent registry for agent discovery.

    Thread Safety:
        Access via :func:`get_analysis_service` singleton.
        Individual analysis calls are not re-entrant.
    """
    
    def __init__(self):
        from backend.core.analysis_manager import get_analysis_manager
        self.registry = get_agent_registry()
        self.analysis_manager = get_analysis_manager()
        self._orchestrator = None
        # Ensure plugins are loaded
        # self.registry.discover_agents() # This is usually done on import or app startup
        logger.info("AnalysisService initialized")
    
    @property
    def orchestrator(self):
        """Get query orchestrator (singleton)"""
        if self._orchestrator is None:
            from backend.core.engine.query_orchestrator import get_query_orchestrator
            self._orchestrator = get_query_orchestrator()
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
                logger.info("SelfCorrectionEngine initialized")
            except Exception as e:
                logger.warning("Failed to initialize SelfCorrectionEngine: %s", e)
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
        Reuses pre-computed execution plan from streaming context if available.
        """
        try:
            # Check if plan was already computed by the streaming endpoint
            if context.get('execution_plan'):
                plan = context['execution_plan']
                logger.info("🧠 Reusing pre-computed plan: %s (avoids double orchestrator run)", plan.model)
                return plan.model
            
            # Get LLM client for semantic routing
            manager = get_model_manager()
            llm_client = manager.llm_client
            
            # Create execution plan using QueryOrchestrator with semantic routing
            plan = self.orchestrator.create_execution_plan(
                query=query,
                data=context.get('dataframe'),
                context=context,
                llm_client=llm_client
            )
            
            # Log the brain's reasoning
            logger.info("🧠 QueryOrchestrator Decision: %s", plan.model)
            logger.info("   Reasoning: %s", plan.reasoning)
            if plan.user_override:
                logger.info("   ⚠️ USER OVERRIDE: User explicitly chose this model")
            
            return plan.model
            
        except Exception as e:
            logger.warning("QueryOrchestrator failed, using default: %s", e)
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
        # Start lifecycle tracking (Integration Point 3)
        analysis_id = self.analysis_manager.start_analysis(user_session=user_id or "anonymous")
        self.analysis_manager.update_analysis_stage(analysis_id, "initializing")
        
        context = context or {}
        context['analysis_id'] = analysis_id # Inject ID into context for agents
        
        filename = context.get('filename')
        file_type = Path(filename).suffix if filename else None
        
        # Resolve filepath if not already provided
        if filename and not context.get('filepath'):
            resolved_path = DataPathResolver.resolve_data_file(filename)
            if resolved_path:
                context['filepath'] = str(resolved_path)
        
        # [OPTIMIZATION 1.1] Unified DataFrame Loading via Store
        # Load once, reuse everywhere (Mapping, Routing, Self-Correction, Planning)
        # Skip for document files (PDF, DOCX, etc.) — they cannot be loaded as DataFrames
        _DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.rtf'}
        _skip_df_load = file_type and file_type.lower() in _DOCUMENT_EXTENSIONS
        if context.get('filepath') and context.get('dataframe') is None and not _skip_df_load:
            try:
                _store = get_dataframe_store()
                # Load with default sample size (4500) to ensure sufficient data for all downstream tasks
                context['dataframe'] = _store.get_or_load(
                    context['filepath'],
                    loader=lambda: read_dataframe(context['filepath']) 
                )
                logger.debug("Loaded dataframe from store for analysis context")
            except Exception as e:
                logger.warning("Failed to eager-load dataframe from store: %s", e)
        
        logger.info("AnalysisService received query: %s (file: %s)", query, filename)
        
        # PATENT COMPLIANCE: Cross-session vector memory retrieval (Claim 1e).
        # Retrieve semantically similar past analyses so the pipeline can
        # reuse insights or avoid repeating past mistakes.
        try:
            from backend.agents.model_manager import get_model_manager as _gmm
            _chroma = _gmm().chroma_client
            if _chroma:
                past_analyses = _chroma.semantic_search_history(query, n_results=3)
                if past_analyses:
                    context['cross_session_context'] = past_analyses
                    logger.info("📚 Cross-session memory: retrieved %d related past analyses", len(past_analyses))
        except Exception as e:
            logger.debug("Cross-session retrieval skipped: %s", e)

        # 0a. Apply semantic layer to enhance query
        df_for_mapping = context.get('dataframe')
        # (Redundant loading removed - already loaded above)

        # Keep original query for routing — semantic annotations (e.g. "[Column
        # Concepts: count: ...]") can contain words that mislead agent scoring.
        original_query = query

        if df_for_mapping is not None:
            try:
                mapper = get_semantic_mapper()
                enhanced_query = mapper.enhance_query_context(query, df_for_mapping)
                if enhanced_query != query:
                    logger.info("🔍 Query enhanced with semantic concepts")
                    query = enhanced_query  # Use enhanced version
            except Exception as e:
                logger.warning("Semantic enhancement failed, using original query: %s", e)
        
        # 0b. Intelligent model routing (if enabled)
        self.analysis_manager.update_analysis_stage(analysis_id, "routing")
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
                        logger.info("⚡ Cache HIT for query: %s...", query[:50])
                        if 'metadata' in cached_result:
                            cached_result['metadata']['cached'] = True
                            cached_result['metadata']['cache_retrieval_time'] = 0.01 # Mock time
                            
                        self.analysis_manager.complete_analysis(analysis_id)
                        return cached_result
                else:
                    logger.info("🔄 Force refresh requested - skipping cache lookup")
            except Exception as e:
                logger.warning("Cache lookup failed: %s", e)
                # Ensure we don't crash on cache failure
                pass

        # 0c. Self-Correction / CoT Review (INTEGRATION POINT 1)
        # Check if we should apply the "Two Friends" model before routing to a specific agent
        try:
            # Reuse the execution plan if already computed (avoid duplicate LLM call)
            plan = context.get('execution_plan')
            if not plan:
                manager = get_model_manager()
                llm_client = manager.llm_client
                plan = self.orchestrator.create_execution_plan(
                    query=query,
                    data=None,
                    context=context,
                    llm_client=llm_client
                )
            
            # Override review level if requested (UPGRADE FEATURE)
            if context.get('review_level'):
                try:
                    requested_level = str(context['review_level']).lower()
                    # Map simplified strings to enum values if needed
                    if requested_level == 'mandatory':
                        plan.review_level = ReviewLevel.MANDATORY
                        logger.info("✨ Forcing Review Level: MANDATORY (User Request)")
                    elif requested_level == 'optional':
                        plan.review_level = ReviewLevel.OPTIONAL
                        logger.info("✨ Forcing Review Level: OPTIONAL (User Request)")
                    elif requested_level == 'none':
                        plan.review_level = ReviewLevel.NONE
                        logger.info("✨ Forcing Review Level: NONE (User Request)")
                except Exception as e:
                    logger.warning("Failed to apply review_level override: %s", e)
            
            # INTERCEPTION CRITERIA:
            # 1. Method is DIRECT_LLM only — code_generation and agent_execution
            #    MUST flow through to the real agent pipeline which can actually
            #    load data, generate pandas code, and execute it in a sandbox.
            #    The SelfCorrectionEngine is a pure LLM reasoner — it CANNOT run code,
            #    so letting it intercept code_generation queries causes hallucinated numbers.
            # 2. Review level is MANDATORY or OPTIONAL
            # 3. We have a valid cot_engine
            # 4. Skip for simple queries (low complexity) — huge performance win
            # 5. CRITICAL: Never intercept when a data file is present — the agent
            #    pipeline can actually execute pandas code against real data, whereas
            #    SelfCorrectionEngine can only hallucinate numbers from summary stats.
            complexity = getattr(plan, 'complexity_score', None) or 0
            has_data_file = bool(context.get('filepath') or context.get('filename'))
            cot_compatible_methods = ['direct_llm']  # ONLY direct LLM — not code_gen!
            should_review = (
                plan.execution_method.value in cot_compatible_methods and 
                plan.review_level.value in ['mandatory', 'optional'] and
                self.cot_engine and
                complexity >= 0.4 and  # Skip self-correction for simple queries
                not has_data_file  # Never intercept data-file queries — agent does real code exec
            )
            if should_review:
                
                logger.info("⚡ Intercepting with Self-Correction Engine (Level: %s)", plan.review_level.value)
                
                # We need to load data context for the engine
                df = context.get('dataframe')  
                # (Redundant loading removed - already loaded above)

                # Prepare data context
                data_ctx = {
                    "rows": len(df) if df is not None else 0,
                    "columns": list(df.columns) if df is not None else [],
                    "data_types": str(df.dtypes) if df is not None else {},
                    "stats_summary": str(df.describe()) if df is not None else "No data loaded"
                }
                
                # Execute 'Two Friends' Loop
                # Use the smallest available model as critic (fast, any installation)
                try:
                    critic_model = self.orchestrator.model_simple  # Dynamically discovered smallest model
                except AttributeError:
                    critic_model = plan.model  # Fallback: same model as generator
                
                result = self.cot_engine.run_correction_loop(
                    query=query,
                    data_context=data_ctx,
                    generator_model=plan.model,
                    critic_model=critic_model,
                    analysis_plan=plan
                )
                
                if result.success:
                    # Return result directly, masquerading as an agent result
                    # This ensures the API response structure is maintained
                    # NOTE: Only return on success — failed/rambling output must
                    #       fall through to the real agent pipeline.
                    final_result = {
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
                    # FIX: Moved BEFORE return to prevent unreachable code
                    if self.cache_manager and cache_key and final_result.get('success'):
                        try:
                            # Cache for 1 hour by default
                            self.cache_manager.put_sync(cache_key, final_result, ttl=3600, level='l2_optimized')
                        except Exception as e:
                            logger.warning("Failed to cache result: %s", e)
                            
                    return final_result
                else:
                    logger.warning("Self-correction loop failed/empty, falling back to standard routing")

        except Exception as e:
            logger.error("Self-correction interception error: %s", e, exc_info=True)
            # Fallback to standard flow (Coexistence check)
            pass
            
        # Update stage before main planning
        self.analysis_manager.update_analysis_stage(analysis_id, "planning")

        # Share execution plan with agents (avoids duplicate LLM calls for planning)
        if plan and 'execution_plan' not in context:
            context['execution_plan'] = plan

        # 0d. PATENT COMPLIANCE: Always generate an analysis plan record
        # The patent requires the Planner to decompose ALL queries into subtasks.
        # For simple queries we generate a lightweight plan; for complex ones the
        # DynamicPlanner is invoked with full LLM reasoning.
        try:
            complexity = getattr(plan, 'complexity_score', 0) if plan else 0
            if complexity >= 0.5 and context.get('filepath'):
                # Complex query with data — invoke full DynamicPlanner
                planner = get_dynamic_planner()
                try:
                    df_preview = ""
                    if context.get('dataframe') is not None:
                        _df = context['dataframe']
                        # Use first 3 rows for preview
                        df_preview = f"Columns: {list(_df.columns)}\nShape: {_df.shape}\nSample:\n{_df.head(3).to_string()}"
                    analysis_plan = planner.create_plan(query, df_preview, model=getattr(plan, 'model', None))
                except Exception as pe:
                    logger.warning("DynamicPlanner call failed, using lightweight plan: %s", pe)
                    analysis_plan = AnalysisPlan(
                        domain="General", summary=f"Analyze: {query[:80]}",
                        steps=[AnalysisStep(1, query[:120], "auto", "Direct routing")],
                        confidence=0.6
                    )
            else:
                # Simple / no-data query — lightweight plan record for audit trail
                analysis_plan = AnalysisPlan(
                    domain="General",
                    summary=f"Direct analysis: {query[:80]}",
                    steps=[AnalysisStep(1, query[:120], "direct_llm" if not context.get('filepath') else "agent_execution", "Low complexity — single-step execution")],
                    confidence=0.8
                )
            context['analysis_plan'] = analysis_plan
            logger.info("📋 Analysis plan: domain=%s, steps=%d, confidence=%.2f",
                        analysis_plan.domain, len(analysis_plan.steps), analysis_plan.confidence)
        except Exception as e:
            logger.warning("Plan generation failed (non-blocking): %s", e)

        # 1. Route the query to the best agent
        # When we have a data file AND the plan requires code generation,
        # always prefer DataAnalyst — it has the most robust code_gen + sandbox
        # execution pipeline.  Other specialist agents (Financial, Statistical, etc.)
        # can still be reached for direct_llm/deterministic queries.
        # EXCEPTION: Document files (PDF, DOCX, PPTX, RTF) are unstructured and
        # cannot be loaded into pandas — route them to RagAgent instead.
        from backend.core.engine.query_orchestrator import ExecutionMethod
        _DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.rtf'}  # Unstructured doc types
        _is_document_file = file_type and file_type.lower() in _DOCUMENT_EXTENSIONS
        if (context.get('filename') and plan and
                getattr(plan, 'execution_method', None) == ExecutionMethod.CODE_GENERATION
                and not _is_document_file):
            agent = self.registry.get_agent("DataAnalyst")
            if agent:
                topic, confidence = "data_analysis", 1.0
                logger.info("📊 File + code_generation plan → routing to DataAnalyst")
            else:
                topic, confidence, agent = self.registry.route_query(original_query, file_type=file_type)
        elif _is_document_file:
            # Document files → prefer RagAgent which can extract and reason over text
            agent = self.registry.get_agent("RagAgent")
            if agent:
                topic, confidence = "document_processing", 0.95
                logger.info("📄 Document file (%s) → routing to RagAgent", file_type)
            else:
                topic, confidence, agent = self.registry.route_query(original_query, file_type=file_type)
        else:
            topic, confidence, agent = self.registry.route_query(original_query, file_type=file_type)
        
        if not agent:
            # Fallback logic:
            # If we have a file, use DataAnalyst.
            # If not, execution fails unless we handle it here.
            if context.get('filename') or context.get('text_data'):
                logger.warning("No specific agent found, falling back to DataAnalyst")
                agent = self.registry.get_agent("DataAnalyst")
            else:
                # No agent + no data = General chat query?
                logger.warning("No agent found and no data provided - attempting generic response")
                agent = self.registry.get_agent("ChatAgent")
                
                if not agent:
                    # Try direct online LLM call if in online mode
                    try:
                        from backend.core.mode_manager import get_mode_manager
                        _mm = get_mode_manager()
                        if _mm.get_mode() == "online":
                            # Build fallback chain: Groq → OpenRouter
                            _clients = []
                            _g = _mm._get_groq()
                            if _g:
                                _clients.append((_g, "Groq"))
                            _o = _mm._get_openrouter()
                            if _o:
                                _clients.append((_o, "OpenRouter"))
                            
                            for _client, _name in _clients:
                                try:
                                    logger.info("Using %s directly for no-data query", _name)
                                    llm_response = _client.generate(query, tier="medium")
                                    self.analysis_manager.complete_analysis(analysis_id)
                                    return {
                                        "success": True,
                                        "result": llm_response,
                                        "agent": f"DirectOnlineLLM:{_name}",
                                        "interpretation": llm_response,
                                        "type": "analysis_result",
                                        "analysis_id": analysis_id
                                    }
                                except Exception as _e:
                                    logger.warning("Direct %s failed: %s, trying next...", _name, _e)
                    except Exception as online_err:
                        logger.warning("Direct online LLM fallback failed: %s", online_err)
                    
                    # Offline mode or online LLM failed — return graceful message
                    return {
                        "success": True,
                        "result": "I'm not sure which agent to use for this. Please provide a file for analysis or ask a specific data question.",
                        "error": "No capable agent and no data provided",
                        "type": "response"
                    }
            
        if not agent:
            error_msg = "No capable agent available for this request"
            self.analysis_manager.fail_analysis(analysis_id, error_msg)
            return {
                "success": False,
                "error": error_msg,
                "type": "error"
            }
            
        logger.info("Routed to agent: %s (confidence: %s)", agent.metadata.name, confidence)
        self.analysis_manager.update_analysis_stage(analysis_id, f"executing:{agent.metadata.name}")
        
        # 2. Execute analysis
        try:
            # Check if agent has async execute method
            if hasattr(agent, 'execute_async'):
                logger.debug("Using async execution for %s", agent.metadata.name)
                result = await agent.execute_async(query, **context)
            else:
                # Fallback to sync execution (run in thread pool to avoid blocking)
                # USE execute_with_logging for backend visibility
                logger.debug("Using sync execution for %s", agent.metadata.name)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: agent.execute_with_logging(query, **context))
            
            # 2b. AGENT FAILURE RETRY: If the routed agent failed, try DataAnalyst
            # as a fallback before giving up (e.g. SQLAgent table-not-found → DataAnalyst code-gen)
            if (not result.get("success") and agent.metadata.name != "DataAnalyst"
                    and context.get('filename')):
                fallback_agent = self.registry.get_agent("DataAnalyst")
                if fallback_agent:
                    original_error = result.get("error", "unknown")
                    logger.warning(
                        "⚠️ %s failed (%s) — retrying with DataAnalyst",
                        agent.metadata.name, original_error[:120]
                    )
                    self.analysis_manager.update_analysis_stage(analysis_id, "executing:DataAnalyst")
                    try:
                        if hasattr(fallback_agent, 'execute_async'):
                            result = await fallback_agent.execute_async(query, **context)
                        else:
                            result = await loop.run_in_executor(
                                None, lambda: fallback_agent.execute_with_logging(query, **context)
                            )
                        if result.get("success"):
                            logger.info("✅ DataAnalyst fallback succeeded after %s failure", agent.metadata.name)
                            result.setdefault("metadata", {})["fallback_from"] = agent.metadata.name
                            agent = fallback_agent  # Update agent ref for response
                    except Exception as fb_err:
                        logger.warning("DataAnalyst fallback also failed: %s", fb_err)

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
                    logger.warning("Failed to cache standard result: %s", e)

            # PATENT COMPLIANCE: Index completed analysis into ChromaDB vector
            # memory for semantic cross-session retrieval (Claim 1e).
            if response.get('success'):
                try:
                    manager = get_model_manager()
                    chroma = manager.chroma_client
                    if chroma:
                        import hashlib as _hl
                        _doc_id = _hl.md5(f"{query}_{filename or ''}".encode()).hexdigest()[:16]
                        _summary = str(response.get('interpretation', ''))[:1500]
                        _agent_name = response.get('agent', 'unknown')
                        chroma.add_or_update(
                            collection_name="analysis_history",
                            ids=[f"analysis_{_doc_id}"],
                            documents=[f"Query: {query}\nAgent: {_agent_name}\nResult: {_summary}"],
                            metadatas=[{"query": query[:500], "agent": _agent_name,
                                        "filename": filename or "", "success": "true"}]
                        )
                except Exception as e:
                    logger.debug("Cross-session vector indexing skipped: %s", e)

            # Complete lifecycle
            if response.get('success'):
                self.analysis_manager.complete_analysis(analysis_id)
            else:
                self.analysis_manager.fail_analysis(analysis_id, response.get('error', 'Unknown error'))
                
            # Add ID to response
            response['analysis_id'] = analysis_id
            
            # RAM OPTIMIZATION: Force garbage collection after analysis
            # Clear DataFrame from context to prevent memory leaks
            if 'dataframe' in context:
                del context['dataframe']
            gc.collect()
            
            return response
            
        except Exception as e:
            logger.error("Analysis execution failed: %s", e, exc_info=True)
            self.analysis_manager.fail_analysis(analysis_id, str(e))
            
            # RAM OPTIMIZATION: Clean up on exception as well
            if 'dataframe' in context:
                del context['dataframe']
            gc.collect()
            
            return {
                "success": False,
                "error": str(e),
                "agent": agent.metadata.name,
                "type": "error",
                "analysis_id": analysis_id
            }

# Thread-safe Singleton
_service_instance = None
_service_lock = threading.Lock()

def get_analysis_service() -> AnalysisService:
    """Get or create the singleton AnalysisService instance (thread-safe)."""
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            # Double-check pattern for thread safety
            if _service_instance is None:
                _service_instance = AnalysisService()
    return _service_instance


# =====================================================================
# v2.0 Enterprise Additions — appended; all v1.x code is unchanged
# =====================================================================

from dataclasses import dataclass, field
from collections import Counter


@dataclass
class AnalysisServiceMetrics:
    """Tracks request counts, cache-hit rates, and per-agent routing.

    Attributes:
        total_requests: Number of ``analyze()`` invocations.
        cache_hits: Requests served from cache.
        cache_misses: Requests requiring fresh computation.
        agent_routing: Counter mapping agent name → call count.
        total_latency_ms: Cumulative processing time.

    v2.0 Enterprise Addition.
    """

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    agent_routing: Counter = field(default_factory=Counter)
    total_latency_ms: float = 0.0

    def record(self, *, agent: str = "", cached: bool = False,
               latency_ms: float = 0.0) -> None:
        """Record a single analysis request."""
        self.total_requests += 1
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        if agent:
            self.agent_routing[agent] += 1
        self.total_latency_ms += latency_ms

    def to_dict(self) -> dict:
        """Return a JSON-serialisable snapshot."""
        return {
            "total_requests": self.total_requests,
            "cache_hit_rate": round(
                self.cache_hits / self.total_requests, 4
            ) if self.total_requests else 0.0,
            "agent_routing": dict(self.agent_routing),
            "avg_latency_ms": round(
                self.total_latency_ms / self.total_requests, 2
            ) if self.total_requests else 0.0,
        }
