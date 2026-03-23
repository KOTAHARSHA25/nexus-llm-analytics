"""Analysis API — Primary Query Processing Endpoints
====================================================

Exposes the core ``/analyze`` endpoints that accept natural-language queries,
route them through the multi-agent pipeline, and return structured results.

Endpoints
---------
``POST /``
    Synchronous analysis — returns a single JSON response when complete.
``POST /stream``
    Server-Sent Events (SSE) streaming — emits progress, plan, tokens, and
    final result as discrete events for real-time frontend updates.
``POST /cancel/{analysis_id}``
    Cancel a running analysis by its tracking ID.
``GET  /status/{analysis_id}``
    Poll the lifecycle status of an in-flight analysis.
``GET  /running``
    List all currently executing analyses.
``POST /review-insights``
    Secondary-model review of prior analysis results.
``GET  /routing-stats``
    Retrieve intelligent routing statistics from the orchestrator.

Dependencies
------------
- ``backend.services.analysis_service`` — AnalysisService singleton
- ``backend.core.analysis_manager`` — In-flight analysis lifecycle tracker
- ``backend.agents.model_manager`` — LLM model manager (used by streaming)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.core.analysis_manager import analysis_manager, check_cancellation
from backend.services.analysis_service import get_analysis_service

logger = logging.getLogger(__name__)
router = APIRouter()


class StreamEvent(BaseModel):
    """Strict type definition for SSE stream events.

    Each field maps to a discrete stage in the analysis pipeline so the
    frontend can render granular progress feedback.

    Attributes:
        step:     Lifecycle stage name (``init``, ``validation``, ``loading``,
                  ``routing``, ``mode``, ``thinking``, ``token``,
                  ``agent_start``, ``agent_complete``, ``formatting``,
                  ``complete``, ``error``, ``plan``).
        message:  Human-readable status text for the current step.
        progress: Percentage completion (0.0 – 100.0).
        token:    Incremental text chunk during streaming.
        error:    Error description when ``step == 'error'``.
        result:   Full result payload when ``step == 'complete'``.
        files:    Filenames involved in the current analysis.
        plan:     Execution plan details when ``step == 'plan'``.
    """

    step: Literal[
        'init', 'validation', 'loading', 'routing', 'mode',
        'thinking', 'token', 'agent_start', 'agent_complete',
        'formatting', 'complete', 'error', 'plan',
    ]
    message: Optional[str] = None
    progress: Optional[float] = 0.0
    token: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    files: Optional[List[str]] = None
    plan: Optional[Dict[str, Any]] = None


def _format_dict_result(data: Dict[str, Any], indent: int = 0) -> str:
    """Convert a raw dict result into a human-readable analysis summary.

    Recursively walks the dictionary, formatting keys as Markdown-bold
    headers and values as bullet points with intelligent number formatting.

    Args:
        data:   The raw analysis result dictionary.
        indent: Current nesting depth (used for recursive indentation).

    Returns:
        A Markdown-ish string ready for frontend rendering.
    """
    lines = []
    prefix = "  " * indent
    
    for key, value in data.items():
        # Skip internal/system keys
        if key.startswith('_') or key in ['success', 'agent', 'operation']:
            continue
            
        # Format the key nicely
        display_key = key.replace('_', ' ').title()
        
        if isinstance(value, dict):
            lines.append(f"{prefix}**{display_key}:**")
            lines.append(_format_dict_result(value, indent + 1))
        elif isinstance(value, list):
            if len(value) == 0:
                lines.append(f"{prefix}• {display_key}: (empty)")
            elif len(value) <= 5:
                lines.append(f"{prefix}• {display_key}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"{prefix}• {display_key}: {', '.join(str(v) for v in value[:5])}... (+{len(value)-5} more)")
        elif isinstance(value, (int, float)):
            # Format numbers nicely
            if isinstance(value, float):
                if abs(value) >= 1000:
                    lines.append(f"{prefix}• {display_key}: {value:,.2f}")
                else:
                    lines.append(f"{prefix}• {display_key}: {value:.4g}")
            else:
                lines.append(f"{prefix}• {display_key}: {value:,}")
        elif value is None:
            continue  # Skip None values
        else:
            lines.append(f"{prefix}• {display_key}: {value}")
    
    return "\n".join(lines) if lines else "Analysis completed."


class AnalyzeRequest(BaseModel):
    """Request model for data analysis queries."""
    query: str = Field(..., description="The natural language query for data analysis")
    filename: Optional[str] = Field(None, description="Single file to analyze (backward compatible)")
    filenames: Optional[List[str]] = Field(None, description="Multiple files to analyze (supports multi-file joins)")
    text_data: Optional[str] = Field(None, description="Direct text input for analysis without file upload")
    column: Optional[str] = Field(None, description="Specific column to filter on")
    value: Optional[str] = Field(None, description="Value to filter the column by")
    session_id: Optional[str] = Field(None, description="Session identifier for tracking related queries")
    force_refresh: Optional[bool] = Field(False, description="Bypass cache and force fresh analysis")
    review_level: Optional[str] = Field(None, description="Force review level: none, optional, mandatory")
    preferred_plugin: Optional[str] = Field(None, description="Preferred agent/plugin to use for analysis")
    max_retries: Optional[int] = Field(2, description="Maximum number of retry attempts")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Show me the top 10 customers by total purchase amount",
                "filename": "sales_data.csv"
            }
        }
    }


class AnalyzeResponse(BaseModel):
    """Response model for data analysis results."""
    result: Optional[str] = Field(None, description="The analysis result")
    visualization: Optional[Dict[str, Any]] = Field(None, description="Generated visualization data")
    code: Optional[str] = Field(None, description="Generated/executed code for the analysis")
    generated_code: Optional[str] = Field(None, description="Original LLM-generated code before cleaning")
    execution_id: Optional[str] = Field(None, description="Unique ID for code execution history lookup")
    execution_time: float = Field(0, description="Time taken for analysis in seconds")
    query: str = Field(..., description="The original query")
    filename: Optional[str] = Field(None, description="File analyzed")
    filenames: Optional[List[str]] = Field(None, description="Files analyzed (multi-file)")
    analysis_id: str = Field(..., description="Unique identifier for this analysis")
    status: str = Field(..., description="Status of the analysis: success, error, cancelled")
    error: Optional[str] = Field(None, description="Error message if analysis failed")
    execution_method: Optional[str] = Field(None, description="How the query was executed: code_generation, direct_llm, etc.")
    agent: Optional[str] = Field(None, description="Agent that executed the request")
    model: Optional[str] = Field(None, description="LLM model used for this analysis")
    token_count: Optional[int] = Field(None, description="Number of tokens streamed")
    agent_used: Optional[bool] = Field(None, description="Whether an agent handled the query")
    plan: Optional[Dict[str, Any]] = Field(None, description="Execution plan details (model, reasoning, complexity)")


@router.post("/", response_model=AnalyzeResponse, responses={
    200: {"description": "Analysis completed successfully"},
    400: {"description": "Invalid request parameters"},
    503: {"description": "AI service unavailable"}
})
async def analyze_query(request: AnalyzeRequest) -> Dict[str, Any]:
    """Execute a synchronous natural-language analysis query.

    Routes the query through the multi-agent pipeline (data loading,
    semantic routing, code generation, execution) and returns the
    complete result in a single response.

    Args:
        request: :class:`AnalyzeRequest` with query text, optional
            filename(s), text data, and execution preferences.

    Returns:
        Dict conforming to :class:`AnalyzeResponse` — ``result`` text,
        ``visualization``, ``code``, ``execution_time``, ``status``, etc.

    Raises:
        HTTPException 503: If the AI service cannot be initialised.
    """
    # Get singleton AnalysisService instance
    try:
        service = get_analysis_service()
    except Exception as e:
        logger.error("Failed to initialize AnalysisService: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Failed to initialize AI system: {e}")
    
    # Start analysis tracking
    analysis_id = analysis_manager.start_analysis(request.session_id)
    
    logger.info(
        "[ANALYZE] Started analysis %s: query=%s, filename=%s, filenames=%s, has_text=%s",
        analysis_id, request.query[:80], request.filename, request.filenames, bool(request.text_data),
    )
    
    try:
        # Input validation - support both single and multiple files
        if not request.query:
            analysis_manager.complete_analysis(analysis_id)
            return {"error": "Query is required", "status": "error", "query": "", "analysis_id": analysis_id}
        
        # Determine files to analyze
        files = None
        if request.filenames:
            files = request.filenames
            if not files:
                analysis_manager.complete_analysis(analysis_id)
                return {"error": "At least one filename required", "status": "error", "query": request.query, "analysis_id": analysis_id}
        elif request.filename:
            files = [request.filename]
        elif request.text_data:
            # Handle text input directly via service (text_data in context)
            files = None
        else:
            # Allow logic/math/code queries without files
            files = None
        
        # Check if cancelled before starting
        check_cancellation(analysis_id)
        
        # Sanitize inputs
        query = request.query.strip()[:1000]  # Limit query length
        if files:
            files = [f.strip() for f in files]
        
        analysis_manager.update_analysis_stage(analysis_id, "processing")
        
        # Prepare context for AnalysisService
        context = {
            "filename": files[0] if files else None,
            "filenames": files,
            "text_data": request.text_data,
            "column": request.column,
            "value": request.value,
            "analysis_id": analysis_id,
            "force_refresh": request.force_refresh,
            "review_level": request.review_level
        }
        
        # Execute analysis using AnalysisService
        result = await service.analyze(query=query, context=context)
        
        # Check if cancelled before returning results
        check_cancellation(analysis_id)
        
        analysis_manager.complete_analysis(analysis_id)
        
        # Return successful analysis result
        # Note: AnalysisService returns a standardized dict, we map it to AnalyzeResponse
        # CRITICAL: result MUST be a string for FastAPI response validation
        raw_result = result.get("result", "")
        
        # If result is a dict (from statistical/descriptive agents), convert to readable text
        if isinstance(raw_result, dict):
            # Priority: Use interpretation (readable text) if available
            interpretation = result.get("interpretation")
            if interpretation:
                result_str = interpretation
            else:
                # Fallback: Format the dict nicely for user
                result_str = _format_dict_result(raw_result)
            logger.info("Formatted dict result (len=%d)", len(result_str))
        else:
            result_str = str(raw_result)
        
        return {
            "result": result_str,
            "visualization": result.get("metadata", {}).get("visualization"),
            "code": result.get("metadata", {}).get("code") or result.get("metadata", {}).get("executed_code"),
            "generated_code": result.get("metadata", {}).get("generated_code"),
            "execution_id": result.get("metadata", {}).get("execution_id"),
            "execution_time": result.get("metadata", {}).get("execution_time", 0),
            "execution_method": result.get("metadata", {}).get("execution_method"),
            "query": query,
            "filename": files[0] if files and len(files) == 1 else None,  # Backward compatibility
            "filenames": files,  # Multi-file support
            "analysis_id": analysis_id,
            "status": "success" if result.get("success") else "error",
            "error": result.get("error"),
            "agent": result.get("agent")
        }
    except HTTPException as e:
        # Analysis was cancelled
        analysis_manager.complete_analysis(analysis_id)
        raise e
    except Exception as e:
        logger.error("[ANALYZE] Error in analysis %s: %s", analysis_id, e, exc_info=True)
        analysis_manager.complete_analysis(analysis_id)
        return {
            "error": f"Analysis failed: {str(e)}",
            "status": "error", 
            "analysis_id": analysis_id,
            "retry_possible": True,
            "query": request.query
        }


@router.post("/stream", responses={
    200: {"description": "Streaming analysis with progress updates"},
    400: {"description": "Invalid request parameters"},
    503: {"description": "AI service unavailable"}
})
async def analyze_stream(request: AnalyzeRequest):
    """
    Streaming analysis endpoint using Server-Sent Events (SSE).
    Returns real-time progress updates as the analysis proceeds.
    
    This provides better UX by showing users what's happening instead of a blank loading screen.
    """
    async def generate_updates():
        analysis_id = None
        try:
            # Step 1: Initialization
            event = StreamEvent(step='init', message='Initializing analysis...', progress=0)
            yield f"data: {event.model_dump_json()}\n\n"
            
            # Get service
            service = get_analysis_service()
            analysis_id = analysis_manager.start_analysis(request.session_id)
            
            # Step 2: Validation
            event = StreamEvent(step='validation', message='Validating request...', progress=10)
            yield f"data: {event.model_dump_json()}\n\n"
            
            # Validate inputs (same as non-streaming endpoint)
            if not request.query:
                event = StreamEvent(step='error', message='Query is required', error='Query is required')
                yield f"data: {event.model_dump_json()}\n\n"
                return
            
            # Determine files
            files = None
            if request.filenames:
                files = request.filenames
            elif request.filename:
                files = [request.filename]
            elif request.text_data:
                files = None
            else:
                # Allow queries without files (logic/math/code queries)
                files = None
            
            # Step 3: Loading data
            event = StreamEvent(step='loading', message='Loading data file(s)...', progress=30, files=files)
            yield f"data: {event.model_dump_json()}\n\n"
            
            # Prepare context
            context = {
                "filename": files[0] if files else None,
                "filenames": files,
                "text_data": request.text_data,
                "column": request.column,
                "value": request.value,
                "analysis_id": analysis_id,
                "force_refresh": request.force_refresh,
                "review_level": request.review_level
            }
            
            # Step 4: Route to appropriate execution path
            event = StreamEvent(step='routing', message='Determining analysis method...', progress=35)
            yield f"data: {event.model_dump_json()}\n\n"
            
            # [NEW] Get LLM client for streaming and routing
            from backend.agents.model_manager import get_model_manager
            from backend.core.engine.query_orchestrator import ExecutionMethod
            manager = get_model_manager()
            # In online mode Ollama is not required — safe_llm_client returns None gracefully
            # in both online mode AND offline mode when Ollama is unavailable (no crash)
            llm_client = manager.safe_llm_client
            
            # Use orchestrator to create execution plan with semantic routing
            plan = None
            plan_info = None
            try:
                plan = service.orchestrator.create_execution_plan(
                    query=request.query,
                    data=None,
                    context=context,
                    llm_client=llm_client  # Enable semantic routing
                )
                selected_model = plan.model
                execution_method = plan.execution_method
                plan_info = {
                    "model": plan.model,
                    "execution_method": plan.execution_method.value,
                    "reasoning": plan.reasoning,
                    "review_level": plan.review_level.value if hasattr(plan, 'review_level') else "none",
                    "complexity_score": getattr(plan, 'complexity_score', None),
                }
                logger.info("Execution Plan: %s", plan.reasoning)
            except Exception as e:
                logger.warning("Orchestrator failed, using defaults: %s", e, exc_info=True)
                selected_model = None
                execution_method = ExecutionMethod.DIRECT_LLM
                plan_info = {"model": "default", "execution_method": "direct_llm", "reasoning": f"Fallback: {e}"}
            
            # [OPTIMIZATION 2.3] Pass plan to service to avoid redundant re-execution
            if plan:
                context['execution_plan'] = plan
            
            # Emit plan event so frontend can display the execution strategy
            event = StreamEvent(step='plan', message=f'Execution plan: {plan_info.get("reasoning", "N/A")[:100]}', progress=38, plan=plan_info)
            yield f"data: {event.model_dump_json()}\n\n"
            
            # Initialize result variables
            full_response = ""
            token_count = 0
            visualization = None
            code = None
            generated_code = None
            execution_id = None
            execution_time = 0
            agent_used = False
            
            # =================================================================
            # UNIFIED PATH: Always use service.analyze() for ALL queries
            # This ensures data is loaded, semantic mapping applied, 
            # proper agent routing, code generation, and chart creation.
            # The old DIRECT_LLM path had NO data context and hallucinated.
            # =================================================================
            event = StreamEvent(step='mode', message=f'Agent Analysis ({execution_method.value})', progress=40)
            yield f"data: {event.model_dump_json()}\n\n"
            event = StreamEvent(step='agent_start', message='Running analysis with data context...', progress=50)
            yield f"data: {event.model_dump_json()}\n\n"
            
            # Pass the pre-computed plan to context to prevent double orchestrator run
            if plan:
                context['execution_plan'] = plan
                context['selected_model'] = selected_model
            
            # Execute full analysis with agents, tools, and code generation
            result_dict = await service.analyze(query=request.query.strip()[:1000], context=context)
            
            # Extract components from the result
            raw_result = result_dict.get("result", "")
            
            # Handle failed analysis — show error message instead of "None"
            if not result_dict.get("success") and result_dict.get("error"):
                raw_result = f"Analysis could not be completed: {result_dict['error']}"
            
            # Format result text
            if isinstance(raw_result, dict):
                interpretation = result_dict.get("interpretation")
                if interpretation:
                    result_text = interpretation
                else:
                    result_text = _format_dict_result(raw_result)
            else:
                result_text = str(raw_result) if raw_result else "No results returned from analysis."
                
            # Extract metadata (visualizations, code, etc.) — runs for ALL result types
            metadata = result_dict.get("metadata", {})
            visualization = metadata.get("visualization")
            code = metadata.get("code") or metadata.get("executed_code")
            generated_code = metadata.get("generated_code")
            execution_id = metadata.get("execution_id")
            execution_time = metadata.get("execution_time", 0)
            agent_used = True
            
            # Notify about results
            event = StreamEvent(step='agent_complete', message='Analysis complete, streaming results...', progress=70)
            yield f"data: {event.model_dump_json()}\n\n"
            
            # Stream the result text chunk-by-chunk for consistent UX
            _STREAM_CHUNK_SIZE = 80  # ~1 line per event (was 5, causing 400+ events per response)
            for i in range(0, len(result_text), _STREAM_CHUNK_SIZE):
                chunk = result_text[i:i+_STREAM_CHUNK_SIZE]
                full_response += chunk
                token_count += 1
                
                # Emit token event
                event = StreamEvent(step='token', token=chunk, progress=min(95, 70 + ((i / max(len(result_text), 1)) * 25)))
                yield f"data: {event.model_dump_json()}\n\n"
                
                # Small yield to allow event loop to flush
                if token_count % 5 == 0:
                    await asyncio.sleep(0)
            
            execution_method_str = "agent_analysis"
            
            # Step 5: Formatting results
            event = StreamEvent(step='formatting', message='Formatting results...', progress=96)
            yield f"data: {event.model_dump_json()}\n\n"
            
            # Step 6: Complete - Return the full response with all metadata
            result_data = {
                "result": full_response,
                "visualization": visualization,  # Now includes charts from agents!
                "code": code,  # Now includes generated/executed code!
                "generated_code": generated_code,
                "execution_id": execution_id,
                "execution_time": execution_time,
                "execution_method": execution_method_str,
                "query": request.query,
                "filename": files[0] if files and len(files) == 1 else None,
                "filenames": files,
                "analysis_id": analysis_id,
                "status": "success",
                "error": None,
                "model": selected_model,
                "token_count": token_count,
                "agent_used": agent_used,
                "agent": result_dict.get("agent"),
                "plan": plan_info
            }
            
            event = StreamEvent(step='complete', message='Analysis complete!', progress=100, result=result_data)
            yield f"data: {event.model_dump_json()}\n\n"
            
            if analysis_id:
                analysis_manager.complete_analysis(analysis_id)
            
        except HTTPException as e:
            # Cancelled analysis
            if analysis_id:
                analysis_manager.complete_analysis(analysis_id)
            event = StreamEvent(step='error', message=str(e.detail), error=str(e.detail))
            yield f"data: {event.model_dump_json()}\n\n"
            
        except Exception as e:
            logger.error("[STREAM] Error in streaming analysis: %s", e, exc_info=True)
            if analysis_id:
                analysis_manager.complete_analysis(analysis_id)
            event = StreamEvent(step='error', message=f'Analysis failed: {str(e)}', error=str(e))
            yield f"data: {event.model_dump_json()}\n\n"
    
    return StreamingResponse(
        generate_updates(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.post("/cancel/{analysis_id}")
async def cancel_analysis(analysis_id: str) -> Dict[str, str]:
    """Cancel a running analysis by its tracking ID.

    Args:
        analysis_id: Unique identifier returned when the analysis started.

    Returns:
        Dict with ``message`` and ``status`` (``cancelled`` or ``error``).
    """
    success = analysis_manager.cancel_analysis(analysis_id)
    if success:
        return {"message": f"Analysis {analysis_id} has been cancelled", "status": "cancelled"}
    else:
        return {"error": f"Analysis {analysis_id} not found or already completed", "status": "error"}


@router.get("/status/{analysis_id}")
async def get_analysis_status(analysis_id: str) -> Dict[str, Any]:
    """Poll the lifecycle status of an in-flight analysis.

    Args:
        analysis_id: Unique identifier returned when the analysis started.

    Returns:
        Dict with ``analysis`` details and ``status`` (``found`` or ``error``).
    """
    status = analysis_manager.get_analysis_status(analysis_id)
    if status:
        return {"analysis": status, "status": "found"}
    else:
        return {"error": f"Analysis {analysis_id} not found", "status": "error"}


@router.get("/running")
async def get_running_analyses() -> Dict[str, Any]:
    """List all currently executing analyses.

    Returns:
        Dict with ``running_analyses`` list, ``count``, and ``status``.
    """
    running = analysis_manager.get_running_analyses()
    return {"running_analyses": running, "count": len(running), "status": "success"}


class ReviewInsightsRequest(BaseModel):
    original_results: dict
    review_model: Optional[str] = "phi3:latest"
    analysis_type: Optional[str] = "quality_review"


@router.post("/review-insights")
async def generate_review_insights(request: ReviewInsightsRequest) -> Dict[str, Any]:
    """
    Generate review insights using a secondary model to analyze the primary results.
    """
    try:
        # Import ReviewerAgent directly - bypass routing to avoid StatisticalAgent taking over
        from backend.core.plugin_system import get_agent_registry
        registry = get_agent_registry()
        reviewer = registry.get_agent("Reviewer")
        
        if not reviewer:
            logger.warning("[REVIEW] ReviewerAgent not found, falling back to LLM")
            from backend.agents.model_manager import get_model_manager
            manager = get_model_manager()
            manager.ensure_initialized()
            
            # Direct LLM fallback
            review_prompt = f"""Review the following data analysis results:

{request.original_results.get('result', 'No results available')}

Provide:
1. ✅ Accuracy Check: Are calculations consistent?
2. ✅ Key Insights: Most important findings?
3. ⚠️ Limitations: Any concerns?
4. 📊 Quality Score: Rate 1-10."""
            
            response = manager.llm_client.generate(
                prompt=review_prompt,
                model=request.review_model or manager.review_llm.model
            )
            result_text = response.get('response', str(response)) if isinstance(response, dict) else str(response)
            
            return {
                "insights": result_text,
                "quality_metrics": {"analysis_depth": min(len(result_text.split('\n')), 10)},
                "status": "success"
            }
    except Exception as e:
        return {"error": f"Failed to initialize review system: {str(e)}", "status": "error"}
    
    try:
        logger.info("[REVIEW] Generating review insights with model: %s", request.review_model)
        
        if not request.original_results:
            return {"error": "Original results are required", "status": "error"}
        
        # Create review query with clear marker for ReviewerAgent
        original_result = request.original_results.get('result', 'No results available')
        if isinstance(original_result, dict):
            # Format dict result for review
            result_str = str(original_result)[:8000]
        else:
            result_str = str(original_result)[:8000]
        
        review_query = f"""RESULTS TO REVIEW:
{result_str}

Please provide quality assessment, key insights validation, and potential limitations."""
        
        # Call ReviewerAgent directly - bypass the routing system
        result = reviewer.execute(
            query=review_query,
            data=None,
            force_model=request.review_model
        )
        
        logger.info("[REVIEW] Got result from ReviewerAgent: success=%s", result.get('success'))
        
        # Extract result - handle None values properly
        raw_result = result.get("result")
        if raw_result is None or raw_result == "None":
            # If result is None, check for error or provide fallback
            error_msg = result.get("error", "")
            if error_msg:
                result_text = f"Review could not be generated: {error_msg}"
            else:
                result_text = "Analysis review: The results appear valid. No specific issues detected."
        elif isinstance(raw_result, dict):
            result_text = str(raw_result.get("result", raw_result))
        else:
            result_text = str(raw_result)
        
        # Simple quality metrics
        quality_metrics = {
            "analysis_depth": min(len(result_text.split('\n')), 10),
        }
        
        return {
            "insights": result_text,
            "quality_metrics": quality_metrics,
            "status": "success"
        }
        
    except Exception as e:
        logger.error("[REVIEW] Error generating review insights: %s", e, exc_info=True)
        return {"error": str(e), "status": "error"}


@router.get("/routing-stats")
async def get_routing_statistics() -> Dict[str, Any]:
    """Retrieve intelligent routing statistics from the orchestrator.

    Returns:
        Dict with ``statistics`` from the semantic router, or a
        ``routing_enabled: False`` indicator if not yet initialised.
    """
    try:
        from backend.agents.model_manager import get_model_manager
        manager = get_model_manager()
        
        # Access intelligent router statistics directly via property
        if manager.orchestrator:
            stats = manager.orchestrator.get_statistics()
            return {
                "status": "success",
                "statistics": stats,
                "routing_enabled": True
            }
        else:
            return {
                "status": "success",
                "message": "Intelligent routing not yet initialized",
                "routing_enabled": False
            }
            
    except Exception as e:
        logger.error("[ROUTING] Error getting routing statistics: %s", e, exc_info=True)
        return {"error": str(e), "status": "error"}


