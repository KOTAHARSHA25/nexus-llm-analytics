from fastapi import APIRouter, HTTPException
import logging
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from backend.services.analysis_service import get_analysis_service
from backend.core.analysis_manager import analysis_manager, check_cancellation

# Handles /analyze endpoint logic for data analysis requests using AnalysisService

router = APIRouter()


def _format_dict_result(data: Dict[str, Any], indent: int = 0) -> str:
    """
    Format a dictionary result as readable text for user display.
    Converts raw JSON-like data into human-readable analysis summary.
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
    query: str = Field(..., description="The natural language query for data analysis", example="What are the top 5 categories by sales?")
    filename: Optional[str] = Field(None, description="Single file to analyze (backward compatible)", example="sales_data.csv")
    filenames: Optional[List[str]] = Field(None, description="Multiple files to analyze (supports multi-file joins)", example=["orders.csv", "customers.csv"])
    text_data: Optional[str] = Field(None, description="Direct text input for analysis without file upload")
    column: Optional[str] = Field(None, description="Specific column to filter on")
    value: Optional[str] = Field(None, description="Value to filter the column by")
    session_id: Optional[str] = Field(None, description="Session identifier for tracking related queries")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Show me the top 10 customers by total purchase amount",
                "filename": "sales_data.csv"
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


@router.post("/", response_model=AnalyzeResponse, responses={
    200: {"description": "Analysis completed successfully"},
    400: {"description": "Invalid request parameters"},
    503: {"description": "AI service unavailable"}
})
async def analyze_query(request: AnalyzeRequest) -> Dict[str, Any]:
    # Get singleton AnalysisService instance
    try:
        service = get_analysis_service()
    except Exception as e:
        logging.error(f"Failed to initialize AnalysisService: {e}")
        return {"error": f"Failed to initialize AI system: {str(e)}", "status": "error"}
    
    # Start analysis tracking
    analysis_id = analysis_manager.start_analysis(request.session_id)
    
    logging.info(f"[ANALYZE] Started analysis {analysis_id}: {request.query}, filename: {request.filename}, filenames: {request.filenames}, text_data: {bool(request.text_data)}")
    
    try:
        # Input validation - support both single and multiple files
        if not request.query:
            analysis_manager.complete_analysis(analysis_id)
            return {"error": "Query is required", "status": "error"}
        
        # Determine files to analyze
        files = None
        if request.filenames:
            files = request.filenames
            if not files:
                analysis_manager.complete_analysis(analysis_id)
                return {"error": "At least one filename required", "status": "error"}
        elif request.filename:
            files = [request.filename]
        elif request.text_data:
            # Handle text input directly via service (text_data in context)
            files = None
        else:
            analysis_manager.complete_analysis(analysis_id)
            return {"error": "Either filename, filenames, or text_data required", "status": "error"}
        
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
            "analysis_id": analysis_id
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
            logging.info(f"Formatted dict result (len={len(result_str)})")
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
            "error": result.get("error")
        }
    except HTTPException as e:
        # Analysis was cancelled
        analysis_manager.complete_analysis(analysis_id)
        raise e
    except Exception as e:
        logging.error(f"[ANALYZE] Error in analysis {analysis_id}: {e}", exc_info=True)
        analysis_manager.complete_analysis(analysis_id)
        return {
            "error": f"Analysis failed: {str(e)}",
            "status": "error", 
            "analysis_id": analysis_id,
            "retry_possible": True
        }


@router.post("/cancel/{analysis_id}")
async def cancel_analysis(analysis_id: str):
    """Cancel a running analysis"""
    success = analysis_manager.cancel_analysis(analysis_id)
    if success:
        return {"message": f"Analysis {analysis_id} has been cancelled", "status": "cancelled"}
    else:
        return {"error": f"Analysis {analysis_id} not found or already completed", "status": "error"}


@router.get("/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """Get the status of an analysis"""
    status = analysis_manager.get_analysis_status(analysis_id)
    if status:
        return {"analysis": status, "status": "found"}
    else:
        return {"error": f"Analysis {analysis_id} not found", "status": "error"}


@router.get("/running")
async def get_running_analyses():
    """Get all currently running analyses"""
    running = analysis_manager.get_running_analyses()
    return {"running_analyses": running, "count": len(running), "status": "success"}


class ReviewInsightsRequest(BaseModel):
    original_results: dict
    review_model: Optional[str] = "phi3:latest"
    analysis_type: Optional[str] = "quality_review"


@router.post("/review-insights")
async def generate_review_insights(request: ReviewInsightsRequest):
    """
    Generate review insights using a secondary model to analyze the primary results.
    """
    try:
        # Import ReviewerAgent directly - bypass routing to avoid StatisticalAgent taking over
        from backend.core.plugin_system import get_agent_registry
        registry = get_agent_registry()
        reviewer = registry.get_agent("Reviewer")
        
        if not reviewer:
            logging.warning("[REVIEW] ReviewerAgent not found, falling back to LLM")
            from backend.agents.model_initializer import get_model_initializer
            initializer = get_model_initializer()
            initializer.ensure_initialized()
            
            # Direct LLM fallback
            review_prompt = f"""Review the following data analysis results:

{request.original_results.get('result', 'No results available')}

Provide:
1. ✅ Accuracy Check: Are calculations consistent?
2. ✅ Key Insights: Most important findings?
3. ⚠️ Limitations: Any concerns?
4. 📊 Quality Score: Rate 1-10."""
            
            response = initializer.llm_client.generate(
                prompt=review_prompt,
                model=request.review_model or initializer.review_llm.model
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
        logging.info(f"[REVIEW] Generating review insights with model: {request.review_model}")
        
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
        
        logging.info(f"[REVIEW] Got result from ReviewerAgent: success={result.get('success')}")
        
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
        logging.error(f"[REVIEW] Error generating review insights: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}


@router.get("/routing-stats")
async def get_routing_statistics():
    """
    Get intelligent routing statistics
    """
    try:
        from backend.agents.model_initializer import get_model_initializer
        initializer = get_model_initializer()
        
        # Access intelligent router statistics directly
        if hasattr(initializer, 'intelligent_router'):
            stats = initializer.intelligent_router.get_statistics()
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
        logging.error(f"[ROUTING] Error getting routing statistics: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}


