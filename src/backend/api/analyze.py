from fastapi import APIRouter, HTTPException
import logging
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from backend.core.crew_singleton import get_crew_manager
from backend.core.analysis_manager import analysis_manager, check_cancellation

# Handles /analyze endpoint logic for data analysis requests using CrewAI

router = APIRouter()


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
    code: Optional[str] = Field(None, description="Generated code for the analysis")
    execution_time: float = Field(0, description="Time taken for analysis in seconds")
    query: str = Field(..., description="The original query")
    filename: Optional[str] = Field(None, description="File analyzed")
    filenames: Optional[List[str]] = Field(None, description="Files analyzed (multi-file)")
    analysis_id: str = Field(..., description="Unique identifier for this analysis")
    status: str = Field(..., description="Status of the analysis: success, error, cancelled")
    error: Optional[str] = Field(None, description="Error message if analysis failed")


@router.post("/", response_model=AnalyzeResponse, responses={
    200: {"description": "Analysis completed successfully"},
    400: {"description": "Invalid request parameters"},
    503: {"description": "AI service unavailable"}
})
async def analyze_query(request: AnalyzeRequest) -> Dict[str, Any]:
    # Get singleton CrewManager instance
    try:
        crew_manager = get_crew_manager()
    except Exception as e:
        logging.error(f"Failed to initialize CrewManager: {e}")
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
            # Handle text input - analyze directly without file-based routing
            import os
            from datetime import datetime
            
            logging.info(f"[ANALYZE] Processing direct text input (length: {len(request.text_data)} chars)")
            
            # Sanitize inputs
            query = request.query.strip()[:1000]
            text_data = request.text_data.strip()
            
            analysis_manager.update_analysis_stage(analysis_id, "processing")
            
            # Use LLM directly for text analysis (bypass file-based routing)
            try:
                from backend.core.llm_client import LLMClient
                llm_client = LLMClient()
                
                # Create analysis prompt
                analysis_prompt = f"""Analyze the following text and answer the user's question.

User's Question: {query}

Text Content:
{text_data[:2000]}

Provide a clear, concise answer based on the text content. Extract specific information, insights, or patterns as requested."""

                # Get LLM response
                llm_result = llm_client.generate_primary(analysis_prompt)
                
                # Extract the actual response text
                if isinstance(llm_result, dict):
                    llm_response = llm_result.get('response', str(llm_result))
                else:
                    llm_response = str(llm_result)
                
                analysis_manager.complete_analysis(analysis_id)
                
                return {
                    "success": True,
                    "result": llm_response,
                    "query": query,
                    "filename": "Text Input",
                    "type": "text_analysis",
                    "execution_time": 0,
                    "analysis_id": analysis_id,
                    "status": "success"
                }
                
            except Exception as e:
                logging.error(f"[ANALYZE] Text analysis failed: {e}")
                analysis_manager.complete_analysis(analysis_id)
                return {
                    "error": f"Text analysis failed: {str(e)}",
                    "status": "error",
                    "analysis_id": analysis_id
                }
            
            # This code won't be reached, but kept for structure
            files = None
        else:
            analysis_manager.complete_analysis(analysis_id)
            return {"error": "Either filename, filenames, or text_data required", "status": "error"}
        
        # Check if cancelled before starting
        check_cancellation(analysis_id)
        
        # Sanitize inputs
        query = request.query.strip()[:1000]  # Limit query length
        files = [f.strip() for f in files]
        
        analysis_manager.update_analysis_stage(analysis_id, "processing")
        
        # Execute analysis using CrewAI with cancellation checks
        # Pass filenames list for multi-file support
        result = crew_manager.handle_query(
            query=query,
            filenames=files,  # Pass as list for multi-file support
            column=request.column,
            value=request.value,
            analysis_id=analysis_id  # Pass analysis ID for cancellation checks
        )
        
        # Check if cancelled before returning results
        check_cancellation(analysis_id)
        
        analysis_manager.complete_analysis(analysis_id)
        
        # Return successful analysis result
        return {
            "result": result.get("result", ""),
            "visualization": result.get("visualization"),
            "code": result.get("code"),
            "execution_time": result.get("execution_time", 0),
            "query": query,
            "filename": files[0] if len(files) == 1 else None,  # Backward compatibility
            "filenames": files,  # Multi-file support
            "analysis_id": analysis_id,
            "status": "success"
        }
    except HTTPException as e:
        # Analysis was cancelled
        analysis_manager.complete_analysis(analysis_id)
        raise e
    except Exception as e:
        logging.error(f"[ANALYZE] Error in analysis {analysis_id}: {e}", exc_info=True)
        analysis_manager.complete_analysis(analysis_id)
        
        # Provide more specific error messages based on error type
        error_msg = str(e)
        if "connection" in error_msg.lower() and "refused" in error_msg.lower():
            return {
                "error": "AI service is currently unavailable",
                "details": "The local AI model server (Ollama) is not running or not accessible. Please ensure Ollama is installed and running on port 11434.",
                "suggestions": [
                    "Start Ollama service: 'ollama serve'",
                    "Check if the correct model is installed: 'ollama list'",
                    "Verify Ollama is running on localhost:11434"
                ],
                "status": "service_unavailable",
                "analysis_id": analysis_id,
                "retry_possible": True
            }
        elif "timeout" in error_msg.lower():
            return {
                "error": "Analysis request timed out",
                "details": "The AI model is taking longer than expected. This may be due to limited system resources or swap memory usage.",
                "suggestions": [
                    "Try with a smaller model (check available models in Settings)",
                    "Reduce the complexity of your query",
                    "Ensure sufficient system memory is available",
                    "Install a lightweight model: ollama pull tinyllama"
                ],
                "status": "timeout",
                "analysis_id": analysis_id,
                "retry_possible": True
            }
        elif "memory" in error_msg.lower():
            return {
                "error": "Insufficient system memory",
                "details": "The selected AI model requires more system memory than is currently available.",
                "suggestions": [
                    "Switch to a smaller model in Model Settings",
                    "Enable swap memory usage in Model Settings",
                    "Close other applications to free up memory"
                ],
                "status": "insufficient_memory",
                "analysis_id": analysis_id,
                "retry_possible": True
            }
        else:
            return {
                "error": f"Analysis failed: {error_msg}",
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
    This provides a different perspective and quality assessment of the analysis.
    """
    # Get singleton CrewManager instance
    try:
        crew_manager = get_crew_manager()
    except Exception as e:
        logging.error(f"Failed to initialize CrewManager: {e}")
        return {"error": f"Failed to initialize AI system: {str(e)}", "status": "error"}
    
    try:
        logging.info(f"[REVIEW] Generating review insights with model: {request.review_model}")
        
        # Input validation
        if not request.original_results:
            return {"error": "Original results are required", "status": "error"}
        
        # Create review query based on analysis type
        if request.analysis_type == "quality_review":
            review_query = f"""
Please review the following data analysis results and provide insights:

Original Analysis Results:
{request.original_results.get('result', 'No results available')}

Please provide:
1. Quality assessment of the analysis approach
2. Key insights and findings validation
3. Potential limitations or biases
4. Suggestions for further analysis
5. Alternative interpretations of the data

Focus on being constructive and providing actionable feedback.
"""
        else:
            review_query = f"""
Please analyze and provide insights on the following analysis results:
{request.original_results.get('result', 'No results available')}

Provide additional perspectives and recommendations.
"""
        
        # Use crew manager to generate review insights
        # For now, we'll use a simplified approach - in a full system, you might
        # want to use a different model or specialized review agent
        result = crew_manager.handle_query(
            query=review_query,
            filename=request.original_results.get('filename', 'analysis_results'),
            column=None,
            value=None,
            force_model=request.review_model  # Force use of review model
        )
        
        # Extract quality metrics if available
        quality_metrics = {}
        if result.get("result"):
            # Simple quality metrics extraction (could be enhanced)
            result_text = str(result.get("result", ""))
            quality_metrics = {
                "analysis_depth": min(len(result_text.split('\n')), 10),
                "key_points": result_text.count('•') + result_text.count('-') + result_text.count('*'),
                "confidence_indicators": result_text.lower().count('confident') + result_text.lower().count('likely'),
                "suggestions_count": result_text.lower().count('suggest') + result_text.lower().count('recommend')
            }
        
        return {
            "insights": result.get("result", ""),
            "quality_metrics": quality_metrics,
            "review_model": request.review_model,
            "analysis_type": request.analysis_type,
            "execution_time": result.get("execution_time", 0),
            "status": "success"
        }
        
    except Exception as e:
        logging.error(f"[REVIEW] Error generating review insights: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}


@router.get("/routing-stats")
async def get_routing_statistics():
    """
    Get intelligent routing statistics for research/analysis
    
    Returns statistics about query complexity routing decisions:
    - Total decisions made
    - Distribution across tiers (fast/balanced/full_power)
    - Average complexity scores
    - Recent routing decisions
    """
    try:
        crew_manager = get_crew_manager()
        
        # Access intelligent router statistics
        if hasattr(crew_manager, 'intelligent_router'):
            stats = crew_manager.intelligent_router.get_statistics()
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
