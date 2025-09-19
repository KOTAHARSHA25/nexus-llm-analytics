from fastapi import APIRouter
import logging
from pydantic import BaseModel
from typing import Optional
from backend.agents.crew_manager import CrewManager

# Handles /analyze endpoint logic for data analysis requests using CrewAI

router = APIRouter()
crew_manager = None  # Initialize on first use to avoid import-time issues


class AnalyzeRequest(BaseModel):
    query: str
    filename: str
    column: Optional[str] = None
    value: Optional[str] = None


@router.post("/")
async def analyze_query(request: AnalyzeRequest):
    global crew_manager
    
    # Initialize crew manager on first use
    if crew_manager is None:
        try:
            crew_manager = CrewManager()
        except Exception as e:
            logging.error(f"Failed to initialize CrewManager: {e}")
            return {"error": f"Failed to initialize AI system: {str(e)}", "status": "error"}
    
    logging.info(f"[ANALYZE] Received query: {request.query}, filename: {request.filename}")
    
    try:
        # Input validation
        if not request.query or not request.filename:
            return {"error": "Query and filename are required", "status": "error"}
        
        # Sanitize inputs
        query = request.query.strip()[:1000]  # Limit query length
        filename = request.filename.strip()
        
        # Execute analysis using CrewAI
        result = crew_manager.handle_query(
            query=query,
            filename=filename,
            column=request.column,
            value=request.value
        )
        
        # Return successful analysis result
        return {
            "result": result.get("result", ""),
            "visualization": result.get("visualization"),
            "code": result.get("code"),
            "execution_time": result.get("execution_time", 0),
            "query": query,
            "filename": filename,
            "status": "success"
        }
    except Exception as e:
        logging.error(f"[ANALYZE] Error: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}
