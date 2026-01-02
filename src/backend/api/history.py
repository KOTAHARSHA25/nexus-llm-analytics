from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import json
import os
from datetime import datetime, timezone

# API endpoints for query history management

router = APIRouter()

class QueryHistoryItem(BaseModel):
    query: str
    timestamp: Optional[str] = None
    results_summary: Optional[str] = None
    files_used: Optional[List[str]] = None

class QueryHistoryResponse(BaseModel):
    history: List[QueryHistoryItem]
    total_count: int

# Simple file-based storage for query history
from backend.core.config import settings

MAX_HISTORY_ITEMS = 100  # Keep only the last 100 queries

def _get_history_file():
    """Get the path to the history file dynamically"""
    # Use reports path as base, similar to original logic
    # Original: DATA_DIR = settings.get_reports_path().parent
    # We want it to be securely within the allowed data/reports structure
    data_dir = settings.get_reports_path().parent
    history_dir = data_dir / 'history'
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir / 'query_history.json'

def ensure_history_dir():
    """Ensure the history directory exists (deprecated, handled in _get_history_file)"""
    _get_history_file().parent.mkdir(parents=True, exist_ok=True)

def load_history() -> List[Dict[str, Any]]:
    """Load query history from file"""
    history_file = _get_history_file()
    
    if not os.path.exists(history_file):
        return []
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('history', [])
    except Exception as e:
        logging.error(f"Failed to load query history: {e}")
        return []

def save_history(history: List[Dict[str, Any]]) -> bool:
    """Save query history to file"""
    history_file = _get_history_file()
    
    try:
        # Keep only the most recent items
        trimmed_history = history[-MAX_HISTORY_ITEMS:] if len(history) > MAX_HISTORY_ITEMS else history
        
        data = {
            'history': trimmed_history,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        logging.error(f"Failed to save query history: {e}")
        return False

@router.get("/", response_model=QueryHistoryResponse)
async def get_query_history() -> QueryHistoryResponse:
    """Get all query history"""
    try:
        history = load_history()
        
        # Convert to QueryHistoryItem objects
        history_items = [
            QueryHistoryItem(
                query=item.get('query', ''),
                timestamp=item.get('timestamp'),
                results_summary=item.get('results_summary'),
                files_used=item.get('files_used', [])
            )
            for item in history
        ]
        
        return QueryHistoryResponse(
            history=history_items,
            total_count=len(history_items)
        )
        
    except Exception as e:
        logging.error(f"Failed to get query history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve query history")

@router.post("/add")
async def add_query_to_history(item: QueryHistoryItem) -> Dict[str, str]:
    """Add a new query to history"""
    try:
        history = load_history()
        
        # Add timestamp if not provided
        if not item.timestamp:
            item.timestamp = datetime.now(timezone.utc).isoformat()
        
        # Convert to dict and add to history
        history_dict = {
            'query': item.query,
            'timestamp': item.timestamp,
            'results_summary': item.results_summary,
            'files_used': item.files_used or []
        }
        
        history.append(history_dict)
        
        # Save updated history
        if save_history(history):
            return {"status": "success", "message": "Query added to history"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save query to history")
            
    except Exception as e:
        logging.error(f"Failed to add query to history: {e}")
        raise HTTPException(status_code=500, detail="Failed to add query to history")

@router.delete("/clear")
async def clear_query_history() -> Dict[str, str]:
    """Clear all query history"""
    try:
        # Save empty history
        if save_history([]):
            return {"status": "success", "message": "Query history cleared"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear query history")
            
    except Exception as e:
        logging.error(f"Failed to clear query history: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear query history")

@router.delete("/{index}")
async def delete_query_from_history(index: int) -> Dict[str, str]:
    """Delete a specific query from history by index"""
    try:
        history = load_history()
        
        if index < 0 or index >= len(history):
            raise HTTPException(status_code=404, detail="Query not found")
        
        # Remove the item at the specified index
        removed_query = history.pop(index)
        
        # Save updated history
        if save_history(history):
            return {
                "status": "success", 
                "message": f"Query removed from history",
                "removed_query": removed_query.get('query', '')
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to remove query from history")
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to delete query from history: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete query from history")

@router.get("/search")
async def search_query_history(q: str) -> QueryHistoryResponse:
    """Search query history"""
    try:
        history = load_history()
        
        # Simple text search (case-insensitive)
        search_term = q.lower()
        filtered_history = [
            item for item in history 
            if search_term in item.get('query', '').lower()
        ]
        
        # Convert to QueryHistoryItem objects
        history_items = [
            QueryHistoryItem(
                query=item.get('query', ''),
                timestamp=item.get('timestamp'),
                results_summary=item.get('results_summary'),
                files_used=item.get('files_used', [])
            )
            for item in filtered_history
        ]
        
        return QueryHistoryResponse(
            history=history_items,
            total_count=len(history_items)
        )
        
    except Exception as e:
        logging.error(f"Failed to search query history: {e}")
        raise HTTPException(status_code=500, detail="Failed to search query history")

@router.get("/stats")
async def get_history_stats() -> Dict[str, Any]:
    """Get query history statistics"""
    try:
        history = load_history()
        
        if not history:
            return {
                "total_queries": 0,
                "date_range": None,
                "most_common_files": [],
                "recent_activity": []
            }
        
        # Calculate stats
        total_queries = len(history)
        
        # Date range
        timestamps = [item.get('timestamp') for item in history if item.get('timestamp')]
        date_range = None
        if timestamps:
            try:
                dates = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
                date_range = {
                    "earliest": min(dates).isoformat(),
                    "latest": max(dates).isoformat()
                }
            except Exception:
                logging.debug("Operation failed (non-critical) - continuing")
        
        # Most common files
        file_count = {}
        for item in history:
            files = item.get('files_used', [])
            for file in files:
                file_count[file] = file_count.get(file, 0) + 1
        
        most_common_files = sorted(file_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Recent activity (last 10 queries)
        recent_activity = [
            {
                "query": item.get('query', '')[:100] + ('...' if len(item.get('query', '')) > 100 else ''),
                "timestamp": item.get('timestamp'),
                "files_count": len(item.get('files_used', []))
            }
            for item in history[-10:]
        ]
        
        return {
            "total_queries": total_queries,
            "date_range": date_range,
            "most_common_files": most_common_files,
            "recent_activity": recent_activity
        }
        
    except Exception as e:
        logging.error(f"Failed to get history stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get history statistics")


# ============================================================================
# CODE EXECUTION HISTORY ENDPOINTS
# ============================================================================
# These endpoints expose the code generation/execution history for user review

class CodeExecutionItem(BaseModel):
    """A single code execution record"""
    execution_id: str
    timestamp: str
    query: str
    model_used: str
    success: bool
    execution_time_ms: float
    result_preview: Optional[str] = None
    result_type: Optional[str] = None
    error: Optional[str] = None
    generated_code: Optional[str] = None
    cleaned_code: Optional[str] = None
    data_file: Optional[str] = None
    columns: Optional[List[str]] = None
    attempt_count: int = 1

class CodeExecutionHistoryResponse(BaseModel):
    """Response for code execution history"""
    executions: List[CodeExecutionItem]
    total_count: int
    summary: Optional[Dict[str, Any]] = None

class CodeReplayRequest(BaseModel):
    """Request to replay a past execution"""
    execution_id: str
    data_file: Optional[str] = None

class CodeReplayResponse(BaseModel):
    """Response from replaying an execution"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    code: Optional[str] = None


@router.get("/code-executions", response_model=CodeExecutionHistoryResponse)
async def get_code_execution_history(
    limit: int = 20,
    success_only: bool = False,
    query_filter: Optional[str] = None
) -> CodeExecutionHistoryResponse:
    """
    Get code execution history with generated code and results.
    
    Args:
        limit: Maximum number of records to return
        success_only: Only return successful executions
        query_filter: Filter by query text
    """
    try:
        from backend.core.code_execution_history import get_execution_history
        
        history = get_execution_history()
        
        # Search with filters
        records = history.search_executions(
            query_contains=query_filter,
            success_only=success_only,
            limit=limit
        )
        
        # Get summary stats
        summary = history.get_execution_summary()
        
        # Convert to response format
        executions = [
            CodeExecutionItem(
                execution_id=r.execution_id,
                timestamp=r.timestamp,
                query=r.query,
                model_used=r.model_used,
                success=r.success,
                execution_time_ms=r.execution_time_ms,
                result_preview=r.result_preview,
                result_type=r.result_type,
                error=r.error,
                generated_code=r.generated_code,
                cleaned_code=r.cleaned_code,
                data_file=r.data_file,
                columns=r.columns,
                attempt_count=r.attempt_count
            )
            for r in records
        ]
        
        return CodeExecutionHistoryResponse(
            executions=executions,
            total_count=len(executions),
            summary=summary
        )
        
    except Exception as e:
        logging.error(f"Failed to get code execution history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve code execution history")


@router.get("/code-executions/{execution_id}")
async def get_code_execution_detail(execution_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific code execution.
    
    Returns the full execution record including code and results.
    """
    try:
        from backend.core.code_execution_history import get_execution_history
        
        history = get_execution_history()
        record = history.get_execution(execution_id)
        
        if not record:
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
        
        return {
            "execution_id": record.execution_id,
            "timestamp": record.timestamp,
            "query": record.query,
            "model_used": record.model_used,
            "success": record.success,
            "execution_time_ms": record.execution_time_ms,
            "result": record.result_preview,  # Use preview to avoid serialization issues
            "result_type": record.result_type,
            "error": record.error,
            "generated_code": record.generated_code,
            "cleaned_code": record.cleaned_code,
            "data_file": record.data_file,
            "columns": record.columns,
            "row_count": record.row_count,
            "attempt_count": record.attempt_count,
            "retry_errors": record.retry_errors
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get code execution detail: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve execution details")


@router.get("/code-executions/{execution_id}/code")
async def get_execution_code(execution_id: str) -> Dict[str, str]:
    """
    Get just the code from a specific execution for copying/replay.
    """
    try:
        from backend.core.code_execution_history import get_execution_history
        
        history = get_execution_history()
        replay_info = history.get_code_for_replay(execution_id)
        
        if not replay_info:
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
        
        return replay_info
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get execution code: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve execution code")


@router.post("/code-executions/replay")
async def replay_code_execution(request: CodeReplayRequest) -> CodeReplayResponse:
    """
    Replay a past code execution using the stored code.
    
    Note: Requires the same data file to be available.
    """
    try:
        import pandas as pd
        from backend.core.code_generator import get_code_generator
        from backend.core.code_execution_history import get_execution_history
        from backend.core.config import settings
        
        # Get the execution record
        history = get_execution_history()
        record = history.get_execution(request.execution_id)
        
        if not record:
            raise HTTPException(status_code=404, detail=f"Execution {request.execution_id} not found")
        
        # Determine data file to use
        data_file = request.data_file or record.data_file
        if not data_file:
            raise HTTPException(status_code=400, detail="No data file specified for replay")
        
        # Load the data file
        uploads_dir = settings.get_uploads_path()
        file_path = uploads_dir / data_file
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Data file {data_file} not found")
        
        # Load DataFrame
        if data_file.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif data_file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Replay the execution
        generator = get_code_generator()
        result = generator.replay_execution(request.execution_id, df)
        
        return CodeReplayResponse(
            success=result.success,
            result=str(result.result) if result.result is not None else None,
            error=result.error,
            execution_time_ms=result.execution_time_ms,
            code=result.code
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to replay execution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to replay execution: {str(e)}")


@router.get("/code-executions/summary/stats")
async def get_code_execution_stats() -> Dict[str, Any]:
    """Get summary statistics for code executions"""
    try:
        from backend.core.code_execution_history import get_execution_history
        
        history = get_execution_history()
        return history.get_execution_summary()
        
    except Exception as e:
        logging.error(f"Failed to get code execution stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get execution statistics")


@router.delete("/code-executions/clear")
async def clear_code_execution_history() -> Dict[str, Any]:
    """Clear all code execution history"""
    try:
        from backend.core.code_execution_history import get_execution_history
        
        history = get_execution_history()
        count = history.clear_history()
        
        return {
            "status": "success",
            "message": f"Cleared {count} execution records"
        }
        
    except Exception as e:
        logging.error(f"Failed to clear code execution history: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear execution history")


@router.get("/code-executions/{execution_id}/export")
async def export_code_execution(execution_id: str) -> Dict[str, Any]:
    """
    Export an execution record for sharing or documentation.
    """
    try:
        from backend.core.code_execution_history import get_execution_history
        
        history = get_execution_history()
        export_data = history.export_execution(execution_id)
        
        if not export_data:
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
        
        return export_data
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to export execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to export execution")