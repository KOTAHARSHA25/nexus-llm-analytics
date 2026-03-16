"""History API — Query and Code-Execution History Endpoints
=========================================================

Manages two separate history stores:

1. **Query history** — file-backed JSONL that tracks every natural-language
   query submitted by users (query, timestamp, results summary, files used).
2. **Code execution history** — backed by
   ``backend.core.code_execution_history`` — that records generated code,
   cleaned code, execution result, timing, and retry metadata.

Endpoints — query history
-------------------------
``GET    /``                  Return full query history.
``POST   /add``               Append a query to history.
``DELETE /clear``             Clear all query history.
``DELETE /{index}``           Delete a specific query by index.
``GET    /search``            Text-search within queries.
``GET    /stats``             Aggregate query history statistics.

Endpoints — code execution history
----------------------------------
``GET    /code-executions``                   List recent code executions.
``GET    /code-executions/{id}``              Full execution detail.
``GET    /code-executions/{id}/code``         Retrieve code for replay.
``POST   /code-executions/replay``            Replay a past execution.
``GET    /code-executions/summary/stats``     Aggregate stats.
``DELETE /code-executions/clear``             Clear execution history.
``GET    /code-executions/{id}/export``       Export execution record.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# API endpoints for query history management

router = APIRouter()

class QueryHistoryItem(BaseModel):
    """Single query history entry persisted in JSONL storage."""
    query: str
    timestamp: Optional[str] = None
    results_summary: Optional[str] = None
    files_used: Optional[List[str]] = None

class QueryHistoryResponse(BaseModel):
    """Paginated response wrapper for query history."""
    history: List[QueryHistoryItem]
    total_count: int

# SQLite storage for query history
from backend.core.database import get_db_manager

# Deprecated: File-based constants
# MAX_HISTORY_ITEMS: int = 100 

def load_history() -> List[Dict[str, Any]]:
    """Load query history from database"""
    return get_db_manager().get_recent_queries()

def save_history(history: List[Dict[str, Any]]) -> bool:
    """
    Deprecated: No-op for save_history as DB handles it per-insert.
    Kept if any legacy code calls it directly, but typically endpoints call add_query directly.
    """
    return True

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
        logger.error("Failed to get query history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve query history")

@router.post("/add")
async def add_query_to_history(item: QueryHistoryItem) -> Dict[str, str]:
    """Add a new query to history"""
    try:
        # Add timestamp if not provided
        if not item.timestamp:
            item.timestamp = datetime.now(timezone.utc).isoformat()
        
        # Use DB manager directly
        success = get_db_manager().add_query(
            query=item.query,
            results_summary=item.results_summary,
            files_used=item.files_used,
            timestamp=item.timestamp
        )
        
        if success:
            return {"status": "success", "message": "Query added to history"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save query to history database")
            
    except Exception as e:
        logger.error("Failed to add query to history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to add query to history")

@router.delete("/clear")
async def clear_query_history() -> Dict[str, str]:
    """Clear all query history"""
    try:
        if get_db_manager().clear_history():
            return {"status": "success", "message": "Query history cleared"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear query history")
            
    except Exception as e:
        logger.error("Failed to clear query history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear query history")

@router.delete("/{index}")
async def delete_query_from_history(index: int) -> Dict[str, str]:
    """Delete a specific query from history by index"""
    try:
        result = get_db_manager().delete_query_by_index(index)
        
        if result:
            return {
                "status": "success", 
                "message": f"Query removed from history",
                "removed_query": result.get('query', '')
            }
        else:
            raise HTTPException(status_code=404, detail="Query not found or failed to delete")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete query from history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete query from history")

@router.get("/search")
async def search_query_history(q: str) -> QueryHistoryResponse:
    """Search query history"""
    try:
        history = get_db_manager().search_queries(q)
        
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
        logger.error("Failed to search query history: %s", e, exc_info=True)
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
                logger.debug("Failed to parse history timestamps, skipping date range")
        
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
        logger.error("Failed to get history stats: %s", e, exc_info=True)
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
        logger.error("Failed to get code execution history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve code execution history")


@router.get("/code-executions/{execution_id}")
async def get_code_execution_detail(execution_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific code execution.
    
    Returns the full execution record including code and results.
    """
    try:
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
        logger.error("Failed to get code execution detail: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve execution details")


@router.get("/code-executions/{execution_id}/code")
async def get_execution_code(execution_id: str) -> Dict[str, str]:
    """
    Get just the code from a specific execution for copying/replay.
    """
    try:
        history = get_execution_history()
        replay_info = history.get_code_for_replay(execution_id)
        
        if not replay_info:
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
        
        return replay_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get execution code: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve execution code")


@router.post("/code-executions/replay")
async def replay_code_execution(request: CodeReplayRequest) -> CodeReplayResponse:
    """
    Replay a past code execution using the stored code.
    
    Note: Requires the same data file to be available.
    """
    try:
        import pandas as pd
        from backend.io.code_generator import get_code_generator
        
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
        logger.error("Failed to replay execution: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to replay execution: {e}")


@router.get("/code-executions/summary/stats")
async def get_code_execution_stats() -> Dict[str, Any]:
    """Get summary statistics for code executions"""
    try:
        history = get_execution_history()
        return history.get_execution_summary()
        
    except Exception as e:
        logger.error("Failed to get code execution stats: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get execution statistics")


@router.delete("/code-executions/clear")
async def clear_code_execution_history() -> Dict[str, Any]:
    """Clear all code execution history"""
    try:
        history = get_execution_history()
        count = history.clear_history()
        
        return {
            "status": "success",
            "message": f"Cleared {count} execution records"
        }
        
    except Exception as e:
        logger.error("Failed to clear code execution history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear execution history")


@router.get("/code-executions/{execution_id}/export")
async def export_code_execution(execution_id: str) -> Dict[str, Any]:
    """
    Export an execution record for sharing or documentation.
    """
    try:
        history = get_execution_history()
        export_data = history.export_execution(execution_id)
        
        if not export_data:
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
        
        return export_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to export execution: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to export execution")