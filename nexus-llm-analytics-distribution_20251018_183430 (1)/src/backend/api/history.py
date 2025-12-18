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
HISTORY_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'history')
HISTORY_FILE = os.path.join(HISTORY_DIR, 'query_history.json')
MAX_HISTORY_ITEMS = 100  # Keep only the last 100 queries

def ensure_history_dir():
    """Ensure the history directory exists"""
    os.makedirs(HISTORY_DIR, exist_ok=True)

def load_history() -> List[Dict[str, Any]]:
    """Load query history from file"""
    ensure_history_dir()
    
    if not os.path.exists(HISTORY_FILE):
        return []
    
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('history', [])
    except Exception as e:
        logging.error(f"Failed to load query history: {e}")
        return []

def save_history(history: List[Dict[str, Any]]) -> bool:
    """Save query history to file"""
    ensure_history_dir()
    
    try:
        # Keep only the most recent items
        trimmed_history = history[-MAX_HISTORY_ITEMS:] if len(history) > MAX_HISTORY_ITEMS else history
        
        data = {
            'history': trimmed_history,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
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