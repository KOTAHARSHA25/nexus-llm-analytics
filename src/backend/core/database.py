"""Nexus Database — SQLite Persistence Layer
========================================

Handles persistent storage for query history, sessions, and analysis results using SQLite.
Replaces legacy file-based JSON storage (history.py) for improved reliability and concurrency.

Schema:
    - query_history: Stores parsed user queries and metadata
    - analysis_sessions: (Future) Groups queries into conversational sessions

v2.0 Enterprise Addition
"""

import sqlite3
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from backend.core.config import settings

logger = logging.getLogger(__name__)

DB_FILENAME = "nexus_history.db"

class DatabaseManager:
    """Manages SQLite connection and schema migrations."""
    
    def __init__(self):
        # Store DB in the data directory (same as reports)
        self.db_path = settings.get_reports_path().parent / DB_FILENAME
        self._init_db()

    def _get_connection(self):
        """Get a configured SQLite connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn

    def _init_db(self):
        """Initialize database schema if not exists."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Table: Query History
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS query_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        results_summary TEXT,
                        files_used TEXT,  -- JSON list
                        session_id TEXT,
                        metadata TEXT     -- JSON dict for extra fields
                    )
                """)
                
                # Create index on timestamp for fast retrieval
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON query_history(timestamp DESC)
                """)
                
                conn.commit()
                # logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    # =========================================================================
    # Query History Methods
    # =========================================================================

    def add_query(self, query: str, results_summary: str = None, 
                 files_used: List[str] = None, timestamp: str = None,
                 session_id: str = None) -> bool:
        """Add a new query to history."""
        try:
            if not timestamp:
                timestamp = datetime.now(timezone.utc).isoformat()
            
            files_json = json.dumps(files_used or [])
            
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO query_history (query, timestamp, results_summary, files_used, session_id)
                    VALUES (?, ?, ?, ?, ?)
                """, (query, timestamp, results_summary, files_json, session_id))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to add query to DB: {e}")
            return False

    def get_recent_queries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent queries, ordered by newest first."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT query, timestamp, results_summary, files_used
                    FROM query_history
                    ORDER BY id DESC -- Newer items have higher IDs
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                result = []
                for row in rows:
                    result.append({
                        "query": row["query"],
                        "timestamp": row["timestamp"],
                        "results_summary": row["results_summary"],
                        "files_used": json.loads(row["files_used"] or "[]")
                    })
                
                # API expects chronological order (oldest -> newest)? 
                # Actually history.py load_history just read the file (chrono).
                # But typical UIs reverse it.
                # Let's return filtered list.
                # history.py currently returns oldest->newest (append order).
                
                # So we should reverse this list to match file-append behavior
                return result[::-1] 
        except Exception as e:
            logger.error(f"Failed to fetch queries from DB: {e}")
            return []

    def search_queries(self, search_term: str) -> List[Dict[str, Any]]:
        """Search queries by text."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT query, timestamp, results_summary, files_used
                    FROM query_history
                    WHERE lower(query) LIKE ?
                    ORDER BY id DESC
                    LIMIT 50
                """, (f"%{search_term.lower()}%",))
                
                rows = cursor.fetchall()
                result = []
                for row in rows:
                    result.append({
                        "query": row["query"],
                        "timestamp": row["timestamp"],
                        "results_summary": row["results_summary"],
                        "files_used": json.loads(row["files_used"] or "[]")
                    })
                return result
        except Exception as e:
            logger.error(f"Failed to search queries in DB: {e}")
            return []

    def clear_history(self) -> bool:
        """Clear all query history."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM query_history")
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return False

    def delete_query_by_index(self, index: int, limit: int = 100) -> Optional[Dict[str, Any]]:
        """
        Delete a query by its 'index' in the loaded list.
        This is tricky with DBs. We simulate it by fetching, getting ID, and deleting.
        Legacy support for /delete/{index}.
        """
        try:
            # 1. Fetch current list (to map index to ID)
            # Fetch in same order as get_recent_queries (reversed of desc -> asc)
            # Actually, get_recent_queries returns ASC order (oldest first)
            
            with self._get_connection() as conn:
                # Get IDs in ASC order
                cursor = conn.execute("SELECT id, query FROM query_history ORDER BY id ASC")
                rows = cursor.fetchall()
                
                # Apply limit logic (history.py only keeps last MAX_HISTORY_ITEMS)
                # But here we have everything.
                # If we mimic history.py, index 0 is the oldest.
                
                if index < 0 or index >= len(rows):
                    return None
                
                target_row = rows[index]
                row_id = target_row["id"]
                query_text = target_row["query"]
                
                # Delete
                conn.execute("DELETE FROM query_history WHERE id = ?", (row_id,))
                conn.commit()
                
                return {"query": query_text}
        except Exception as e:
            logger.error(f"Failed to delete query by index: {e}")
            return None

# Singleton instance
db_manager = DatabaseManager()

def get_db_manager():
    return db_manager
