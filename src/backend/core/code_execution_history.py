"""
Code Execution History — Nexus LLM Analytics v2.0
=================================================

Persistent storage for code generation / execution history, enabling
users to review generated code, inspect outputs, and replay past
analyses.

Enterprise v2.0 Additions
-------------------------
* **ExecutionAnalytics** — Aggregated metrics (success rate by model,
  average execution time, error frequency breakdown).
* **HistoryRetentionPolicy** — Configurable TTL-based and count-based
  retention with automatic pruning.
* Thread-safe singleton accessor ``get_execution_history()`` now
  uses double-checked locking.

Backward Compatibility
----------------------
All v1.x classes (``CodeExecutionRecord``, ``CodeExecutionHistory``)
and the ``get_execution_history()`` accessor retain their original
signatures.

.. versionchanged:: 2.0
   Added analytics, retention policies, and thread-safe singleton.

Author: Nexus Analytics Research Team
Date: February 2026
"""

import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CodeExecutionRecord:
    """A single code execution record with all details for replay"""
    
    # Unique identifiers
    execution_id: str
    timestamp: str
    
    # Query context
    query: str
    model_used: str
    
    # Generated code
    generated_code: str
    cleaned_code: str
    
    # Execution details
    success: bool
    execution_time_ms: float
    
    # Results
    result: Any = None
    result_type: str = "unknown"
    result_preview: str = ""
    error: Optional[str] = None
    
    # Data context (for replay)
    data_file: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    
    # Metadata
    attempt_count: int = 1
    retry_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        data = asdict(self)
        # Convert result to string representation if complex
        if data['result'] is not None:
            try:
                # Try to serialize, if fails use repr
                json.dumps(data['result'])
            except (TypeError, ValueError):
                data['result'] = str(data['result'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeExecutionRecord':
        """Create from dictionary"""
        return cls(**data)


class CodeExecutionHistory:
    """
    Manages code execution history storage and retrieval.
    
    Features:
    - Save every code execution with full context
    - Retrieve past executions by ID, query, or time range
    - Replay past executions with same code
    - Search through execution history
    """
    
    def __init__(self, history_dir: Optional[Path] = None):
        """
        Initialize the history manager.
        
        Args:
            history_dir: Directory to store history files (default: data/history)
        """
        if history_dir is None:
            # Default to project's data/history folder
            self.history_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'history'
        else:
            self.history_dir = Path(history_dir)
        
        self.history_dir.mkdir(parents=True, exist_ok=True)
        # Use .jsonl for append-only storage
        self.history_file = self.history_dir / 'code_execution_history.jsonl'
        self.legacy_history_file = self.history_dir / 'code_execution_history.json'
        
        # Load existing history
        self._history: List[CodeExecutionRecord] = []
        self._load_history()
        
        logger.info(f"CodeExecutionHistory initialized with {len(self._history)} records")
    
    def _load_history(self) -> None:
        """Load history from JSONL (preferred) or JSON (legacy migration)"""
        # 1. Try JSONL first
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            record_dict = json.loads(line)
                            self._history.append(CodeExecutionRecord.from_dict(record_dict))
                return
            except Exception as e:
                logger.warning(f"Failed to load JSONL history: {e}")
        
        # 2. Fallback to Legacy JSON (Migration)
        if self.legacy_history_file.exists():
            try:
                logger.info("Migrating legacy JSON history to JSONL...")
                with open(self.legacy_history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._history = [
                        CodeExecutionRecord.from_dict(record) 
                        for record in data.get('executions', [])
                    ]
                # Trigger save to create JSONL
                self._save_history(append_new_only=False)
                # Rename legacy to backup
                self.legacy_history_file.rename(self.legacy_history_file.with_suffix('.json.bak'))
            except Exception as e:
                logger.error(f"Failed to migrate legacy history: {e}")

    def _save_history(self, append_new_only: bool = False) -> None:
        """
        Save history to disk.
        Args:
           append_new_only: If True, only append the last record to the file.
        """
        try:
            if append_new_only and self._history:
                # Optimized append
                with open(self.history_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(self._history[-1].to_dict(), default=str) + '\n')
            else:
                # Full rewrite (e.g. after pruning)
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    for record in self._history:
                        f.write(json.dumps(record.to_dict(), default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def _generate_execution_id(self, query: str, timestamp: str) -> str:
        """Generate unique execution ID"""
        unique_string = f"{query}_{timestamp}_{len(self._history)}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def _get_result_preview(self, result: Any, max_length: int = 500) -> str:
        """Generate a preview string of the result"""
        if result is None:
            return "None"
        
        result_str = str(result)
        if len(result_str) > max_length:
            return result_str[:max_length] + "..."
        return result_str
    
    def _get_result_type(self, result: Any) -> str:
        """Determine the type of result for display"""
        if result is None:
            return "none"
        
        type_name = type(result).__name__
        
        # Check for common data types
        try:
            import pandas as pd
            if isinstance(result, pd.DataFrame):
                return f"dataframe ({len(result)} rows)"
            elif isinstance(result, pd.Series):
                return f"series ({len(result)} items)"
        except ImportError:
            pass
        
        if isinstance(result, dict):
            return f"dict ({len(result)} keys)"
        elif isinstance(result, (list, tuple)):
            return f"{type_name} ({len(result)} items)"
        elif isinstance(result, (int, float)):
            return "number"
        elif isinstance(result, str):
            return "string"
        
        return type_name
    
    def save_execution(self,
                       query: str,
                       model_used: str,
                       generated_code: str,
                       cleaned_code: str,
                       success: bool,
                       execution_time_ms: float,
                       result: Any = None,
                       error: Optional[str] = None,
                       data_file: Optional[str] = None,
                       columns: Optional[List[str]] = None,
                       row_count: int = 0,
                       attempt_count: int = 1,
                       retry_errors: Optional[List[str]] = None) -> str:
        """
        Save a code execution to history.
        
        Returns:
            execution_id: Unique ID for this execution
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        execution_id = self._generate_execution_id(query, timestamp)
        
        record = CodeExecutionRecord(
            execution_id=execution_id,
            timestamp=timestamp,
            query=query,
            model_used=model_used,
            generated_code=generated_code,
            cleaned_code=cleaned_code,
            success=success,
            execution_time_ms=execution_time_ms,
            result=result,
            result_type=self._get_result_type(result),
            result_preview=self._get_result_preview(result),
            error=error,
            data_file=data_file,
            columns=columns or [],
            row_count=row_count,
            attempt_count=attempt_count,
            retry_errors=retry_errors or []
        )
        
        self._history.append(record)
        self._save_history(append_new_only=True)
        
        # Enforce retention policy every 50 saves to prevent unbounded growth
        if len(self._history) % 50 == 0:
            try:
                policy = HistoryRetentionPolicy(max_age_days=90, max_records=5000)
                removed = policy.enforce(self)
                if removed:
                    logger.info("Auto-pruned %d old history records", removed)
            except Exception as e:
                logger.debug("Retention policy enforcement skipped: %s", e)
        
        logger.info(f"Saved execution {execution_id}: {query[:50]}...")
        return execution_id
    
    def get_execution(self, execution_id: str) -> Optional[CodeExecutionRecord]:
        """Get a specific execution by ID"""
        for record in self._history:
            if record.execution_id == execution_id:
                return record
        return None
    
    def get_recent_executions(self, limit: int = 20) -> List[CodeExecutionRecord]:
        """Get the most recent executions"""
        return list(reversed(self._history[-limit:]))
    
    def get_successful_executions(self, limit: int = 20) -> List[CodeExecutionRecord]:
        """Get only successful executions"""
        successful = [r for r in self._history if r.success]
        return list(reversed(successful[-limit:]))
    
    def search_executions(self, 
                         query_contains: Optional[str] = None,
                         model: Optional[str] = None,
                         success_only: bool = False,
                         limit: int = 50) -> List[CodeExecutionRecord]:
        """
        Search through execution history.
        
        Args:
            query_contains: Filter by query containing this text
            model: Filter by model used
            success_only: Only return successful executions
            limit: Maximum results to return
        """
        results = []
        
        for record in reversed(self._history):
            if len(results) >= limit:
                break
            
            # Apply filters
            if success_only and not record.success:
                continue
            if model and record.model_used != model:
                continue
            if query_contains and query_contains.lower() not in record.query.lower():
                continue
            
            results.append(record)
        
        return results
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary statistics of execution history"""
        if not self._history:
            return {
                'total_executions': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0.0,
                'avg_execution_time_ms': 0.0,
                'models_used': [],
                'unique_queries': 0
            }
        
        successful = sum(1 for r in self._history if r.success)
        total = len(self._history)
        
        return {
            'total_executions': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': (successful / total) * 100 if total > 0 else 0.0,
            'avg_execution_time_ms': sum(r.execution_time_ms for r in self._history) / total,
            'models_used': list(set(r.model_used for r in self._history)),
            'unique_queries': len(set(r.query for r in self._history))
        }
    
    def clear_history(self) -> int:
        """Clear all history. Returns count of cleared records."""
        count = len(self._history)
        self._history = []
        self._save_history()
        logger.info(f"Cleared {count} execution records")
        return count
    
    def export_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Export a single execution for sharing/replay"""
        record = self.get_execution(execution_id)
        if not record:
            return None
        
        return {
            'export_version': '1.0',
            'exported_at': datetime.now(timezone.utc).isoformat(),
            'execution': record.to_dict()
        }
    
    def get_code_for_replay(self, execution_id: str) -> Optional[Dict[str, str]]:
        """Get the code and context needed to replay an execution"""
        record = self.get_execution(execution_id)
        if not record:
            return None
        
        return {
            'execution_id': record.execution_id,
            'query': record.query,
            'cleaned_code': record.cleaned_code,
            'original_code': record.generated_code,
            'data_file': record.data_file,
            'columns': record.columns,
            'model_used': record.model_used
        }


# Singleton instance
_execution_history: Optional[CodeExecutionHistory] = None
_execution_history_lock = threading.Lock()

def get_execution_history() -> CodeExecutionHistory:
    """Return the singleton :class:`CodeExecutionHistory` instance.

    Thread-safe with double-checked locking.

    Returns:
        The shared ``CodeExecutionHistory`` instance.
    """
    global _execution_history
    if _execution_history is None:
        with _execution_history_lock:
            if _execution_history is None:
                _execution_history = CodeExecutionHistory()
    return _execution_history


# ============================================================================
# Enterprise v2.0 — Analytics & Retention
# ============================================================================


class ExecutionAnalytics:
    """Aggregated analytics over execution history.

    Provides derived metrics useful for dashboards and observability:
    per-model success rates, average execution times, and error
    frequency breakdowns.

    Args:
        history: The :class:`CodeExecutionHistory` instance to analyse.

    Example::

        analytics = ExecutionAnalytics(get_execution_history())
        print(analytics.success_rate_by_model())

    .. versionadded:: 2.0
    """

    def __init__(self, history: CodeExecutionHistory) -> None:
        self._history = history

    def success_rate_by_model(self) -> Dict[str, float]:
        """Return success rate (0–100) keyed by model name."""
        model_stats: Dict[str, Dict[str, int]] = {}
        for r in self._history._history:
            m = r.model_used
            if m not in model_stats:
                model_stats[m] = {"ok": 0, "total": 0}
            model_stats[m]["total"] += 1
            if r.success:
                model_stats[m]["ok"] += 1
        return {
            m: (s["ok"] / s["total"] * 100) if s["total"] else 0.0
            for m, s in model_stats.items()
        }

    def avg_execution_time_by_model(self) -> Dict[str, float]:
        """Return average execution time (ms) keyed by model name."""
        model_times: Dict[str, List[float]] = {}
        for r in self._history._history:
            model_times.setdefault(r.model_used, []).append(r.execution_time_ms)
        return {
            m: sum(ts) / len(ts) if ts else 0.0
            for m, ts in model_times.items()
        }

    def error_frequency(self) -> Dict[str, int]:
        """Return error string frequency from failed executions."""
        freq: Dict[str, int] = {}
        for r in self._history._history:
            if not r.success and r.error:
                # Normalise error to first 80 chars for grouping
                key = r.error[:80]
                freq[key] = freq.get(key, 0) + 1
        return dict(sorted(freq.items(), key=lambda kv: kv[1], reverse=True))


class HistoryRetentionPolicy:
    """Configurable retention policy for execution history.

    Supports both age-based (TTL) and count-based limits.  Call
    :meth:`enforce` periodically (e.g. via a background task) to prune
    stale records.

    Args:
        max_age_days: Records older than this are pruned.
        max_records: Hard cap on total record count.

    Example::

        policy = HistoryRetentionPolicy(max_age_days=30, max_records=5000)
        removed = policy.enforce(get_execution_history())

    .. versionadded:: 2.0
    """

    def __init__(self, max_age_days: int = 90, max_records: int = 10000) -> None:
        self.max_age_days = max_age_days
        self.max_records = max_records

    def enforce(self, history: CodeExecutionHistory) -> int:
        """Prune records exceeding the retention limits.

        Args:
            history: The history instance to prune.

        Returns:
            Number of records removed.
        """
        before = len(history._history)
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_age_days)
        cutoff_iso = cutoff.isoformat()

        # Age-based pruning
        history._history = [
            r for r in history._history
            if r.timestamp >= cutoff_iso
        ]

        # Count-based pruning (keep most recent)
        if len(history._history) > self.max_records:
            history._history = history._history[-self.max_records:]

        removed = before - len(history._history)
        if removed:
            history._save_history()
            logger.info("HistoryRetentionPolicy: pruned %d records", removed)
        return removed
