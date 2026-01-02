"""
Code Execution History Storage

Stores and retrieves code generation/execution history for user review and replay.
Enables users to see generated code, outputs, and recreate past analyses.

Author: Research Team
Date: December 27, 2025
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
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
        self.history_file = self.history_dir / 'code_execution_history.json'
        
        # Load existing history
        self._history: List[CodeExecutionRecord] = []
        self._load_history()
        
        logger.info(f"CodeExecutionHistory initialized with {len(self._history)} records")
    
    def _load_history(self) -> None:
        """Load history from disk"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._history = [
                        CodeExecutionRecord.from_dict(record) 
                        for record in data.get('executions', [])
                    ]
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to load history: {e}")
                self._history = []
    
    def _save_history(self) -> None:
        """Save history to disk"""
        try:
            data = {
                'version': '1.0',
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'total_executions': len(self._history),
                'executions': [record.to_dict() for record in self._history]
            }
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
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
        self._save_history()
        
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

def get_execution_history() -> CodeExecutionHistory:
    """Get singleton CodeExecutionHistory instance"""
    global _execution_history
    if _execution_history is None:
        _execution_history = CodeExecutionHistory()
    return _execution_history
