"""
Analysis Manager for handling analysis cancellation and state tracking
"""
import threading
import time
import logging
from typing import Dict, Optional, Set
from uuid import uuid4
from fastapi import HTTPException

class AnalysisManager:
    """Manages running analyses and provides cancellation capabilities"""
    
    def __init__(self):
        self._running_analyses: Dict[str, Dict] = {}
        self._cancelled_analyses: Set[str] = set()
        self._lock = threading.Lock()
        
    def start_analysis(self, user_session: str = None) -> str:
        """Start a new analysis and return analysis ID"""
        analysis_id = str(uuid4())
        
        with self._lock:
            self._running_analyses[analysis_id] = {
                'id': analysis_id,
                'user_session': user_session,
                'start_time': time.time(),
                'status': 'running',
                'stage': 'initializing'
            }
        
        logging.info(f"[ANALYSIS_MANAGER] Started analysis {analysis_id} for session {user_session}")
        return analysis_id
    
    def update_analysis_stage(self, analysis_id: str, stage: str):
        """Update the current stage of analysis"""
        with self._lock:
            if analysis_id in self._running_analyses:
                self._running_analyses[analysis_id]['stage'] = stage
                self._running_analyses[analysis_id]['last_update'] = time.time()
                logging.info(f"[ANALYSIS_MANAGER] Analysis {analysis_id} stage: {stage}")
    
    def is_cancelled(self, analysis_id: str) -> bool:
        """Check if analysis has been cancelled"""
        with self._lock:
            return analysis_id in self._cancelled_analyses
    
    def cancel_analysis(self, analysis_id: str) -> bool:
        """Cancel a running analysis"""
        with self._lock:
            if analysis_id in self._running_analyses:
                self._cancelled_analyses.add(analysis_id)
                self._running_analyses[analysis_id]['status'] = 'cancelled'
                self._running_analyses[analysis_id]['cancelled_time'] = time.time()
                logging.info(f"[ANALYSIS_MANAGER] Cancelled analysis {analysis_id}")
                return True
            return False
    
    def complete_analysis(self, analysis_id: str):
        """Mark analysis as completed"""
        with self._lock:
            if analysis_id in self._running_analyses:
                self._running_analyses[analysis_id]['status'] = 'completed'
                self._running_analyses[analysis_id]['end_time'] = time.time()
                # Remove from cancelled set if it was there
                self._cancelled_analyses.discard(analysis_id)
                logging.info(f"[ANALYSIS_MANAGER] Completed analysis {analysis_id}")
    
    def get_analysis_status(self, analysis_id: str) -> Optional[Dict]:
        """Get current status of analysis"""
        with self._lock:
            return self._running_analyses.get(analysis_id)
    
    def get_running_analyses(self) -> Dict[str, Dict]:
        """Get all currently running analyses"""
        with self._lock:
            return {
                aid: info for aid, info in self._running_analyses.items() 
                if info['status'] == 'running'
            }
    
    def cleanup_old_analyses(self, max_age_hours: int = 24):
        """Clean up old analysis records"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        with self._lock:
            to_remove = []
            for analysis_id, info in self._running_analyses.items():
                age = current_time - info['start_time']
                if age > max_age_seconds and info['status'] in ['completed', 'cancelled']:
                    to_remove.append(analysis_id)
            
            for analysis_id in to_remove:
                del self._running_analyses[analysis_id]
                self._cancelled_analyses.discard(analysis_id)
            
            if to_remove:
                logging.info(f"[ANALYSIS_MANAGER] Cleaned up {len(to_remove)} old analyses")

# Global analysis manager instance
analysis_manager = AnalysisManager()

def check_cancellation(analysis_id: str):
    """Helper function to check if analysis should be cancelled"""
    if analysis_manager.is_cancelled(analysis_id):
        raise HTTPException(status_code=499, detail="Analysis was cancelled by user")