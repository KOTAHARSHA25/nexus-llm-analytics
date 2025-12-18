"""
CrewAI Import Optimization Manager
Handles expensive CrewAI imports with background loading and caching
"""

import threading
import time
import logging
from typing import Optional, Any, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor

class CrewAIImportManager:
    """
    Manages expensive CrewAI imports with optimization strategies:
    1. Background pre-loading during startup
    2. Import caching and reuse
    3. Lazy loading with fallback
    """
    
    def __init__(self):
        self._import_cache: Dict[str, Any] = {}
        self._import_lock = threading.Lock()
        self._import_status = {
            'crewai_loaded': False,
            'loading_in_progress': False,
            'load_start_time': None,
            'load_duration': None
        }
        self._background_loader: Optional[threading.Thread] = None
        
    def start_background_loading(self):
        """Start background loading of CrewAI components during startup"""
        if self._background_loader and self._background_loader.is_alive():
            logging.debug("Background CrewAI loading already in progress")
            return
            
        logging.debug("Starting background CrewAI import optimization...")
        self._background_loader = threading.Thread(
            target=self._background_import_crewai,
            daemon=True,
            name="CrewAI-Background-Loader"
        )
        self._background_loader.start()
        
    def _background_import_crewai(self):
        """Background thread to pre-load CrewAI components"""
        with self._import_lock:
            if self._import_status['crewai_loaded']:
                return
                
            self._import_status['loading_in_progress'] = True
            self._import_status['load_start_time'] = time.perf_counter()
            
        try:
            logging.debug("Pre-loading CrewAI components in background...")
            
            # Import CrewAI components
            from crewai import Agent, Task, Crew
            from crewai.process import Process
            
            # Cache the imported modules
            with self._import_lock:
                self._import_cache['Agent'] = Agent
                self._import_cache['Task'] = Task
                self._import_cache['Crew'] = Crew
                self._import_cache['Process'] = Process
                
                self._import_status['crewai_loaded'] = True
                self._import_status['loading_in_progress'] = False
                load_end_time = time.perf_counter()
                self._import_status['load_duration'] = load_end_time - self._import_status['load_start_time']
                
            logging.debug(f"CrewAI pre-loaded successfully in {self._import_status['load_duration']:.2f}s")
            
        except Exception as e:
            logging.error(f"❌ Background CrewAI import failed: {e}")
            with self._import_lock:
                self._import_status['loading_in_progress'] = False
                self._import_status['crewai_loaded'] = False
    
    def get_crewai_components(self, timeout: float = 60.0):
        """
        Get CrewAI components, waiting for background loading if necessary
        
        Args:
            timeout: Maximum time to wait for background loading
            
        Returns:
            Dict with CrewAI components or None if failed
        """
        start_wait = time.perf_counter()
        
        # Wait for background loading to complete
        while (time.perf_counter() - start_wait) < timeout:
            with self._import_lock:
                if self._import_status['crewai_loaded']:
                    logging.info("🎯 Using pre-loaded CrewAI components")
                    return {
                        'Agent': self._import_cache['Agent'],
                        'Task': self._import_cache['Task'],
                        'Crew': self._import_cache['Crew'],
                        'Process': self._import_cache['Process']
                    }
                elif not self._import_status['loading_in_progress']:
                    # Background loading failed or wasn't started, do synchronous import
                    break
                    
            # Wait a bit before checking again
            time.sleep(0.1)
        
        # Fallback: synchronous import if background loading didn't work
        logging.warning("⚠️ Background loading timeout/failed, falling back to synchronous import")
        return self._synchronous_import()
    
    def _synchronous_import(self):
        """Fallback synchronous import of CrewAI components"""
        try:
            logging.info("📦 Performing synchronous CrewAI import...")
            start_time = time.perf_counter()
            
            from crewai import Agent, Task, Crew
            from crewai.process import Process
            
            duration = time.perf_counter() - start_time
            logging.info(f"✅ Synchronous CrewAI import completed in {duration:.2f}s")
            
            return {
                'Agent': Agent,
                'Task': Task,
                'Crew': Crew,
                'Process': Process
            }
            
        except Exception as e:
            logging.error(f"❌ Synchronous CrewAI import failed: {e}")
            return None
    
    def is_loaded(self) -> bool:
        """Check if CrewAI components are loaded and ready"""
        with self._import_lock:
            return self._import_status['crewai_loaded']
    
    def get_status(self) -> Dict:
        """Get current loading status"""
        with self._import_lock:
            return self._import_status.copy()

# Global instance
_crewai_import_manager = None

def get_crewai_import_manager() -> CrewAIImportManager:
    """Get global CrewAI import manager instance"""
    global _crewai_import_manager
    if _crewai_import_manager is None:
        _crewai_import_manager = CrewAIImportManager()
    return _crewai_import_manager

def start_crewai_preloading():
    """Start CrewAI pre-loading (call during application startup)"""
    manager = get_crewai_import_manager()
    manager.start_background_loading()
    
def get_crewai_components(timeout: float = 60.0):
    """Get CrewAI components with optimization"""
    manager = get_crewai_import_manager()
    return manager.get_crewai_components(timeout)