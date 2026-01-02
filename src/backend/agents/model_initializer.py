"""
Model Initializer Module
========================
Handles lazy loading and initialization of LLM models and related components.
Extracted from crew_manager.py for better maintainability.
"""

import logging
import os
from typing import Dict, Any, Optional

# Singleton instance
_model_initializer: Optional['ModelInitializer'] = None


class ModelInitializer:
    """
    Handles lazy loading of LLM models and related components.
    Uses singleton pattern to ensure only one instance exists.
    """
    
    def __init__(self):
        self._initialized = False
        self._llm_client = None
        self._chroma_client = None
        self._primary_llm = None
        self._review_llm = None
        self._intelligent_router = None

        self._tools = None
        self._query_parser = None
        self._cot_engine = None
        self._cot_config = None
        
        # Cache for loaded models
        self.cached_models: Dict[str, str] = {}
        
        logging.info("🔧 ModelInitializer created (lazy loading enabled)")
    
    @property
    def intelligent_router(self):
        """Get or create the intelligent router."""
        if self._intelligent_router is None:
            from backend.core.intelligent_router import get_intelligent_router
            self._intelligent_router = get_intelligent_router()
        return self._intelligent_router
    
    @property
    def llm_client(self):
        """Get or create LLM client."""
        if self._llm_client is None:
            from backend.core.llm_client import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client
    
    @property
    def chroma_client(self):
        """Get or create ChromaDB client."""
        if self._chroma_client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                persist_dir = os.path.join(
                    os.path.dirname(__file__), '..', '..', '..', 'data', 'chroma_db'
                )
                os.makedirs(persist_dir, exist_ok=True)
                
                # Updated for modern ChromaDB API (v0.4+)
                # Removed deprecated chroma_db_impl parameter
                self._chroma_client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logging.info("✅ ChromaDB client initialized (PersistentClient)")
            except Exception as e:
                logging.warning(f"ChromaDB initialization failed: {e}")
                # Return a mock client for graceful degradation
                self._chroma_client = None
        return self._chroma_client
    
    @property
    def primary_llm(self):
        """Get the primary LLM for analysis tasks."""
        if self._primary_llm is None:
            self._initialize_llms()
        return self._primary_llm
    
    @property
    def review_llm(self):
        """Get the review LLM for validation tasks."""
        if self._review_llm is None:
            self._initialize_llms()
        return self._review_llm
    

    @property
    def tools(self):
        """Get analysis tools."""
        if self._tools is None:
            self._tools = self._create_tools()
        return self._tools
    
    @property
    def query_parser(self):
        """Get the query parser."""
        if self._query_parser is None:
            from backend.core.query_parser import AdvancedQueryParser
            self._query_parser = AdvancedQueryParser(self.llm_client)
        return self._query_parser
    
    def _initialize_llms(self):
        """Initialize both primary and review LLMs."""
        try:
            # Use new langchain-ollama package (replaces deprecated langchain_community.llms.Ollama)
            try:
                from langchain_ollama import OllamaLLM
            except ImportError:
                # Fallback to old import if new package not installed
                from langchain_community.llms import Ollama as OllamaLLM
            
            from backend.core.user_preferences import get_preferences_manager
            from backend.core.model_selector import ModelSelector
            
            prefs = get_preferences_manager().load_preferences()
            # select_optimal_models returns tuple: (primary, review, embedding)
            primary_model, review_model, _ = ModelSelector.select_optimal_models()
            
            # Use preferences as fallback if selection returns None
            primary_model = primary_model or prefs.primary_model or 'llama3.1:8b'
            review_model = review_model or prefs.review_model or 'phi3:mini'
            
            # Store full model names
            self.cached_models['primary'] = primary_model
            self.cached_models['review'] = review_model
            
            # Strip ollama/ prefix for LangChain OllamaLLM class which expects just the model name
            primary_model_clean = primary_model.replace('ollama/', '')
            review_model_clean = review_model.replace('ollama/', '')
            
            self._primary_llm = OllamaLLM(model=primary_model_clean, timeout=120)
            
            self._review_llm = OllamaLLM(model=review_model_clean, timeout=60)
            
            logging.info(f"✅ LLMs initialized: primary={primary_model}, review={review_model}")
            
        except Exception as e:
            logging.error(f"❌ LLM initialization failed: {e}")
            raise
    
    def _create_tools(self) -> list:
        """Create analysis tools for agents."""
        # Genesis Compliance: Removed CrewAI tools.
        # Future tools should be custom implementations if needed.
        return []
    
    def ensure_initialized(self):
        """Ensure all components are initialized."""
        if self._initialized:
            return
        
        # Trigger lazy loading of key components
        _ = self.llm_client
        _ = self.primary_llm
        _ = self.review_llm

        
        self._initialized = True
        logging.info("✅ ModelInitializer fully initialized")
    
    def ensure_cot_engine(self):
        """Ensure Chain-of-Thought engine is initialized."""
        if self._cot_engine is None:
            config = self._load_cot_config()
            if config.get('enabled', False):
                from backend.core.self_correction_engine import SelfCorrectionEngine
                self._cot_engine = SelfCorrectionEngine(config)
        return self._cot_engine
    
    def _load_cot_config(self) -> Dict[str, Any]:
        """Load Chain-of-Thought configuration."""
        if self._cot_config is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', '..',
                'config', 'cot_review_config.json'
            )
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    self._cot_config = json.load(f)
            else:
                self._cot_config = {'enabled': False}
        return self._cot_config
    
    
    # Genesis Compliance: Removed CrewAI class getters (get_task_class, get_crew_class, get_process_class)


def get_model_initializer() -> ModelInitializer:
    """Get the singleton ModelInitializer instance."""
    global _model_initializer
    if _model_initializer is None:
        _model_initializer = ModelInitializer()
    return _model_initializer
