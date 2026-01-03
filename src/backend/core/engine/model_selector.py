"""
Unified Model Selection System
==============================
Consolidates model discovery, RAM-aware selection, and smart fallback.

This module combines functionality from:
- model_selector (original)
- model_discovery (Phase 1)
- ram_aware_selector (Phase 1)
- model_detector (deprecated, merged)

Design Principles:
- Zero Hardcoding: All models discovered dynamically from Ollama
- RAM-Aware: Real-time memory monitoring prevents OOM
- Adaptive: Automatically downgrades under memory pressure
- Domain Agnostic: Works with any installed models
"""

import psutil
import logging
import os
import time
import re
import threading
import httpx
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Dict, List, Any, Optional
from .user_preferences import get_preferences_manager

logger = logging.getLogger(__name__)

# =============================================================================
# CACHES (Performance optimization)
# =============================================================================
_system_memory_cache = {"data": None, "timestamp": 0, "ttl": 300}
_model_selection_cache = {"data": None, "timestamp": 0, "ttl": 300}
_compatibility_cache = {}


# =============================================================================
# ENUMS AND DATA CLASSES (from model_discovery + ram_aware_selector)
# =============================================================================

class ModelCapability(Enum):
    """Model capability categories"""
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    INSTRUCTION_FOLLOWING = "instruction_following"
    CONVERSATION = "conversation"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    MATH = "math"


class MemoryPressureLevel(Enum):
    """System memory pressure levels"""
    LOW = "low"           # Plenty of RAM available (>60% free)
    MODERATE = "moderate" # Normal usage (30-60% free)
    HIGH = "high"         # Getting tight (15-30% free)
    CRITICAL = "critical" # Very low (<15% free)


@dataclass
class ModelInfo:
    """Information about a discovered model"""
    name: str
    size_bytes: int
    parameter_count: Optional[str] = None
    family: Optional[str] = None
    quantization: Optional[str] = None
    modified_at: Optional[str] = None
    capabilities: List[ModelCapability] = field(default_factory=list)
    estimated_ram_gb: float = 0.0
    complexity_score: float = 0.5
    
    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size_gb": round(self.size_gb, 2),
            "parameter_count": self.parameter_count,
            "family": self.family,
            "quantization": self.quantization,
            "estimated_ram_gb": round(self.estimated_ram_gb, 2),
            "complexity_score": round(self.complexity_score, 3),
            "capabilities": [c.value for c in self.capabilities]
        }


@dataclass
class MemorySnapshot:
    """Snapshot of system memory state"""
    timestamp: float
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    swap_used_gb: float
    pressure_level: MemoryPressureLevel
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_gb": round(self.total_gb, 2),
            "available_gb": round(self.available_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "percent_used": round(self.percent_used, 1),
            "swap_used_gb": round(self.swap_used_gb, 2),
            "pressure_level": self.pressure_level.value
        }


@dataclass
class ModelSelectionResult:
    """Result of model selection process"""
    selected_model: str
    reason: str
    available_ram_gb: float
    estimated_model_ram_gb: float
    safety_margin_gb: float
    pressure_level: MemoryPressureLevel
    fallback_triggered: bool = False
    original_model: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "selected_model": self.selected_model,
            "reason": self.reason,
            "available_ram_gb": round(self.available_ram_gb, 2),
            "estimated_model_ram_gb": round(self.estimated_model_ram_gb, 2),
            "safety_margin_gb": round(self.safety_margin_gb, 2),
            "pressure_level": self.pressure_level.value,
            "fallback_triggered": self.fallback_triggered,
            "original_model": self.original_model
        }

class ModelSelector:
    """
    Intelligent model selection based on system resources and model requirements.
    Ensures optimal performance while preventing out-of-memory errors.
    """
    
    # NO HARDCODED MODELS - Fetch dynamically from Ollama
    MODEL_REQUIREMENTS = {}  # Will be populated dynamically
    
    @staticmethod
    def _get_installed_models() -> Dict[str, Dict]:
        """Fetch all installed models from Ollama dynamically (NO HARDCODING)"""
        import requests
        from backend.core.config import get_settings
        
        try:
            settings = get_settings()
            ollama_url = settings.ollama_base_url
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            models_data = response.json().get("models", [])
            
            models_info = {}
            for model in models_data:
                model_name = model.get("name", "")
                size_bytes = model.get("size", 0)
                size_gb = size_bytes / (1024**3)
                
                # Estimate RAM requirements (roughly 1.2x model size for inference)
                min_ram = max(size_gb * 1.2, 0.5)  # At least 500MB
                recommended_ram = size_gb * 1.5
                
                is_embedding = "embed" in model_name.lower()
                
                models_info[model_name] = {
                    "min_ram_gb": min_ram,
                    "recommended_ram_gb": recommended_ram,
                    "size_gb": size_gb,
                    "description": f"{'Embedding model' if is_embedding else 'Text generation model'} ({size_gb:.1f}GB)",
                    "capabilities": ["embeddings"] if is_embedding else ["text_generation", "analysis"],
                    "is_embedding": is_embedding
                }
            
            logging.debug(f"Found {len(models_info)} installed models from Ollama")
            return models_info
            
        except Exception as e:
            # Log the actual error to help debugging
            logging.error(f"Could not fetch models from Ollama at {ollama_url}: {e}")
            return {}
    
    @staticmethod
    def get_system_memory() -> Dict[str, float]:
        """Get system memory information in GB (cached for performance)"""
        global _system_memory_cache
        
        current_time = time.time()
        
        # Return cached data if still valid
        if (_system_memory_cache["data"] is not None and 
            current_time - _system_memory_cache["timestamp"] < _system_memory_cache["ttl"]):
            return _system_memory_cache["data"]
        
        # Fetch fresh system memory data
        memory = psutil.virtual_memory()
        memory_data = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent
        }
        
        # Cache the result
        _system_memory_cache["data"] = memory_data
        _system_memory_cache["timestamp"] = current_time
        
        return memory_data
    
    @staticmethod
    def select_optimal_models() -> Tuple[str, str, str]:
        """
        Select optimal models based on available system memory and user preferences.
        DYNAMICALLY fetches installed models - NO HARDCODING!
        
        Returns:
            Tuple[primary_model, review_model, embedding_model]
        """
        memory_info = ModelSelector.get_system_memory()
        available_ram = memory_info["available_gb"]
        total_ram = memory_info["total_gb"]
        
        logging.info(f"System Memory: {total_ram:.1f}GB total, {available_ram:.1f}GB available")
        
        # Fetch installed models dynamically
        installed_models = ModelSelector._get_installed_models()
        if not installed_models:
            # Don't log as error - Ollama not running is expected during development
            logging.debug("No models found in Ollama (Ollama not running or no models installed)")
            raise RuntimeError("No models installed. Run: ollama pull <model-name>")
        
        # Update MODEL_REQUIREMENTS with actual installed models
        ModelSelector.MODEL_REQUIREMENTS = installed_models
        
        # Get user preferences
        preferences_manager = get_preferences_manager()
        user_prefs = preferences_manager.load_preferences()
        
        # If user has preferences set, use them
        if user_prefs.primary_model and user_prefs.review_model and user_prefs.embedding_model:
            preferred_primary = user_prefs.primary_model
            preferred_review = user_prefs.review_model
            preferred_embedding = user_prefs.embedding_model
            auto_selection = user_prefs.auto_model_selection
        else:
            # No preferences set - auto-select from installed models
            logging.info("No user preferences found - auto-selecting from installed models")
            auto_selection = True
            
            # Find first non-embedding model for primary
            non_embedding = [m for m, info in installed_models.items() if not info.get("is_embedding", False)]
            embedding_models = [m for m, info in installed_models.items() if info.get("is_embedding", False)]
            
            if not non_embedding:
                raise RuntimeError("No text generation models installed! Install: ollama pull llama3.1:8b")
            if not embedding_models:
                logging.warning("No embedding models found - RAG will not work properly")
                embedding_models = [non_embedding[0]]  # Fallback
            
            # Sort by size (larger = better quality)
            non_embedding.sort(key=lambda m: installed_models[m]["size_gb"], reverse=True)
            
            preferred_primary = non_embedding[0]
            preferred_review = non_embedding[-1] if len(non_embedding) > 1 else non_embedding[0]
            preferred_embedding = embedding_models[0]
        
        if not auto_selection:
            # Use exact user configuration
            primary = f"ollama/{preferred_primary}"
            review = f"ollama/{preferred_review}"
            embedding = f"ollama/{preferred_embedding}"
            
            logging.debug(f"Using user-selected models: Primary={preferred_primary}, Review={preferred_review}, Embedding={preferred_embedding}")
            return primary, review, embedding
        
        # Smart selection with memory validation
        allow_swap = user_prefs.allow_swap_usage if hasattr(user_prefs, 'allow_swap_usage') else False
        
        primary_model = ModelSelector._select_best_model(
            preferred_primary, "primary", available_ram, total_ram, allow_swap, installed_models
        )
        review_model = ModelSelector._select_best_model(
            preferred_review, "review", available_ram, total_ram, allow_swap, installed_models
        )
        embedding_model = f"ollama/{preferred_embedding}"
        
        return primary_model, review_model, embedding_model
    
    @staticmethod
    def _select_best_model(preferred: str, role: str, available_ram: float, total_ram: float, allow_swap: bool, installed_models: Dict) -> str:
        """Select the best model based on preferences and memory constraints - NO HARDCODING"""
        
        # Clean model name
        clean_preferred = preferred.replace("ollama/", "")
        
        # Check if preferred model is installed and can fit
        if clean_preferred in installed_models:
            required_ram = installed_models[clean_preferred]["min_ram_gb"]
            
            if available_ram >= required_ram:
                logging.debug(f"Using preferred {role} model: {clean_preferred} (needs {required_ram:.1f}GB, have {available_ram:.1f}GB)")
                return f"ollama/{clean_preferred}"
            elif allow_swap and total_ram >= required_ram:
                logging.warning(f"Using {role} model with swap: {clean_preferred} (needs {required_ram:.1f}GB, have {available_ram:.1f}GB available)")
                logging.warning(f"Performance will be slower due to swap usage")
                return f"ollama/{clean_preferred}"
            else:
                logging.debug(f"Cannot use preferred {role} model: {clean_preferred} (needs {required_ram:.1f}GB, have {available_ram:.1f}GB)")
        
        # Dynamic fallback - find smallest non-embedding model that fits
        non_embedding = [(name, info) for name, info in installed_models.items() if not info.get("is_embedding", False)]
        
        if not non_embedding:
            raise RuntimeError("No text generation models available!")
        
        # Sort by RAM requirement (smallest first for fallback)
        non_embedding.sort(key=lambda x: x[1]["min_ram_gb"])
        
        # Try each model in order
        for model_name, model_info in non_embedding:
            required_ram = model_info["min_ram_gb"]
            
            if available_ram >= required_ram:
                logging.debug(f"Fallback to {role} model: {model_name} (needs {required_ram:.1f}GB, have {available_ram:.1f}GB)")
                return f"ollama/{model_name}"
            elif allow_swap and total_ram >= required_ram:
                logging.debug(f"Fallback to {role} model with swap: {model_name}")
                return f"ollama/{model_name}"
        
        # If no models fit in memory, use the smallest one anyway (will be slow)
        smallest_model = non_embedding[0][0]
        logging.debug(f"No models fit in available RAM ({available_ram:.1f}GB), using smallest: {smallest_model}")
        return f"ollama/{smallest_model}"
    

    
    @staticmethod
    def validate_model_compatibility(model_name: str) -> Tuple[bool, str]:
        """
        Validate if a model can run on the current system (cached for performance).
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Tuple[is_compatible, message]
        """
        global _compatibility_cache
        
        # Check cache first (compatibility rarely changes during runtime)
        if model_name in _compatibility_cache:
            return _compatibility_cache[model_name]
        
        # Clean model name (remove ollama/ prefix if present)
        clean_name = model_name.replace("ollama/", "")
        
        # Fetch installed models dynamically
        installed_models = ModelSelector._get_installed_models()
        
        if clean_name not in installed_models:
            result = (False, f"❌ Model '{clean_name}' not installed")
            _compatibility_cache[model_name] = result
            return result
        
        model_info = installed_models[clean_name]
        memory_info = ModelSelector.get_system_memory()
        
        required_ram = model_info["min_ram_gb"]
        available_ram = memory_info["available_gb"]
        
        # Add small buffer tolerance (50MB) for borderline cases
        buffer_gb = 0.05
        
        if available_ram >= (required_ram - buffer_gb):
            status = "✅" if available_ram >= required_ram else "⚠️"
            result = (True, f"{status} {clean_name} compatible (needs {required_ram:.1f}GB, have {available_ram:.1f}GB)")
        else:
            result = (False, f"❌ {clean_name} incompatible (needs {required_ram:.1f}GB, only {available_ram:.1f}GB available)")
        
        # Cache the result for future calls
        _compatibility_cache[model_name] = result
        return result
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict:
        """Get detailed information about a model"""
        clean_name = model_name.replace("ollama/", "")
        return ModelSelector.MODEL_REQUIREMENTS.get(clean_name, {
            "min_ram_gb": 0,
            "description": f"Unknown model: {clean_name}",
            "capabilities": ["unknown"]
        })
    
    @staticmethod
    def recommend_system_config() -> Dict[str, any]:
        """Provide system configuration recommendations"""
        memory_info = ModelSelector.get_system_memory()
        total_ram = memory_info["total_gb"]
        available_ram = memory_info["available_gb"]
        
        recommendations = {
            "current_config": {
                "total_ram_gb": total_ram,
                "available_ram_gb": available_ram,
                "optimal_models": ModelSelector.select_optimal_models()
            },
            "recommendations": []
        }
        
        if total_ram < 8:
            recommendations["recommendations"].append({
                "type": "hardware_upgrade",
                "priority": "high",
                "message": f"Consider upgrading to 16GB+ RAM for optimal performance with Llama 3.1 8B",
                "current": f"{total_ram:.1f}GB",
                "recommended": "16GB+"
            })
        
        if available_ram < total_ram * 0.5:
            recommendations["recommendations"].append({
                "type": "memory_optimization",
                "priority": "medium",
                "message": "Close unnecessary applications to free up memory",
                "current": f"{available_ram:.1f}GB available",
                "target": f"{total_ram * 0.7:.1f}GB available"
            })
        
        return recommendations


# =============================================================================
# DYNAMIC MODEL DISCOVERY (merged from model_discovery.py)
# =============================================================================

class DynamicModelDiscovery:
    """
    Discovers and catalogs available LLM models dynamically.
    No hardcoding - queries Ollama API directly.
    """
    
    def __init__(self, 
                 ollama_base_url: str = "http://localhost:11434",
                 cache_ttl: int = 300):
        self.ollama_base_url = ollama_base_url
        self.cache_ttl = cache_ttl
        self._model_cache: Dict[str, ModelInfo] = {}
        self._cache_timestamp: float = 0
        self._discovery_in_progress: bool = False
        
        # Capability patterns (regex, not hardcoded model names)
        self._capability_patterns = {
            ModelCapability.CODE_GENERATION: [r"code", r"starcoder", r"codellama", r"deepseek.*coder"],
            ModelCapability.MATH: [r"math", r"wizard.*math", r"metamath"],
            ModelCapability.REASONING: [r"llama", r"phi", r"mistral", r"qwen"],
            ModelCapability.INSTRUCTION_FOLLOWING: [r"instruct", r"chat", r"alpaca"],
        }
        
        logger.info(f"DynamicModelDiscovery initialized with endpoint: {ollama_base_url}")
    
    def discover_models_sync(self, force_refresh: bool = False) -> List[ModelInfo]:
        """Synchronous model discovery"""
        current_time = time.time()
        if not force_refresh and self._model_cache:
            if current_time - self._cache_timestamp < self.cache_ttl:
                return list(self._model_cache.values())
        
        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(f"{self.ollama_base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
            
            analyzed_models = []
            for model in models:
                model_info = self._analyze_model(model)
                analyzed_models.append(model_info)
                self._model_cache[model_info.name] = model_info
            
            self._cache_timestamp = current_time
            analyzed_models.sort(key=lambda m: m.complexity_score, reverse=True)
            
            logger.info(f"Discovered {len(analyzed_models)} models")
            return analyzed_models
            
        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
            return list(self._model_cache.values()) if self._model_cache else []
    
    def _analyze_model(self, model_data: dict) -> ModelInfo:
        """Analyze a model and extract characteristics"""
        name = model_data.get("name", "unknown")
        size_bytes = model_data.get("size", 0)
        modified_at = model_data.get("modified_at")
        
        param_count = self._extract_param_count(name)
        family = self._extract_family(name)
        quantization = self._extract_quantization(name)
        estimated_ram = self._estimate_ram_requirement(size_bytes, param_count)
        capabilities = self._detect_capabilities(name, family)
        complexity_score = self._calculate_complexity_score(param_count, size_bytes, len(capabilities))
        
        return ModelInfo(
            name=name, size_bytes=size_bytes, parameter_count=param_count,
            family=family, quantization=quantization, modified_at=modified_at,
            capabilities=capabilities, estimated_ram_gb=estimated_ram,
            complexity_score=complexity_score
        )
    
    def _extract_param_count(self, name: str) -> Optional[str]:
        patterns = [r'(\d+(?:\.\d+)?)[bB]', r'(\d+)x(\d+)[bB]']
        for pattern in patterns:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                return match.group(0).upper()
        return None
    
    def _extract_family(self, name: str) -> Optional[str]:
        name_lower = name.lower()
        patterns = [r'^(llama)[\d\.]', r'^(phi)[\d\.]', r'^(mistral)', r'^(qwen)[\d\.]',
                   r'^(gemma)', r'^(codellama)', r'^(deepseek)', r'^(tinyllama)', r'^(orca)']
        for pattern in patterns:
            match = re.match(pattern, name_lower)
            if match:
                return match.group(1)
        parts = re.split(r'[\d:\-_]', name_lower)
        return parts[0] if parts else None
    
    def _extract_quantization(self, name: str) -> Optional[str]:
        patterns = [r'(q\d+_\w+)', r'(fp16)', r'(f16)', r'(int4)', r'(int8)']
        for pattern in patterns:
            match = re.search(pattern, name.lower())
            if match:
                return match.group(1).upper()
        return None
    
    def _estimate_ram_requirement(self, size_bytes: int, param_count: Optional[str]) -> float:
        base_estimate = (size_bytes / (1024 ** 3)) * 1.2
        if param_count:
            match = re.search(r'(\d+(?:\.\d+)?)', param_count)
            if match:
                params_b = float(match.group(1))
                param_estimate = params_b * 0.75
                return (base_estimate + param_estimate) / 2
        return base_estimate
    
    def _detect_capabilities(self, name: str, family: Optional[str]) -> List[ModelCapability]:
        capabilities = []
        name_lower = name.lower()
        for capability, patterns in self._capability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    if capability not in capabilities:
                        capabilities.append(capability)
                    break
        if ModelCapability.INSTRUCTION_FOLLOWING not in capabilities:
            capabilities.append(ModelCapability.INSTRUCTION_FOLLOWING)
        if ModelCapability.ANALYSIS not in capabilities:
            capabilities.append(ModelCapability.ANALYSIS)
        return capabilities
    
    def _calculate_complexity_score(self, param_count: Optional[str], size_bytes: int, num_capabilities: int) -> float:
        score = 0.0
        size_gb = size_bytes / (1024 ** 3)
        if size_gb > 10: score += 0.5
        elif size_gb > 5: score += 0.4
        elif size_gb > 2: score += 0.3
        elif size_gb > 1: score += 0.2
        else: score += 0.1
        
        if param_count:
            match = re.search(r'(\d+)', param_count)
            if match:
                params = int(match.group(1))
                if params >= 70: score += 0.3
                elif params >= 13: score += 0.25
                elif params >= 7: score += 0.2
                elif params >= 3: score += 0.15
                else: score += 0.1
        
        score += min(num_capabilities * 0.04, 0.2)
        return min(score, 1.0)
    
    def get_model_for_complexity(self, complexity: float, available_ram_gb: Optional[float] = None) -> Optional[ModelInfo]:
        """Get best model for a given complexity level"""
        if not self._model_cache:
            self.discover_models_sync()
        if not self._model_cache:
            return None
        
        if available_ram_gb is None:
            memory = psutil.virtual_memory()
            available_ram_gb = memory.available / (1024 ** 3)
        
        viable_models = [m for m in self._model_cache.values() if m.estimated_ram_gb <= available_ram_gb]
        if not viable_models:
            all_models = sorted(self._model_cache.values(), key=lambda m: m.size_bytes)
            return all_models[0] if all_models else None
        
        viable_models.sort(key=lambda m: m.complexity_score, reverse=True)
        if complexity > 0.7:
            return viable_models[0]
        elif complexity > 0.3:
            return viable_models[len(viable_models) // 2]
        else:
            return viable_models[-1]
    
    def get_model_chain(self, available_ram_gb: Optional[float] = None) -> List[str]:
        """Get ordered fallback chain of models"""
        if not self._model_cache:
            self.discover_models_sync()
        if not self._model_cache:
            return []
        
        if available_ram_gb is None:
            memory = psutil.virtual_memory()
            available_ram_gb = memory.available / (1024 ** 3)
        
        viable = [m for m in self._model_cache.values() if m.estimated_ram_gb <= available_ram_gb * 1.2]
        viable.sort(key=lambda m: m.complexity_score, reverse=True)
        return [m.name for m in viable]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics"""
        if not self._model_cache:
            return {"models_discovered": 0, "cache_valid": False}
        total_size = sum(m.size_bytes for m in self._model_cache.values())
        avg_complexity = sum(m.complexity_score for m in self._model_cache.values()) / len(self._model_cache)
        return {
            "models_discovered": len(self._model_cache),
            "cache_valid": time.time() - self._cache_timestamp < self.cache_ttl,
            "total_size_gb": round(total_size / (1024 ** 3), 2),
            "average_complexity_score": round(avg_complexity, 3),
            "models": [m.to_dict() for m in self._model_cache.values()]
        }


# =============================================================================
# RAM-AWARE SELECTOR (merged from ram_aware_selector.py)
# =============================================================================

class RAMAwareSelector:
    """
    Selects LLM models based on real-time RAM availability.
    Features: Real-time monitoring, predictive selection, automatic downgrade.
    """
    
    def __init__(self, safety_margin_percent: float = 15.0, min_free_ram_gb: float = 2.0,
                 monitoring_interval: float = 1.0, history_size: int = 60):
        self.safety_margin_percent = safety_margin_percent
        self.min_free_ram_gb = min_free_ram_gb
        self.monitoring_interval = monitoring_interval
        self._memory_history: deque = deque(maxlen=history_size)
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._model_ram_estimates: Dict[str, float] = {}
        self._last_snapshot: Optional[MemorySnapshot] = None
        self._last_snapshot_time: float = 0
        self._snapshot_ttl: float = 0.5
        logger.info(f"RAMAwareSelector initialized (safety_margin: {safety_margin_percent}%)")
    
    def get_memory_snapshot(self, force_refresh: bool = False) -> MemorySnapshot:
        """Get current memory state with caching"""
        current_time = time.time()
        if not force_refresh and self._last_snapshot:
            if current_time - self._last_snapshot_time < self._snapshot_ttl:
                return self._last_snapshot
        
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        free_percent = 100 - mem.percent
        
        if free_percent > 60: pressure = MemoryPressureLevel.LOW
        elif free_percent > 30: pressure = MemoryPressureLevel.MODERATE
        elif free_percent > 15: pressure = MemoryPressureLevel.HIGH
        else: pressure = MemoryPressureLevel.CRITICAL
        
        snapshot = MemorySnapshot(
            timestamp=current_time, total_gb=total_gb, available_gb=available_gb,
            used_gb=mem.used / (1024 ** 3), percent_used=mem.percent,
            swap_used_gb=swap.used / (1024 ** 3), pressure_level=pressure
        )
        self._last_snapshot = snapshot
        self._last_snapshot_time = current_time
        self._memory_history.append(snapshot)
        return snapshot
    
    def get_available_ram_for_model(self) -> float:
        """Calculate RAM available for model loading"""
        snapshot = self.get_memory_snapshot()
        safety_gb = snapshot.total_gb * (self.safety_margin_percent / 100)
        reserve = max(safety_gb, self.min_free_ram_gb)
        return max(0.0, snapshot.available_gb - reserve)
    
    def can_load_model(self, model_name: str, estimated_ram_gb: float) -> Tuple[bool, str]:
        """Check if model can be safely loaded"""
        available = self.get_available_ram_for_model()
        if estimated_ram_gb <= available:
            return True, f"Sufficient RAM: {available:.1f}GB available, {estimated_ram_gb:.1f}GB needed"
        return False, f"Insufficient RAM: need {estimated_ram_gb:.1f}GB, only {available:.1f}GB available"
    
    def select_model(self, preferred_model: str, model_options: List[Tuple[str, float]], 
                     complexity: float = 0.5) -> ModelSelectionResult:
        """Select best model given RAM constraints"""
        snapshot = self.get_memory_snapshot()
        available = self.get_available_ram_for_model()
        
        preferred_ram = None
        for name, ram in model_options:
            if name == preferred_model:
                preferred_ram = ram
                break
        
        if preferred_ram is not None and preferred_ram <= available:
            return ModelSelectionResult(
                selected_model=preferred_model,
                reason=f"Preferred model fits: {preferred_ram:.1f}GB ≤ {available:.1f}GB",
                available_ram_gb=available, estimated_model_ram_gb=preferred_ram,
                safety_margin_gb=available - preferred_ram, pressure_level=snapshot.pressure_level,
                fallback_triggered=False
            )
        
        sorted_options = sorted(model_options, key=lambda x: x[1])
        for name, ram in sorted_options:
            if ram <= available:
                return ModelSelectionResult(
                    selected_model=name, reason=f"Fallback to {name}: {ram:.1f}GB fits",
                    available_ram_gb=available, estimated_model_ram_gb=ram,
                    safety_margin_gb=available - ram, pressure_level=snapshot.pressure_level,
                    fallback_triggered=True, original_model=preferred_model
                )
        
        if sorted_options:
            smallest = sorted_options[0]
            return ModelSelectionResult(
                selected_model=smallest[0], reason=f"CRITICAL: Using smallest: {smallest[0]}",
                available_ram_gb=available, estimated_model_ram_gb=smallest[1],
                safety_margin_gb=available - smallest[1], pressure_level=MemoryPressureLevel.CRITICAL,
                fallback_triggered=True, original_model=preferred_model
            )
        
        return ModelSelectionResult(
            selected_model="", reason="CRITICAL: No models available",
            available_ram_gb=available, estimated_model_ram_gb=0, safety_margin_gb=available,
            pressure_level=MemoryPressureLevel.CRITICAL, fallback_triggered=True, original_model=preferred_model
        )
    
    def get_memory_trend(self) -> Dict[str, Any]:
        """Analyze recent memory usage trend"""
        if len(self._memory_history) < 2:
            return {"trend": "unknown", "samples": len(self._memory_history)}
        
        history_list = list(self._memory_history)
        midpoint = len(history_list) // 2
        avg_first = sum(s.percent_used for s in history_list[:midpoint]) / midpoint
        avg_second = sum(s.percent_used for s in history_list[midpoint:]) / len(history_list[midpoint:])
        change = avg_second - avg_first
        
        if change > 5: trend, rec = "increasing", "Consider smaller models"
        elif change < -5: trend, rec = "decreasing", "More capacity available"
        else: trend, rec = "stable", "Memory usage is steady"
        
        return {"trend": trend, "change_percent": round(change, 2), "recommendation": rec,
                "samples": len(self._memory_history), "current_available_gb": round(self.get_available_ram_for_model(), 2)}
    
    def start_background_monitoring(self):
        """Start background memory monitoring"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True, name="RAMAwareSelector-Monitor")
        self._monitor_thread.start()
    
    def stop_background_monitoring(self):
        """Stop background monitoring"""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                snapshot = self.get_memory_snapshot(force_refresh=True)
                if snapshot.pressure_level == MemoryPressureLevel.CRITICAL:
                    logger.warning(f"CRITICAL memory pressure: {snapshot.available_gb:.1f}GB available")
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
            self._stop_monitoring.wait(self.monitoring_interval)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics"""
        snapshot = self.get_memory_snapshot()
        return {
            "current_state": snapshot.to_dict(),
            "available_for_model_gb": round(self.get_available_ram_for_model(), 2),
            "memory_trend": self.get_memory_trend(),
            "known_model_estimates": dict(self._model_ram_estimates),
            "history_samples": len(self._memory_history),
            "monitoring_active": self._monitor_thread is not None and self._monitor_thread.is_alive()
        }


# =============================================================================
# SINGLETONS (for easy access)
# =============================================================================

_model_discovery: Optional[DynamicModelDiscovery] = None
_ram_selector: Optional[RAMAwareSelector] = None


def get_model_discovery() -> DynamicModelDiscovery:
    """Get or create singleton model discovery instance"""
    global _model_discovery
    if _model_discovery is None:
        _model_discovery = DynamicModelDiscovery()
    return _model_discovery


def get_ram_selector() -> RAMAwareSelector:
    """Get or create singleton RAM selector"""
    global _ram_selector
    if _ram_selector is None:
        _ram_selector = RAMAwareSelector()
    return _ram_selector


if __name__ == "__main__":
    # Test the model selector
    logging.basicConfig(level=logging.INFO)
    
    print("🧠 Nexus LLM Analytics - Model Selector")
    print("=" * 50)
    
    # Show system info
    memory_info = ModelSelector.get_system_memory()
    print(f"💾 System Memory: {memory_info['total_gb']:.1f}GB total, {memory_info['available_gb']:.1f}GB available")
    
    # Select optimal models
    primary, review, embedding = ModelSelector.select_optimal_models()
    print(f"🤖 Selected Models:")
    print(f"   Primary: {primary}")
    print(f"   Review: {review}")  
    print(f"   Embedding: {embedding}")
    
    # Validate compatibility
    print(f"\n[>] Compatibility Check:")
    for model in [primary, review, embedding]:
        compatible, message = ModelSelector.validate_model_compatibility(model)
        print(f"   {message}")
    
    # Show recommendations
    recommendations = ModelSelector.recommend_system_config()
    if recommendations["recommendations"]:
        print(f"\n[i] Recommendations:")
        for rec in recommendations["recommendations"]:
            print(f"   [{rec['priority'].upper()}] {rec['message']}")