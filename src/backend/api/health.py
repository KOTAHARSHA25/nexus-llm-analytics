"""Health & Diagnostics API — System Status Monitoring Endpoints
================================================================

Exposes health-check, network-info, cache, and comprehensive system-status
endpoints consumed by frontend dashboards, load balancers, and admin tooling.

Endpoints
---------
``GET /``
    Minimal liveness probe — returns ``{"status": "ok"}``.
``GET /status``
    Full system diagnostic: memory, CPU, disk, Ollama, ChromaDB, circuit
    breakers, cache performance, model status, and recommendations.
``GET /network-info``
    Local IP and sharing instructions for multi-device access.
``GET /cache-info``
    Detailed cache tier statistics.
``POST /clear-cache``
    Flush all cache tiers.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from typing import Any, Dict, List

import psutil
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()

def _get_local_ip() -> str:
    """Resolve the host’s LAN IP via a non-blocking UDP socket probe.

    Returns:
        The LAN-facing IPv4 address, or ``"127.0.0.1"`` on failure.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """Return comprehensive system health diagnostics.

    Aggregates memory, CPU, disk, Ollama reachability, ChromaDB status,
    circuit-breaker state, cache performance, active model config, and
    actionable recommendations into a single response.

    Returns:
        Dict with top-level ``status`` (``healthy`` | ``unhealthy`` | ``error``),
        plus nested sections for each subsystem.
    """
    try:
        try:
            from backend.infra.circuit_breaker import get_all_circuit_breaker_status
        except ImportError:
            get_all_circuit_breaker_status = None
        try:
            from backend.core.advanced_cache import get_cache_status
        except ImportError:
            get_cache_status = None
        try:
            from backend.core.engine.model_selector import ModelSelector
        except ImportError:
            ModelSelector = None
        
        # System resources
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0)  # Non-blocking: uses delta since last call
        disk_usage = psutil.disk_usage('/')
        
        # Model status
        try:
            if ModelSelector is None:
                raise ImportError("ModelSelector not available")
            selected_models = ModelSelector.select_optimal_models()
            model_status = {
                "primary": selected_models[0],
                "review": selected_models[1], 
                "embedding": selected_models[2],
                "selection_successful": True
            }
        except Exception as e:
            model_status = {
                "selection_successful": False,
                "error": str(e)
            }
        
        # Service health checks
        services = {}
        
        # Check Ollama connection
        try:
            from backend.core.config import settings
            import requests, asyncio
            response = await asyncio.to_thread(
                lambda: requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
            )
            services["ollama"] = {
                "status": "healthy" if response.status_code == 200 else "degraded",
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "models_available": len(response.json().get("models", [])) if response.status_code == 200 else 0
            }
        except Exception as e:
            services["ollama"] = {
                "status": "unhealthy",
                "error": str(e),
                "models_available": 0
            }
        
        # Check ChromaDB
        try:
            from backend.core.chromadb_client import ChromaDBClient
            chroma_client = ChromaDBClient()
            collections = chroma_client.list_collections()
            services["chromadb"] = {
                "status": "healthy",
                "collections_count": len(collections)
            }
        except Exception as e:
            services["chromadb"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Circuit breaker status
        # [FIX] Check if function exists before calling
        if get_all_circuit_breaker_status:
            circuit_status = get_all_circuit_breaker_status()
        else:
            circuit_status = {"status": "unavailable", "reason": "module_not_found"}
        
        # Cache performance
        # [FIX] Check if function exists before calling
        if get_cache_status:
            cache_status = get_cache_status()
        else:
            cache_status = {"status": "unavailable", "reason": "module_not_found"}
        
        # Overall health determination
        critical_services_healthy = services["ollama"]["status"] in ["healthy", "degraded"]
        overall_health = "healthy" if critical_services_healthy else "unhealthy"
        
        return {
            "status": overall_health,
            "timestamp": time.time(),
            "network": {
                "local_ip": _get_local_ip(),
                "backend_url": f"http://{_get_local_ip()}:8000",
                "share_info": "Other devices on the same WiFi can use this URL"
            },
            "system_resources": {
                "memory": {
                    "total_gb": round(memory_info.total / (1024**3), 2),
                    "available_gb": round(memory_info.available / (1024**3), 2),
                    "used_percent": memory_info.percent
                },
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "disk": {
                    "total_gb": round(disk_usage.total / (1024**3), 2),
                    "free_gb": round(disk_usage.free / (1024**3), 2),
                    "used_percent": round((disk_usage.used / disk_usage.total) * 100, 1)
                }
            },
            "services": services,
            "models": model_status,
            "circuit_breakers": circuit_status,
            "cache_performance": cache_status,
            "recommendations": _generate_recommendations(services, model_status, memory_info)
        }
        
    except Exception as e:
        logger.error("Health check failed: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")

@router.get("/")
async def health_check() -> Dict[str, str]:
    """Minimal liveness probe for orchestrators and load balancers."""
    return {"status": "ok", "message": "Nexus LLM Analytics is running"}

@router.get("/network-info")
async def get_network_info() -> Dict[str, Any]:
    """Return LAN connection details for multi-device sharing."""
    local_ip = _get_local_ip()
    return {
        "local_ip": local_ip,
        "backend_url": f"http://{local_ip}:8000",
        "websocket_url": f"ws://{local_ip}:8000/ws",
        "share_instructions": {
            "step1": "Ensure both devices are on the same WiFi network",
            "step2": f"On the other device, open Nexus LLM Analytics",
            "step3": f"In Backend Connection settings, enter: http://{local_ip}:8000",
            "step4": "Click 'Test' to verify connection, then 'Save'"
        },
        "firewall_note": "If connection fails, ensure port 8000 is allowed in your firewall"
    }

@router.get("/cache-info")
async def get_cache_info() -> Dict[str, Any]:
    """Return detailed cache-tier statistics."""
    try:
        try:
            from backend.core.advanced_cache import get_cache_status
        except ImportError:
            return {"error": "Cache module not available"}
        return get_cache_status()
    except Exception as e:
        return {"error": str(e)}

@router.post("/clear-cache")
async def clear_cache() -> Dict[str, str]:
    """Flush all cache tiers (L1/L2/L3) and return confirmation."""
    try:
        try:
            from backend.core.advanced_cache import clear_all_caches
        except ImportError:
            return {"status": "error", "error": "Cache module not available"}
        clear_all_caches()
        return {"status": "success", "message": "All caches cleared"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def _generate_recommendations(
    services: Dict[str, Any],
    model_status: Dict[str, Any],
    memory_info: Any,
) -> List[Dict[str, str]]:
    """Generate actionable optimization recommendations based on current system state.

    Args:
        services:     Health-check results keyed by service name.
        model_status: Model selection outcome dict.
        memory_info:  ``psutil`` virtual-memory named tuple.

    Returns:
        List of recommendation dicts with ``type``, ``title``, ``description``,
        and ``action`` keys.
    """
    recommendations = []
    
    # Ollama service recommendations
    if services["ollama"]["status"] == "unhealthy":
        recommendations.append({
            "type": "critical",
            "title": "Ollama Service Down",
            "description": "The AI model service is not running",
            "action": "Start Ollama: 'ollama serve' or check installation"
        })
    elif services["ollama"]["status"] == "degraded":
        recommendations.append({
            "type": "warning", 
            "title": "Ollama Performance Issues",
            "description": "AI service is responding slowly",
            "action": "Check system resources or restart Ollama"
        })
    
    # Memory recommendations
    available_gb = memory_info.available / (1024**3)
    if available_gb < 4:
        recommendations.append({
            "type": "warning",
            "title": "Low Memory",
            "description": f"Only {available_gb:.1f}GB RAM available",
            "action": "Close unused applications or switch to smaller AI models"
        })
    
    # Model recommendations
    if not model_status.get("selection_successful", False):
        recommendations.append({
            "type": "error",
            "title": "Model Selection Failed",
            "description": "Unable to select appropriate AI models",
            "action": "Check model configuration and system requirements"
        })
    
    # Performance recommendations
    if services.get("ollama", {}).get("models_available", 0) == 0:
        recommendations.append({
            "type": "warning",
            "title": "No AI Models Available",
            "description": "No models are installed in Ollama",
            "action": "Install models based on your system: 'ollama pull llama3.1:8b' (8GB+ RAM) or 'ollama pull phi3:mini' (4GB+ RAM) or 'ollama pull tinyllama' (low memory)"
        })
    
    return recommendations