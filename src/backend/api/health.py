from fastapi import APIRouter
from typing import Dict, Any
import logging
import time
import psutil
import os
import socket

# Health monitoring and system status endpoint

router = APIRouter()

def _get_local_ip() -> str:
    """Get the local IP address for network access"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system health status"""
    try:
        from backend.core.circuit_breaker import get_all_circuit_breaker_status
        from backend.core.advanced_cache import get_cache_status
        from backend.core.model_selector import ModelSelector
        
        # System resources
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_usage = psutil.disk_usage('/')
        
        # Model status
        try:
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
            import requests
            response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
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
        circuit_status = get_all_circuit_breaker_status()
        
        # Cache performance
        cache_status = get_cache_status()
        
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
        logging.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Simple health check endpoint"""
    return {"status": "ok", "message": "Nexus LLM Analytics is running"}

@router.get("/network-info")
async def get_network_info() -> Dict[str, Any]:
    """Get network information for sharing with other devices"""
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
    """Get detailed cache information"""
    try:
        from backend.core.advanced_cache import get_cache_status
        return get_cache_status()
    except Exception as e:
        return {"error": str(e)}

@router.post("/clear-cache")
async def clear_cache() -> Dict[str, str]:
    """Clear all system caches"""
    try:
        from backend.core.advanced_cache import clear_all_caches
        clear_all_caches()
        return {"status": "success", "message": "All caches cleared"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def _generate_recommendations(services: Dict, model_status: Dict, memory_info) -> list:
    """Generate system optimization recommendations"""
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