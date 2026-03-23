"""Nexus LLM Analytics — FastAPI Application Entry-Point
======================================================

Configures the :class:`FastAPI` application instance with:

* CORS, rate-limiting, and global error-handling middleware.
* Lifespan hooks for model warm-up and periodic cleanup.
* API router mounting (analyse, upload, report, visualise,
  models, health, history, feedback, WebSocket).
* Prometheus ``/metrics`` endpoint for monitoring.

v2.0 Enterprise Additions
-------------------------
* Comprehensive module docstring.
* Inline documentation for every middleware and router mount.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import sys
import os
from pathlib import Path

# Add the src directory to Python path for proper imports
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

# Also add current backend directory for relative imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# FORCE UTF-8 ENCODING FOR WINDOWS CONSOLES (Fixes emoji crashes)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

try:
    from backend.api import analyze, upload, report, visualize, models, health, viz_enhance
    from backend.core.config import get_settings, validate_config
    from backend.core.rate_limiter import RateLimitMiddleware, global_rate_limiter
    from backend.core.error_handling import error_handler
    from backend.api.mode import router as mode_router
    
    
    # Auto-configure models on startup is now handled in lifespan (see below)
    # This prevents double-loading and ensures logging is configured first
        
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Available paths: {sys.path}")
    raise

# Load environment and settings
load_dotenv()
settings = get_settings()

# Setup logging
settings.setup_logging()

# Validate configuration
validate_config()

# Lifespan event handler (replaces deprecated @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown tasks.

    Startup:
        1. Initialise :class:`ModelManager` singleton.
        2. Async warm-up of the primary LLM into VRAM/RAM.
        3. Run non-blocking startup optimisation pass.
        4. Schedule hourly cleanup of stale analysis records.

    Shutdown:
        Cancel the periodic cleanup task and call
        ``ModelManager.shutdown()`` for graceful resource release.
    """
    import logging
    import asyncio
    
    # Startup tasks
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Nexus LLM Analytics backend...")
    
    # STEP 1: Initialize ModelManager singleton — lazy-loads models & infra
    from backend.agents import get_model_manager
    manager = get_model_manager()
    
    try:
        manager.ensure_initialized()
        logger.info(
            "✅ Models ready: Primary=%s, Review=%s",
            manager.cached_models.get("primary", "?"),
            manager.cached_models.get("review", "?"),
        )
    except Exception as init_error:
        logger.warning("⚠️ ModelManager init warning: %s", init_error)

    # STEP 1b: Initialize ModeManager singleton (always starts OFFLINE)
    try:
        from backend.core.mode_manager import get_mode_manager as _get_mm
        _mode_mgr = _get_mm()
        logger.info("✅ ModeManager ready — mode=%s", _mode_mgr.get_mode())
    except Exception as mm_err:
        logger.warning("⚠️ ModeManager init warning: %s", mm_err)
    
    # STEP 2: Async warmup — forces Ollama to load model into VRAM/RAM
    try:
        logger.info("🔥 Warming up primary model (this may take 10-30 seconds)...")
        warmup_result = await manager.async_warmup()
        if warmup_result["success"]:
            logger.info(
                "✅ Model warmed up in %.1fs — ready!",
                warmup_result["latency_seconds"],
            )
        else:
            logger.warning("⚠️ Warmup issue: %s", warmup_result.get("error"))
    except Exception as warmup_error:
        logger.warning("⚠️ Model warmup skipped: %s", warmup_error)
    
    # STEP 3: Background optimization (non-blocking)
    try:
        from backend.core.optimizers import optimize_startup
        optimization_result = optimize_startup()
    except Exception as opt_error:
        logger.debug("Optimization note: %s", opt_error)
    
    logger.info("✅ Backend ready to serve requests!")
    
    # STEP 4: Schedule periodic cleanup of old analysis records
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(3600)  # Every hour
            try:
                from backend.core.analysis_manager import analysis_manager
                analysis_manager.cleanup_old_analyses(max_age_hours=24)
            except Exception:
                pass
    
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield  # Application serves requests here — model is already warm
    
    # Shutdown tasks — clean up all managed resources
    cleanup_task.cancel()
    
    # [OPTIMIZATION 9.4] Resource Registry Cleanup
    try:
        from backend.core.resource_registry import get_resource_registry
        await get_resource_registry().cleanup_all()
    except Exception as rr_error:
        logger.warning("Resource registry cleanup error: %s", rr_error)

    try:
        await manager.shutdown()
    except Exception as shutdown_error:
        logger.warning("Shutdown error (ignored): %s", shutdown_error)
    logger.info("👋 Nexus LLM Analytics backend shut down cleanly")

# Create FastAPI app with settings and lifespan handler
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    openapi_url=settings.openapi_url,
    lifespan=lifespan
)

# Add rate limiting middleware if enabled
if settings.enable_rate_limiting:
    rate_limiter = RateLimitMiddleware(global_rate_limiter)
    
    @app.middleware("http")
    async def rate_limit_middleware(request, call_next):
        return await rate_limiter(request, call_next)

# CORS settings from config
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# GZip response compression — ~70% reduction for JSON responses
from starlette.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=500)

# Add global error handling
from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from backend.core.error_handling import NexusError

# Custom exception handler for validation errors (Pydantic)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with normalized format"""
    # Extract details from Pydantic error
    details = exc.errors()
    # Create simple error message
    error_msg = f"Validation error: {details[0].get('msg')}" if details else "Validation error"
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": error_msg,
            "status": "error",
            "details": details,
            "code": "validation_error"
        }
    )

# Custom exception handler for FastAPI/Starlette HTTP exceptions (404, 403, etc)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle standard HTTP exceptions with normalized format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": str(exc.detail),
            "status": "error", 
            "code": f"http_{exc.status_code}"
        }
    )


@app.exception_handler(NexusError)
async def nexus_error_handler(request: Request, exc: NexusError):
    """Handle custom Nexus errors"""
    error_response = exc.to_dict()
    error_handler.handle_error(exc, context={"path": str(request.url)}, raise_error=False)
    return JSONResponse(
        status_code=400,
        content=error_response
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    error_response = error_handler.handle_error(
        exc, 
        context={"path": str(request.url)}, 
        raise_error=False
    )
    return JSONResponse(
        status_code=500,
        content=error_response
    )

# Mount API routers
app.include_router(analyze.router, prefix="/api/analyze", tags=["analyze"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(report.router, prefix="/api/report", tags=["report"])
app.include_router(visualize.router, prefix="/api/visualize", tags=["visualize"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(health.router, prefix="/api/health", tags=["health"])

# Mode toggle router (GET/POST /api/mode)
app.include_router(mode_router, prefix="/api")

# Import and mount history router
from backend.api import history
app.include_router(history.router, prefix="/api/history", tags=["history"])

# Mount enhanced visualization router (LIDA-inspired features)
app.include_router(viz_enhance.router, prefix="/api/viz", tags=["visualization-enhancement"])

# Mount feedback router
from backend.api import feedback
app.include_router(feedback.router, tags=["feedback"])  # Router already has prefix="/api/feedback"

# Mount Swarm API router (v2.1)
from backend.api import swarm
app.include_router(swarm.router, prefix="/api/swarm", tags=["swarm"])

# WebSocket support for real-time updates across devices
from fastapi import WebSocket
try:
    from backend.core.websocket_manager import websocket_endpoint, connection_manager
    
    @app.websocket("/ws/{client_id}")
    async def websocket_route(websocket: WebSocket, client_id: str):
        """WebSocket endpoint for real-time updates"""
        await websocket_endpoint(websocket, client_id)
    
    @app.get("/api/ws/status", tags=["websocket"])
    def websocket_status():
        """Get WebSocket connection status"""
        return {
            "active_connections": connection_manager.get_connection_count(),
            "connected_clients": connection_manager.get_connected_clients()
        }
except ImportError as e:
    import logging
    logging.warning(f"WebSocket manager not available: {e}")

@app.get("/")
def root():
    return {"message": "Nexus-LLM-Analytics backend is running."}

@app.get("/health")
def health_check():
    """Lightweight health check without initializing models"""
    return {
        "status": "healthy",
        "message": "Backend server is running",
        "models_loaded": False,
        "note": "Models will be loaded on first analysis request"
    }

# Phase 3.8: Prometheus metrics endpoint
@app.get("/metrics", tags=["monitoring"])
async def metrics_endpoint():
    """
    Prometheus metrics endpoint for monitoring.
    
    Returns metrics in Prometheus text format for scraping by Prometheus server
    or compatible monitoring systems.
    """
    from fastapi.responses import Response
    from backend.core.metrics import generate_metrics_output, get_metrics_content_type
    
    return Response(
        content=generate_metrics_output(),
        media_type=get_metrics_content_type()
    )

@app.get("/metrics/json", tags=["monitoring"])
async def metrics_json_endpoint():
    """
    Get metrics in JSON format for easier debugging and UI integration.
    """
    from backend.core.metrics import METRICS
    from backend.core.advanced_cache import get_cache_status
    
    return {
        "prometheus_available": METRICS.__class__.__name__ != "DummyMetric",
        "cache_status": get_cache_status(),
        "fallback_stats": METRICS.get_fallback_stats() if hasattr(METRICS, 'get_fallback_stats') else {}
    }

@app.get("/download-report/")
async def download_report_direct(filename: str = None):
    """Direct download endpoint for reports"""
    from fastapi import Query
    from backend.api.report import download_report
    return download_report(filename=filename)


if __name__ == "__main__":
    import uvicorn
    import argparse
    import sys
    
    # Configure stdout for Windows/Emoji support
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    
    parser = argparse.ArgumentParser(description="Nexus LLM Analytics Backend Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Nexus Backend on {args.host}:{args.port} (Workers: {args.workers}, Reload: {args.reload})")
    
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info"
    )
