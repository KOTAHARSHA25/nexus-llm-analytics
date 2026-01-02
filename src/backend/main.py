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

try:
    from backend.api import analyze, upload, report, visualize, models, health, viz_enhance
    from backend.core.config import get_settings, validate_config
    from backend.core.rate_limiter import RateLimitMiddleware, global_rate_limiter
    from backend.core.error_handling import error_handler
    
    # Auto-configure models on startup (quiet mode - details go to log file)
    try:
        from backend.core.model_selector import ModelSelector
        primary, review, embedding = ModelSelector.select_optimal_models()
        # Print minimal startup info
        print(f"[AI] Models: {primary.replace('ollama/', '')}, {review.replace('ollama/', '')}")
    except Exception as model_error:
        print(f"[!] Model selection warning: {model_error}")
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Available paths:", sys.path)
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
    """Lifespan event handler for startup and shutdown tasks"""
    import logging
    import asyncio
    
    # Startup tasks
    logger = logging.getLogger(__name__)
    logger.debug("Starting Nexus LLM Analytics backend...")
    
    # Auto-configure models on startup to prevent hardcoded fallbacks
    try:
        from backend.core.model_selector import ModelSelector
        # Trigger model selection to cache results and detect issues early
        primary, review, embedding = ModelSelector.select_optimal_models()
        logger.debug(f"Startup model selection: Primary={primary}, Review={review}")
    except Exception as model_error:
        logger.warning(f"Model selection warning during startup: {model_error}")
    
    # Start background optimization (non-blocking)
    try:
        from backend.core.optimizers import optimize_startup
        optimization_result = optimize_startup()
        logger.info("Backend ready")
    except Exception as opt_error:
        logger.warning(f"Optimization warning: {opt_error}")
    
    # Run model test in background to avoid blocking startup (optional validation)
    asyncio.create_task(test_model_on_startup())
    
    yield  # Application runs here
    
    # Shutdown tasks
    logger.debug("Shutting down Nexus LLM Analytics backend...")

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

# Add global error handling
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from backend.core.error_handling import NexusError

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

# Import and mount history router
from backend.api import history
app.include_router(history.router, prefix="/api/history", tags=["history"])

# Mount enhanced visualization router (LIDA-inspired features)
app.include_router(viz_enhance.router, prefix="/api/viz", tags=["visualization-enhancement"])

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

async def test_model_on_startup():
    """
    Test the currently configured model on startup (background task).
    Integrated from src2 for better startup validation.
    """
    import logging
    import asyncio
    
    logger = logging.getLogger(__name__)
    
    # Wait for the server to fully start
    await asyncio.sleep(5)
    
    try:
        # Check if Ollama is available first
        from backend.core.llm_client import LLMClient
        try:
            llm_client = LLMClient()
            available_models = llm_client.get_available_models()
            if not available_models:
                logger.debug("Skipping startup model test - Ollama not running or no models installed")
                return
        except Exception:
            logger.debug("Skipping startup model test - Ollama not available")
            return
        
        # Test is disabled to reduce startup logs
        return
                
    except Exception as e:
        # Gracefully handle any errors - don't block startup
        logger.debug(f"Startup model test skipped: {str(e)[:100]}")
        # Don't log as error since it's optional validation

# WebSocket endpoint for real-time updates
# NOTE (v1.1): WebSocket functionality archived - file moved to archive/removed_v1.1/
# Uncomment and restore websocket_manager.py if real-time updates are needed in future
# if settings.enable_websockets:
#     from fastapi import WebSocket
#     from backend.core.websocket_manager import websocket_endpoint
#     
#     @app.websocket("/ws/{client_id}")
#     async def websocket_route(websocket: WebSocket, client_id: str):
#         await websocket_endpoint(websocket, client_id)
