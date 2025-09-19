from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    from api import analyze, upload, report, visualize, models
    from core.config import get_settings, validate_config
    from core.rate_limiter import RateLimitMiddleware, global_rate_limiter
    from core.error_handling import error_handler
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

# Create FastAPI app with settings
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    openapi_url=settings.openapi_url
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

# Mount API routers
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
app.include_router(upload.router, prefix="/upload-documents", tags=["upload"])
app.include_router(report.router, prefix="/generate-report", tags=["report"])
app.include_router(visualize.router, prefix="/visualize", tags=["visualize"])
app.include_router(models.router, prefix="/models", tags=["models"])

@app.get("/")
def root():
    return {"message": "Nexus-LLM-Analytics backend is running."}

@app.on_event("startup")
async def startup_event():
    """Run startup tasks including model testing"""
    import logging
    import asyncio
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Nexus LLM Analytics backend...")
    
    # Run model test in background to avoid blocking startup
    asyncio.create_task(test_model_on_startup())
    
    logger.info("✅ Backend startup completed")

async def test_model_on_startup():
    """Test the currently configured model on startup"""
    import logging
    import asyncio
    
    logger = logging.getLogger(__name__)
    
    # Wait a bit for the server to fully start
    await asyncio.sleep(5)
    
    try:
        logger.info("🧪 Running automatic model test on startup...")
        
        # Import here to avoid circular imports
        from core.user_preferences import UserPreferencesManager
        
        # Get current model preferences
        prefs_manager = UserPreferencesManager()
        preferences = prefs_manager.load_preferences()
        
        current_model = preferences.primary_model if preferences else "ollama/tinyllama"
        
        logger.info(f"🤖 Testing model: {current_model}")
        
        # Create a simple test
        from agents.crew_manager import CrewManager
        
        crew_manager = CrewManager()
        
        # Run a simple test query
        test_result = crew_manager.analyze_unstructured_data(
            query="Hello, please respond with 'Model test successful' and confirm you are working properly.",
            filename="startup_test.txt"
        )
        
        if test_result.get("success"):
            logger.info("✅ Model test completed successfully!")
            logger.info(f"📊 Test result: {test_result.get('result', 'No result')[:100]}...")
            
            # Save test results
            prefs_manager.save_test_result({
                "model": current_model,
                "success": True,
                "message": "Startup test successful",
                "timestamp": str(asyncio.get_event_loop().time())
            })
        else:
            logger.warning("⚠️ Model test failed during startup")
            logger.warning(f"❌ Error: {test_result.get('error', 'Unknown error')}")
            
            # Save failed test results
            prefs_manager.save_test_result({
                "model": current_model,
                "success": False,
                "message": test_result.get('error', 'Startup test failed'),
                "timestamp": str(asyncio.get_event_loop().time())
            })
            
    except Exception as e:
        logger.error(f"💥 Error during startup model test: {e}")
        
        # Save error test results
        try:
            from core.user_preferences import UserPreferencesManager
            prefs_manager = UserPreferencesManager()
            prefs_manager.save_test_result({
                "model": "unknown",
                "success": False,
                "message": f"Startup test error: {str(e)}",
                "timestamp": str(asyncio.get_event_loop().time())
            })
        except Exception:
            pass  # Don't fail if we can't save the error

# WebSocket endpoint for real-time updates
if settings.enable_websockets:
    from fastapi import WebSocket
    from backend.core.websocket_manager import websocket_endpoint
    
    @app.websocket("/ws/{client_id}")
    async def websocket_route(websocket: WebSocket, client_id: str):
        await websocket_endpoint(websocket, client_id)
