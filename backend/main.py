from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import analyze, upload, report, visualize
from core.utils import setup_logging
from dotenv import load_dotenv
import os

load_dotenv()
setup_logging()
app = FastAPI(title="Nexus-LLM-Analytics API")

# CORS settings for frontend-backend communication
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
app.include_router(upload.router, prefix="/upload-documents", tags=["upload"])
app.include_router(report.router, prefix="/generate-report", tags=["report"])
app.include_router(visualize.router, prefix="/visualize", tags=["visualize"])

@app.get("/")
def root():
    return {"message": "Nexus-LLM-Analytics backend is running."}
