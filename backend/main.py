from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import analyze, upload, report

app = FastAPI(title="Nexus-LLM-Analytics API")

# CORS settings for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
app.include_router(upload.router, prefix="/upload-documents", tags=["upload"])
app.include_router(report.router, prefix="/generate-report", tags=["report"])

@app.get("/")
def root():
    return {"message": "Nexus-LLM-Analytics backend is running."}
