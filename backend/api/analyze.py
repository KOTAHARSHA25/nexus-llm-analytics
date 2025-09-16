from fastapi import APIRouter
import logging
from pydantic import BaseModel
from typing import Optional
from agents.crew_manager import CrewManager

# Handles /analyze endpoint logic for data analysis requests using CrewAI

router = APIRouter()
crew_manager = CrewManager()


class AnalyzeRequest(BaseModel):
    query: str
    filename: str
    column: Optional[str] = None
    value: Optional[str] = None


@router.post("/")
async def analyze_query(request: AnalyzeRequest):
    logging.info(f"[ANALYZE] Received query: {request.query}, filename: {request.filename}, column: {request.column}, value: {request.value}")
    result = crew_manager.handle_query(
        request.query,
        request.filename,
        column=request.column,
        value=request.value,
    )
    logging.info(f"[ANALYZE] Result: {result}")
    return result
