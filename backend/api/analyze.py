from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from backend.agents.controller_agent import ControllerAgent

# Handles /analyze endpoint logic for data analysis requests

router = APIRouter()
controller = ControllerAgent()


class AnalyzeRequest(BaseModel):
    query: str
    filename: str
    column: Optional[str] = None
    value: Optional[str] = None


@router.post("/")
async def analyze_query(request: AnalyzeRequest):
    result = controller.handle_query(
        request.query,
        request.filename,
        column=request.column,
        value=request.value,
    )
    return result
