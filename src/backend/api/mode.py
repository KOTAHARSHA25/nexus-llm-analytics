"""Mode API Router — Online/Offline toggle endpoint
==================================================

Provides two endpoints for reading and changing the system operating mode:

    GET  /api/mode  → returns current mode
    POST /api/mode  → switches mode

The router is registered in ``main.py`` via::

    app.include_router(mode_router, prefix="/api")
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from backend.core.mode_manager import get_mode_manager
from backend.agents.model_manager import get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mode"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ModeResponse(BaseModel):
    mode: str
    status: str
    ollama_status: str = "unknown"  # "running", "stopped", "unknown"


class ModeSetRequest(BaseModel):
    mode: str

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized not in ("online", "offline"):
            raise ValueError("mode must be 'online' or 'offline'")
        return normalized


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/mode", response_model=ModeResponse, summary="Get current operating mode")
async def get_mode() -> ModeResponse:
    """Return the current system operating mode.

    Returns:
        ``{"mode": "online"|"offline", "status": "ok"}``
    """
    manager = get_mode_manager()
    ollama_ok = "running" if manager.is_ollama_running() else "stopped"
    return ModeResponse(mode=manager.get_mode(), status="ok", ollama_status=ollama_ok)


@router.post("/mode", response_model=ModeResponse, summary="Switch operating mode")
async def set_mode(request: ModeSetRequest) -> ModeResponse:
    """Switch between online and offline operating modes.

    Body::

        {"mode": "online"}   # or "offline"

    Returns:
        ``{"mode": "...", "status": "switched"}``

    Raises:
        422 Unprocessable Entity: If *mode* is not ``"online"`` or ``"offline"``.
    """
    manager = get_mode_manager()
    manager.set_mode(request.mode)
    # Reset lazy cloud clients so new API keys are probed on next use
    manager.reset_clients()
    # Re-initialize cached LLMs so they switch between Ollama (offline)
    # and cloud shim (online) immediately — not on the next unrelated request
    try:
        get_model_manager().reload_models()
    except Exception as exc:
        logger.warning("[mode] LLM reload after mode switch failed: %s", exc)
    # Ollama start/stop is already triggered in set_mode() via background thread;
    # report the current known state (may still be starting/stopping)
    ollama_ok = "running" if manager.is_ollama_running() else "stopped"
    return ModeResponse(mode=manager.get_mode(), status="switched", ollama_status=ollama_ok)
