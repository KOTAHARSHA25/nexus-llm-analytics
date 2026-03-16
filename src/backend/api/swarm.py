"""Swarm API Router — Nexus LLM Analytics
======================================

Exposes internal Swarm state to the frontend (Next.js) for real-time visualization.

Endpoints
---------
GET /api/swarm/agents
    List all active agents and their capabilities.
GET /api/swarm/tasks
    List the current task graph (pending, running, completed).
GET /api/swarm/insights
    Query the persistent vector memory for insights.
GET /api/swarm/events
    Get the recent event log (messages broadcast on the Swarm bus).

Dependencies
------------
- backend.services.analysis_service.get_analysis_service
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.services.analysis_service import get_analysis_service
from backend.core.swarm import SwarmEvent

router = APIRouter()

# --- Helpers ---

def get_swarm_context():
    """Retrieve the active SwarmContext from the orchestrator."""
    service = get_analysis_service()
    if not service.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return service.orchestrator.swarm_context

def get_registry():
    """Retrieve the active AgentRegistry."""
    service = get_analysis_service()
    return service.registry

# --- Models ---

class AgentModel(BaseModel):
    name: str
    version: str
    description: str
    capabilities: List[str]
    priority: int

class TaskModel(BaseModel):
    id: str
    description: str
    status: str
    dependencies: List[str]
    assigned_to: Optional[str]
    created_at: float
    result: Optional[Any] = None

class InsightModel(BaseModel):
    content: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None

class EventModel(BaseModel):
    id: str
    type: str
    source: str
    content: Any
    timestamp: float

# --- Endpoints ---

@router.get("/agents", response_model=List[AgentModel])
async def get_agents():
    """List all active agents in the swarm."""
    registry = get_registry()
    agents = []
    for name, agent in registry.agents.items():
        meta = agent.metadata
        agents.append(AgentModel(
            name=meta.name,
            version=meta.version,
            description=meta.description,
            capabilities=[c.name for c in meta.capabilities],
            priority=meta.priority
        ))
    return agents

@router.get("/tasks", response_model=List[TaskModel])
async def get_tasks():
    """Get the current task graph status."""
    swarm = get_swarm_context()
    tasks = []
    # Access internal task graph (safe for read)
    for tid, info in swarm._task_graph.items():
        tasks.append(TaskModel(
            id=tid,
            description=info["description"],
            status=info["status"],
            dependencies=info["dependencies"],
            assigned_to=info["assigned_to"],
            created_at=info["created_at"],
            result=info.get("result")
        ))
    return tasks

@router.get("/insights", response_model=List[InsightModel])
async def search_insights(
    query: str = Query(..., min_length=1, description="Search query for vector memory")
):
    """Search persistent vector memory for insights."""
    swarm = get_swarm_context()
    results = swarm.query_insights(query, n_results=5)
    return [InsightModel(**r) for r in results]

class JobModel(BaseModel):
    id: str
    user_session: Optional[str]
    status: str
    stage: str
    start_time: float
    end_time: Optional[float] = None
    error: Optional[str] = None

@router.get("/jobs", response_model=List[JobModel])
async def get_jobs(active_only: bool = True):
    """Get status of analysis jobs (High-level lifecycle)."""
    from backend.core.analysis_manager import get_analysis_manager
    manager = get_analysis_manager()
    
    if active_only:
        searched = manager.get_running_analyses()
    else:
        # In a real app we might query a DB, but AnalysisManager currently only keeps memory of recent
        # Here we just return running for consistency unless we expanded AnalysisManager to keep history
        searched = manager.get_running_analyses() 
        # Note: AnalysisManager currently only exposes get_running_analyses publicly in its interface
        # We can expand if needed, but for now this is sufficient for "Active Jobs"
    
    jobs = []
    for jid, info in searched.items():
        jobs.append(JobModel(
            id=info['id'],
            user_session=info.get('user_session'),
            status=info['status'],
            stage=info.get('stage', 'unknown'),
            start_time=info['start_time'],
            end_time=info.get('end_time'),
            error=info.get('error')
        ))
    return jobs

@router.get("/events", response_model=List[EventModel])
async def get_events(limit: int = 20):
    """Get recent swarm events."""
    swarm = get_swarm_context()
    events = []
    # Reverse to show newest first
    for msg in reversed(swarm._message_history[-limit:]):
        events.append(EventModel(
            id=msg.id,
            type=msg.type.name,
            source=msg.source_agent,
            content=msg.content,
            timestamp=msg.timestamp
        ))
    return events
