"""
Swarm Coordination Infrastructure for Nexus LLM Analytics.

This module provides the `SwarmContext` class, which acts as the shared "Blackboard"
and message bus for multiple agents to coordinate, share insights, and track
task dependencies.

Key Features:
- Shared Memory (Blackboard): Agents can read/write shared state.
- Message Bus: Publish/Subscribe mechanism for inter-agent events.
- Task Graph: Track dependencies between agent tasks.

Author: Nexus Team
Since: v2.1.0
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import time

class SwarmEvent(Enum):
    """Types of events that can be broadcast on the swarm bus"""
    INSIGHT_FOUND = "insight_found"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    RESOURCE_UPDATED = "resource_updated"
    CRITIQUE_ISSUED = "critique_issued"
    PLAN_UPDATED = "plan_updated"

@dataclass
class SwarmMessage:
    """A message broadcast on the swarm bus"""
    type: SwarmEvent
    source_agent: str
    content: Any
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

class SwarmContext:
    """
    Shared context (Blackboard) for agent swarm coordination.
    
    Acts as the central nervous system for a group of agents working together.
    """
    
    def __init__(self):
        # Blackboard: Shared state accessible to all agents
        self._shared_memory: Dict[str, Any] = {}
        
        # Message Bus: Subscribers for events
        self._subscribers: Dict[SwarmEvent, List[Callable[[SwarmMessage], None]]] = {}
        
        # Task Graph: Track active tasks and their status
        # Format: {task_id: {"status": "pending|running|done|failed", "dependencies": [], "assigned_to": agent_name}}
        self._task_graph: Dict[str, Dict[str, Any]] = {}
        
        # History: Log of all messages for debugging/audit
        self._message_history: List[SwarmMessage] = []
        
        logging.debug("SwarmContext initialized")

    def publish(self, event_type: SwarmEvent, source_agent: str, content: Any) -> None:
        """Broadcast a message to all subscribers"""
        message = SwarmMessage(
            type=event_type,
            source_agent=source_agent,
            content=content
        )
        
        self._message_history.append(message)
        logging.debug(f"Swarm Event [{event_type.value}] from {source_agent}: {str(content)[:100]}...")
        
        # PERSISTENCE: Automatically store insights in vector memory
        if event_type == SwarmEvent.INSIGHT_FOUND:
            if hasattr(self, 'memory_collection') and self.memory_collection:
                try:
                    # Extract text content for embedding
                    text_content = str(content.get('summary', str(content)))
                    self.store_insight_vector(text_content, {"agent": source_agent, "type": "insight"}, message.id)
                except Exception as e:
                    logging.warning(f"Failed to auto-persist insight: {e}")
        
        # Notify subscribers
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(message)
                except Exception as e:
                    logging.error(f"Error in swarm subscriber callback: {e}")

    def subscribe(self, event_type: SwarmEvent, callback: Callable[[SwarmMessage], None]) -> None:
        """Subscribe to a specific event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def write_shared(self, key: str, value: Any, agent_name: str) -> None:
        """Write to shared memory (Blackboard)"""
        self._shared_memory[key] = value
        # Notify implicitly that memory was updated
        self.publish(SwarmEvent.RESOURCE_UPDATED, agent_name, {"key": key})

    def read_shared(self, key: str, default: Any = None) -> Any:
        """Read from shared memory"""
        return self._shared_memory.get(key, default)

    def get_all_shared(self) -> Dict[str, Any]:
        """Get snapshot of all shared memory"""
        return self._shared_memory.copy()

    # --- Task Management ---

    def add_task(self, task_id: str, description: str, dependencies: List[str] = None, assigned_to: str = None) -> None:
        """Register a new task in the swarm"""
        self._task_graph[task_id] = {
            "description": description,
            "status": "pending",
            "dependencies": dependencies or [],
            "assigned_to": assigned_to,
            "created_at": time.time()
        }

    def update_task_status(self, task_id: str, status: str, agent_name: str, result: Any = None) -> None:
        """Update task status"""
        if task_id in self._task_graph:
            self._task_graph[task_id]["status"] = status
            if result:
                self._task_graph[task_id]["result"] = result
            
            event = SwarmEvent.TASK_COMPLETED if status == "done" else SwarmEvent.TASK_FAILED
            self.publish(event, agent_name, {"task_id": task_id, "status": status})

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get executable tasks (dependencies met)"""
        pending = []
        for tid, info in self._task_graph.items():
            if info["status"] == "pending":
                # Check dependencies
                deps_met = all(
                    self._task_graph.get(dep, {}).get("status") == "done" 
                    for dep in info["dependencies"]
                )
                if deps_met:
                    pending.append({"id": tid, **info})
        return pending
        
    # --- Persistent Memory (Vector Store) ---
    
    def init_vector_memory(self, collection_name: str = "swarm_insights") -> bool:
        """Initialize ChromaDB for persistent insight storage."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Use persistent storage in .swarm_memory
            persist_dir = ".swarm_memory"
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            
            self.memory_collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logging.info(f"SwarmContext: Vector memory initialized at {persist_dir}")
            return True
        except ImportError:
            logging.warning("SwarmContext: chromadb not installed, persistent memory disabled.")
            return False
        except Exception as e:
            logging.error(f"SwarmContext: Failed to init vector memory: {e}")
            return False

    def store_insight_vector(self, insight: str, metadata: Dict[str, Any], insight_id: str) -> None:
        """Store an insight in vector memory."""
        if hasattr(self, 'memory_collection') and self.memory_collection:
            try:
                self.memory_collection.add(
                    documents=[insight],
                    metadatas=[metadata],
                    ids=[insight_id]
                )
                logging.debug(f"Stored insight {insight_id} in vector memory")
            except Exception as e:
                logging.error(f"Failed to store insight in vector memory: {e}")

    def query_insights(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant insights from vector memory."""
        if hasattr(self, 'memory_collection') and self.memory_collection:
            try:
                results = self.memory_collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
                # Parse results into friendly format
                insights = []
                if results['documents']:
                    for i, doc in enumerate(results['documents'][0]):
                        meta = results['metadatas'][0][i] if results['metadatas'] else {}
                        insights.append({"content": doc, "metadata": meta})
                return insights
            except Exception as e:
                logging.error(f"Failed to query vector memory: {e}")
                return []
        return []
