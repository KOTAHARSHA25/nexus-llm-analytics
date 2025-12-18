# WebSocket Manager for Real-time Updates
# Provides real-time communication between backend and frontend

from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
import logging
from datetime import datetime
from enum import Enum

class MessageType(Enum):
    """WebSocket message types"""
    ANALYSIS_START = "analysis_start"
    ANALYSIS_PROGRESS = "analysis_progress"
    ANALYSIS_COMPLETE = "analysis_complete"
    ANALYSIS_ERROR = "analysis_error"
    HEARTBEAT = "heartbeat"
    NOTIFICATION = "notification"
    STATUS_UPDATE = "status_update"

class ConnectionManager:
    """Manages WebSocket connections and message broadcasting"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.message_queue: Dict[str, List[Dict]] = {}
        self.logger = logging.getLogger(__name__)
        
    async def connect(self, websocket: WebSocket, client_id: str, metadata: Optional[Dict] = None):
        """
        Accept and store a new WebSocket connection
        
        Args:
            websocket: WebSocket connection object
            client_id: Unique identifier for the client
            metadata: Optional metadata about the connection
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = metadata or {}
        self.connection_metadata[client_id]["connected_at"] = datetime.utcnow().isoformat()
        
        # Send initial connection confirmation
        await self.send_personal_message(
            client_id,
            MessageType.STATUS_UPDATE,
            {"status": "connected", "client_id": client_id}
        )
        
        self.logger.info(f"WebSocket client {client_id} connected")
        
        # Send any queued messages
        if client_id in self.message_queue:
            for message in self.message_queue[client_id]:
                await self.send_json(websocket, message)
            self.message_queue[client_id].clear()
    
    def disconnect(self, client_id: str):
        """
        Remove a WebSocket connection
        
        Args:
            client_id: Unique identifier for the client
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]
            self.logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_personal_message(
        self,
        client_id: str,
        message_type: MessageType,
        data: Dict[str, Any],
        queue_if_offline: bool = True
    ):
        """
        Send a message to a specific client
        
        Args:
            client_id: Target client ID
            message_type: Type of message
            data: Message payload
            queue_if_offline: Whether to queue message if client is offline
        """
        message = {
            "type": message_type.value,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await self.send_json(websocket, message)
            except Exception as e:
                self.logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
                if queue_if_offline:
                    self._queue_message(client_id, message)
        elif queue_if_offline:
            self._queue_message(client_id, message)
    
    async def broadcast(self, message_type: MessageType, data: Dict[str, Any]):
        """
        Broadcast a message to all connected clients
        
        Args:
            message_type: Type of message
            data: Message payload
        """
        message = {
            "type": message_type.value,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await self.send_json(websocket, message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def send_json(self, websocket: WebSocket, data: Dict):
        """
        Send JSON data through WebSocket
        
        Args:
            websocket: WebSocket connection
            data: Data to send
        """
        await websocket.send_json(data)
    
    def _queue_message(self, client_id: str, message: Dict):
        """Queue a message for offline client"""
        if client_id not in self.message_queue:
            self.message_queue[client_id] = []
        
        # Limit queue size to prevent memory issues
        if len(self.message_queue[client_id]) < 100:
            self.message_queue[client_id].append(message)
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_connected_clients(self) -> List[str]:
        """Get list of connected client IDs"""
        return list(self.active_connections.keys())
    
    async def heartbeat(self, interval: int = 30):
        """
        Send heartbeat messages to keep connections alive
        
        Args:
            interval: Seconds between heartbeats
        """
        while True:
            await asyncio.sleep(interval)
            await self.broadcast(MessageType.HEARTBEAT, {"timestamp": datetime.utcnow().isoformat()})

class AnalysisProgressTracker:
    """Track and report analysis progress via WebSocket"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_analyses: Dict[str, Dict[str, Any]] = {}
    
    async def start_analysis(self, analysis_id: str, client_id: str, query: str):
        """
        Mark analysis as started and notify client
        
        Args:
            analysis_id: Unique analysis identifier
            client_id: Client who initiated the analysis
            query: Analysis query
        """
        self.active_analyses[analysis_id] = {
            "client_id": client_id,
            "query": query,
            "started_at": datetime.utcnow().isoformat(),
            "status": "started",
            "progress": 0
        }
        
        await self.connection_manager.send_personal_message(
            client_id,
            MessageType.ANALYSIS_START,
            {
                "analysis_id": analysis_id,
                "query": query,
                "message": "Analysis started..."
            }
        )
    
    async def update_progress(
        self,
        analysis_id: str,
        progress: int,
        message: str,
        details: Optional[Dict] = None
    ):
        """
        Update analysis progress and notify client
        
        Args:
            analysis_id: Analysis identifier
            progress: Progress percentage (0-100)
            message: Progress message
            details: Optional additional details
        """
        if analysis_id not in self.active_analyses:
            return
        
        self.active_analyses[analysis_id]["progress"] = progress
        self.active_analyses[analysis_id]["last_update"] = datetime.utcnow().isoformat()
        
        client_id = self.active_analyses[analysis_id]["client_id"]
        
        await self.connection_manager.send_personal_message(
            client_id,
            MessageType.ANALYSIS_PROGRESS,
            {
                "analysis_id": analysis_id,
                "progress": progress,
                "message": message,
                "details": details or {}
            }
        )
    
    async def complete_analysis(
        self,
        analysis_id: str,
        result: Dict[str, Any],
        execution_time: float
    ):
        """
        Mark analysis as complete and send results
        
        Args:
            analysis_id: Analysis identifier
            result: Analysis results
            execution_time: Time taken for analysis
        """
        if analysis_id not in self.active_analyses:
            return
        
        self.active_analyses[analysis_id]["status"] = "completed"
        self.active_analyses[analysis_id]["completed_at"] = datetime.utcnow().isoformat()
        
        client_id = self.active_analyses[analysis_id]["client_id"]
        
        await self.connection_manager.send_personal_message(
            client_id,
            MessageType.ANALYSIS_COMPLETE,
            {
                "analysis_id": analysis_id,
                "result": result,
                "execution_time": execution_time,
                "message": "Analysis completed successfully"
            }
        )
        
        # Clean up after a delay
        await asyncio.sleep(60)
        if analysis_id in self.active_analyses:
            del self.active_analyses[analysis_id]
    
    async def report_error(self, analysis_id: str, error: str, details: Optional[Dict] = None):
        """
        Report analysis error
        
        Args:
            analysis_id: Analysis identifier
            error: Error message
            details: Optional error details
        """
        if analysis_id not in self.active_analyses:
            return
        
        self.active_analyses[analysis_id]["status"] = "error"
        self.active_analyses[analysis_id]["error"] = error
        
        client_id = self.active_analyses[analysis_id]["client_id"]
        
        await self.connection_manager.send_personal_message(
            client_id,
            MessageType.ANALYSIS_ERROR,
            {
                "analysis_id": analysis_id,
                "error": error,
                "details": details or {},
                "message": "Analysis failed"
            }
        )
        
        # Clean up
        del self.active_analyses[analysis_id]

# Global instances
connection_manager = ConnectionManager()
progress_tracker = AnalysisProgressTracker(connection_manager)

# WebSocket endpoint handler
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    Handle WebSocket connections
    
    Args:
        websocket: WebSocket connection
        client_id: Client identifier
    """
    await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive and process messages from client
            data = await websocket.receive_json()
            
            # Handle different message types from client
            if data.get("type") == "ping":
                await connection_manager.send_personal_message(
                    client_id,
                    MessageType.HEARTBEAT,
                    {"pong": True}
                )
            elif data.get("type") == "subscribe":
                # Handle subscription requests
                pass
            
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as e:
        logging.error(f"WebSocket error for {client_id}: {e}")
        connection_manager.disconnect(client_id)
