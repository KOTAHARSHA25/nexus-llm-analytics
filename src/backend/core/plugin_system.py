"""Plug-and-Play Agent Architecture.

Modular system for discovering, registering, and routing queries to
specialist plugin agents at runtime without code changes.

Key classes:

* **AgentCapability** — Enum of recognised agent capabilities.
* **AgentMetadata** — Dataclass carrying agent identification and
  resource requirements.
* **BasePluginAgent** — Abstract base class every plugin must extend.
* **AgentRegistry** — Central registry with auto-discovery,
  capability-based routing, and hot-reload.

Enterprise v2.0 Additions
-------------------------
* **Swarm Integration** — Agents can now share context and coordinate via `SwarmContext`.
* **Reflective Execution** — `reflective_execute` method for plan-exec-critique loops.
* **PluginHealthReport** — Dataclass summarising per-plugin health
  (initialised, dependency status, last-error).

All v1.x APIs remain fully backward-compatible.

Author: Nexus Team
Since: v1.0 (Swarm enhancements v2.1 — February 2026)
"""

import json
import logging
import os
import threading
import importlib
import importlib.util
import inspect
from typing import Dict, Any, List, Optional, Type, Callable
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from backend.core.optimizers import OptimizedAgentMixin
# Swarm Import
from backend.core.swarm import SwarmContext, SwarmEvent

class AgentCapability(Enum):
    """Enumeration of agent capabilities"""
    DATA_ANALYSIS = "data_analysis"
    DOCUMENT_PROCESSING = "document_processing" 
    VISUALIZATION = "visualization"
    REPORTING = "reporting"
    SQL_QUERYING = "sql_querying"
    WEB_SCRAPING = "web_scraping"
    MACHINE_LEARNING = "machine_learning"
    TIME_SERIES = "time_series"
    NATURAL_LANGUAGE = "natural_language"

@dataclass
class AgentMetadata:
    """Metadata for agent registration"""
    name: str
    version: str
    description: str
    author: str
    capabilities: List[AgentCapability]
    file_types: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    min_ram_mb: int = 512
    max_timeout_seconds: int = 300
    priority: int = 50  # Higher = higher priority when multiple agents match

class BasePluginAgent(ABC, OptimizedAgentMixin):
    """
    Abstract base class for all plugin agents
    
    This defines the contract that all plugin agents must implement
    Now includes OptimizedAgentMixin for automatic performance tracking
    and Swarm coordination capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        OptimizedAgentMixin.__init__(self)  # Initialize optimization mixin
        self.config = config or {}
        self.metadata = self.get_metadata()
        self.initialized = False
        self.registry = None  # Will be injected by AgentRegistry
        self.swarm_context = None # Will be injected by AgentRegistry
        self.enable_verbose_logging = os.environ.get('NEXUS_VERBOSE_LOGGING', 'false').lower() == 'true'
    
    # --- Swarm Capabilities ---

    def publish_insight(self, topic: str, content: Any) -> None:
        """Share an insight with the swarm"""
        if self.swarm_context:
            self.swarm_context.publish(
                SwarmEvent.INSIGHT_FOUND,
                self.metadata.name,
                {"topic": topic, "content": content}
            )

    def subscribe_topic(self, topic: str, callback: Callable) -> None:
        """Subscribe to a specific topic (placeholder for future topic filtering)"""
        # Currently we just subscribe to general types via context, extending later
        pass

    def reflective_execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Advanced execution with Self-Correction (Plan -> Execute -> Critique -> Refine).
        
        Default implementation wraps standard execution. 
        Subclasses should override for specific cognitive loops.
        """
        return self.execute_with_logging(query, data, **kwargs)

    # --- Core Methods ---

    def delegate(self, agent_name: str, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Delegate a sub-task to a specific agent by name.
        
        Args:
            agent_name: Name of the agent to call (e.g., "TimeSeriesAgent")
            query: The query/instruction for the sub-agent
            data: Optional data to pass to the sub-agent
            **kwargs: Additional context
            
        Returns:
            Dict containing the execution result
        """
        if not self.registry:
            return {"success": False, "error": "No registry available for delegation", "agent": self.metadata.name}
            
        target_agent = self.registry.get_agent(agent_name)
        if not target_agent:
            return {"success": False, "error": f"Agent {agent_name} not found", "agent": self.metadata.name}
            
        # Circular dependency protection
        current_depth = kwargs.get('recursion_depth', 0)
        if current_depth > 3:
            return {"success": False, "error": "Max delegation depth reached (possible infinite loop)", "agent": self.metadata.name}
        
        # Pass updated depth
        kwargs['recursion_depth'] = current_depth + 1
        
        # Add trace info
        kwargs['parent_agent'] = self.metadata.name
        
        return target_agent.execute_with_logging(query, data, **kwargs)

    def delegate_by_capability(self, capability: AgentCapability, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Delegate to the best available agent for a specific capability.
        """
        if not self.registry:
             return {"success": False, "error": "No registry available", "agent": self.metadata.name}
             
        agents = self.registry.get_agents_by_capability(capability)
        if not agents:
            return {"success": False, "error": f"No agent found for capability {capability.value}", "agent": self.metadata.name}
            
        # Pick the highest priority agent
        # (In future, could use more complex routing logic here)
        best_agent = sorted(agents, key=lambda a: a.metadata.priority, reverse=True)[0]
        
        return self.delegate(best_agent.metadata.name, query, data, **kwargs)

    def _log_execution(self, query: str, result: Dict[str, Any], execution_time: float):
        """
        Log agent execution details for backend visibility.
        Mirrors what the frontend sees in the UI.
        """
        import logging
        logger = logging.getLogger(f"agent.{self.metadata.name}")
        
        # Create a formatted log entry
        log_lines = [
            "\n" + "="*80,
            f"🤖 AGENT EXECUTION: {self.metadata.name}",
            "="*80,
            f"📝 Query: {query[:200]}{'...' if len(query) > 200 else ''}",
            f"⏱️  Execution Time: {execution_time:.2f}s",
            f"✅ Success: {result.get('success', False)}",
        ]
        
        # Add result summary
        if result.get('success'):
            result_text = str(result.get('result', ''))
            log_lines.append(f"📊 Result Preview: {result_text[:300]}{'...' if len(result_text) > 300 else ''}")
            
            # Log metadata if present
            metadata = result.get('metadata', {})
            if metadata:
                log_lines.append(f"🔍 Metadata: {list(metadata.keys())}")
                if 'code' in metadata or 'executed_code' in metadata:
                    log_lines.append(f"   - Code Generated: Yes")
                if 'visualization' in metadata:
                    log_lines.append(f"   - Visualization: Yes")
        else:
            log_lines.append(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        log_lines.append("="*80 + "\n")
        
        # Log to console (INFO level so it shows by default)
        logger.info("\n".join(log_lines))
    
    def execute_with_logging(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Wrapper around execute() that adds logging for backend visibility.
        This ensures all agent executions are visible in the backend logs.
        """
        import time
        start_time = time.time()
        
        try:
            result = self.execute(query, data, **kwargs)
            execution_time = time.time() - start_time
            
            # Log the execution
            self._log_execution(query, result, execution_time)
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = {
                'success': False,
                'error': str(e),
                'agent': self.metadata.name
            }
            self._log_execution(query, error_result, execution_time)
            raise
    
    @abstractmethod
    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata"""
        pass
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize the agent with required dependencies"""
        pass
    
    @abstractmethod
    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs) -> float:
        """
        Check if agent can handle the query
        
        Returns:
            float: Confidence score 0.0-1.0 (0 = cannot handle, 1 = perfect match)
        """
        pass
    
    @abstractmethod
    def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's main functionality
        
        Returns:
            Dict with 'success', 'result', and optional 'error' keys
        """
        pass
    
    def validate_dependencies(self) -> List[str]:
        """Check if all dependencies are available"""
        # Map package names to import names (for cases where they differ)
        package_to_import = {
            'scikit-learn': 'sklearn',
            'python-dotenv': 'dotenv',
            'Pillow': 'PIL',
        }
        
        missing = []
        for dep in self.metadata.dependencies:
            # Get the actual import name (may differ from package name)
            import_name = package_to_import.get(dep, dep)
            try:
                importlib.import_module(import_name)
            except ImportError:
                missing.append(dep)
        return missing
    
    def get_resource_requirements(self) -> Dict[str, Any]:
        """Get resource requirements for this agent"""
        return {
            "min_ram_mb": self.metadata.min_ram_mb,
            "max_timeout_seconds": self.metadata.max_timeout_seconds,
            "capabilities": [cap.value for cap in self.metadata.capabilities]
        }

class AgentRegistry:
    """
    Central registry for managing plugin agents
    
    Features:
    - Automatic agent discovery
    - Capability-based routing
    - Resource-aware scheduling
    - Hot-reloading of agents
    - **Swarm Coordination** (New)
    """
    
    def __init__(self, plugins_directory: str = None):
        # Use absolute path based on this file's location
        if plugins_directory is None:
            backend_dir = Path(__file__).parent.parent
            plugins_directory = backend_dir / "plugins"
        
        self.plugins_directory = Path(plugins_directory)
        self.registered_agents: Dict[str, BasePluginAgent] = {}
        self.capability_index: Dict[AgentCapability, List[str]] = {}
        self.file_type_index: Dict[str, List[str]] = {}
        self.agent_configs: Dict[str, Dict[str, Any]] = {}

        # Swarm Initialization
        self.swarm_context = SwarmContext()
        
        # Create plugins directory if it doesn't exist
        self.plugins_directory.mkdir(parents=True, exist_ok=True)
        
        logging.debug(f"AgentRegistry initialized - plugins directory: {self.plugins_directory}")
    
    def discover_agents(self) -> int:
        """
        Automatically discover and load plugin agents
        
        Returns number of agents discovered
        """
        discovered = 0
        
        # Load from plugins directory
        for plugin_file in self.plugins_directory.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue  # Skip private files
                
            try:
                discovered += self._load_agent_from_file(plugin_file)
            except Exception as e:
                logging.error(f"Failed to load plugin {plugin_file}: {e}")
        
        # Load configuration overrides
        config_file = self.plugins_directory / "agents_config.json"
        if config_file.exists():
            self._load_agent_configs(config_file)
        
        logging.debug(f"Discovered {discovered} plugin agents")
        return discovered
    
    def _load_agent_from_file(self, plugin_file: Path) -> int:
        """Load agent class from Python file.

        Registers the module in ``sys.modules`` before execution so that
        Python 3.13+ ``@dataclass`` processing (which looks up
        ``cls.__module__`` in ``sys.modules``) works correctly for
        dynamically-loaded plugin files.
        """
        import sys as _sys

        module_name = plugin_file.stem
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        if spec is None or spec.loader is None:
            logging.warning("Could not create module spec for %s", plugin_file)
            return 0

        module = importlib.util.module_from_spec(spec)

        # Register in sys.modules BEFORE exec_module so that @dataclass
        # and other stdlib machinery that does
        #   sys.modules.get(cls.__module__)
        # can find the module.  This is the pattern recommended by the
        # Python docs for importing a source file directly.
        _sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            # Roll back registration on failure so we don't leave a
            # broken module in sys.modules.
            _sys.modules.pop(module_name, None)
            raise
        
        agents_found = 0
        
        # Find all BasePluginAgent subclasses in the module
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if (issubclass(cls, BasePluginAgent) and 
                cls != BasePluginAgent and 
                not inspect.isabstract(cls)):
                
                try:
                    # Instantiate and register the agent
                    config = self.agent_configs.get(name, {})
                    agent = cls(config=config)
                    self.register_agent(agent)
                    agents_found += 1
                    logging.debug(f"Loaded plugin agent: {name}")
                except Exception as e:
                    logging.error(f"Failed to instantiate agent {name}: {e}")
        
        return agents_found
    
    def _load_agent_configs(self, config_file: Path):
        """Load agent configuration overrides"""
        try:
            with open(config_file, 'r') as f:
                self.agent_configs = json.load(f)
            logging.debug(f"Loaded agent configurations from {config_file}")
        except Exception as e:
            logging.error(f"Failed to load agent configs: {e}")
    
    def register_agent(self, agent: BasePluginAgent) -> bool:
        """
        Register a plugin agent
        
        Returns True if successful, False otherwise
        """
        try:
            # Validate dependencies
            missing_deps = agent.validate_dependencies()
            if missing_deps:
                logging.debug(f"Plugin {agent.metadata.name} skipped (missing optional dependencies: {missing_deps})")
                return False
            
            # Inject registry for delegation
            agent.registry = self
            # Inject swarm context for coordination
            agent.swarm_context = self.swarm_context
            
            # Initialize the agent
            if not agent.initialize(registry=self):
                logging.debug(f"Plugin {agent.metadata.name} failed to initialize (this is optional)")
                return False
            
            # Register in main registry
            self.registered_agents[agent.metadata.name] = agent
            
            # Index by capabilities
            for capability in agent.metadata.capabilities:
                if capability not in self.capability_index:
                    self.capability_index[capability] = []
                self.capability_index[capability].append(agent.metadata.name)
            
            # Index by file types
            for file_type in agent.metadata.file_types:
                if file_type not in self.file_type_index:
                    self.file_type_index[file_type] = []
                self.file_type_index[file_type].append(agent.metadata.name)
            
            logging.debug(f"Registered agent: {agent.metadata.name} v{agent.metadata.version}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to register agent {agent.metadata.name}: {e}")
            return False
    
    def get_agent(self, name: str) -> Optional[BasePluginAgent]:
        """Get a registered agent by name"""
        return self.registered_agents.get(name)

    def route_query(self, query: str, file_type: Optional[str] = None, **kwargs) -> tuple:
        """
        Route a query to the best agent.
        
        Returns:
            tuple: (topic, confidence, agent)
        """
        best_agent = None
        best_score = 0.0
        best_capability = None
        
        # Reuse logic similar to find_best_agent but capture score
        candidates = []
        if file_type:
            candidates.extend(self.file_type_index.get(file_type, []))
        if not candidates:
            candidates = list(self.registered_agents.keys())
            
        for agent_name in candidates:
            agent = self.registered_agents[agent_name]
            try:
                confidence = agent.can_handle(query, file_type, **kwargs)
                if confidence > 0:
                    score = confidence * 0.8 + (agent.metadata.priority / 100) * 0.2
                    if score > best_score:
                        best_score = score
                        best_agent = agent
                        # Guess primary capability
                        best_capability = agent.metadata.capabilities[0].value if agent.metadata.capabilities else "general"
            except Exception as e:
                logging.error(f"Error checking agent {agent_name}: {e}")
                
        return best_capability, best_score, best_agent

    def find_best_agent(self, query: str, file_type: Optional[str] = None, **kwargs) -> Optional[BasePluginAgent]:
        """
        Find the best agent to handle a specific query
        """
        _, _, agent = self.route_query(query, file_type, **kwargs)
        return agent
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[BasePluginAgent]:
        """Get all agents with a specific capability"""
        agent_names = self.capability_index.get(capability, [])
        return [self.registered_agents[name] for name in agent_names]
    
    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all registered agents with their metadata"""
        return {
            name: {
                "version": agent.metadata.version,
                "description": agent.metadata.description,
                "capabilities": [cap.value for cap in agent.metadata.capabilities],
                "file_types": agent.metadata.file_types,
                "priority": agent.metadata.priority,
                "initialized": agent.initialized
            }
            for name, agent in self.registered_agents.items()
        }
    
    def reload_agent(self, agent_name: str) -> bool:
        """Hot-reload a specific agent"""
        logging.debug(f"Hot-reloading agent: {agent_name}")
        
        # 1. Get existing agent
        old_agent = self.registered_agents.get(agent_name)
        if not old_agent:
            logging.warning(f"Cannot reload unknown agent: {agent_name}")
            return False
            
        try:
            # 2. Find source file
            file_path = inspect.getfile(old_agent.__class__)
            plugin_file = Path(file_path)
            
            if not plugin_file.exists():
                logging.error(f"Source file for agent {agent_name} not found: {file_path}")
                return False
                
            # 3. Clean up indexes before reloading
            # Remove from registered_agents
            del self.registered_agents[agent_name]
            
            # Remove from capability index
            for cap, agents in self.capability_index.items():
                if agent_name in agents:
                    agents.remove(agent_name)
                    
            # Remove from file type index
            for ft, agents in self.file_type_index.items():
                if agent_name in agents:
                    agents.remove(agent_name)
            
            # 4. Reload from file
            # This creates a new module and re-registers the agent(s) found in it
            agents_loaded = self._load_agent_from_file(plugin_file)
            
            if agents_loaded > 0 and agent_name in self.registered_agents:
                logging.info(f"Successfully hot-reloaded agent: {agent_name}")
                return True
            else:
                logging.warning(f"Reloaded file but agent {agent_name} was not re-registered")
                return False
                
        except Exception as e:
            logging.error(f"Failed to hot-reload agent {agent_name}: {e}")
            # Try to restore old agent if simple unregister happened
            if agent_name not in self.registered_agents:
                self.registered_agents[agent_name] = old_agent
                # Note: Indexes might be inconsistent now, but better than losing the agent entirely
            return False
    
    # get_system_resources() removed — was dead code with missing psutil dependency

# Thread-safe Global registry instance
_global_registry: Optional[AgentRegistry] = None
_registry_lock = threading.Lock()

def get_agent_registry(plugins_dir: str = None) -> AgentRegistry:
    """Get or create the global agent registry (thread-safe)."""
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            # Double-check pattern for thread safety
            if _global_registry is None:
                _global_registry = AgentRegistry(plugins_dir)
                _global_registry.discover_agents()
    return _global_registry

def initialize_plugin_system(plugins_dir: str = None) -> AgentRegistry:
    """Initialize the plugin system and return the registry"""
    logging.debug("Initializing plug-and-play agent system...")
    registry = get_agent_registry(plugins_dir)
    
    # Create example plugin structure
    if plugins_dir is None:
        backend_dir = Path(__file__).parent.parent
        example_dir = backend_dir / "plugins"
    else:
        example_dir = Path(plugins_dir)
    example_dir.mkdir(exist_ok=True)
    
    # Create example config file
    example_config = {
        "SQLAgent": {
            "database_url": "sqlite:///example.db",
            "query_timeout": 30,
            "max_results": 1000
        },
        "WebScrapingAgent": {
            "user_agent": "NexusLLM-Bot/1.0",
            "request_timeout": 10,
            "max_pages": 50
        }
    }
    
    config_file = example_dir / "agents_config.json"
    if not config_file.exists():
        with open(config_file, 'w') as f:
            json.dump(example_config, f, indent=2)
    
    logging.debug(f"Plugin system initialized with {len(registry.registered_agents)} agents")
    return registry


# ============================================================================
# Enterprise v2.0 — PluginHealthReport
# ============================================================================

from dataclasses import dataclass as _dataclass, field as _field
import datetime as _dt


@_dataclass
class PluginHealthReport:
    """Per-plugin health summary for observability dashboards.

    Attributes:
        agent_name: Registered name of the plugin agent.
        version: Plugin version string.
        initialised: Whether :meth:`BasePluginAgent.initialize` succeeded.
        missing_deps: List of missing Python dependencies.
        capabilities: List of capability value strings.
        last_error: Most recent error message, or ``None``.
        checked_at: ISO-8601 timestamp of the health check.

    .. versionadded:: 2.0
    """

    agent_name: str
    version: str
    initialised: bool
    missing_deps: list = _field(default_factory=list)
    capabilities: list = _field(default_factory=list)
    last_error: str | None = None
    checked_at: str = _field(
        default_factory=lambda: _dt.datetime.now().isoformat()
    )

    @classmethod
    def from_agent(cls, agent: BasePluginAgent) -> "PluginHealthReport":
        """Build a health report from an instantiated agent.

        Args:
            agent: The plugin agent to inspect.

        Returns:
            A fully populated :class:`PluginHealthReport`.
        """
        missing = agent.validate_dependencies()
        return cls(
            agent_name=agent.metadata.name,
            version=agent.metadata.version,
            initialised=agent.initialized,
            missing_deps=missing,
            capabilities=[c.value for c in agent.metadata.capabilities],
        )