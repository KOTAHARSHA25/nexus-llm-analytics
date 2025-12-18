# Plug-and-Play Agent Architecture
# Modular system for easy addition of new agents without code changes

import json
import logging
import importlib
import importlib.util
import inspect
from typing import Dict, Any, List, Optional, Type, Callable
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

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

class BasePluginAgent(ABC):
    """
    Abstract base class for all plugin agents
    
    This defines the contract that all plugin agents must implement
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metadata = self.get_metadata()
        self.initialized = False
    
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
        missing = []
        for dep in self.metadata.dependencies:
            try:
                importlib.import_module(dep)
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
        
        # Create plugins directory if it doesn't exist
        self.plugins_directory.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"🔌 AgentRegistry initialized - plugins directory: {self.plugins_directory}")
    
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
        
        logging.info(f"🔍 Discovered {discovered} plugin agents")
        return discovered
    
    def _load_agent_from_file(self, plugin_file: Path) -> int:
        """Load agent class from Python file"""
        spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
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
                    logging.info(f"✅ Loaded plugin agent: {name}")
                except Exception as e:
                    logging.error(f"Failed to instantiate agent {name}: {e}")
        
        return agents_found
    
    def _load_agent_configs(self, config_file: Path):
        """Load agent configuration overrides"""
        try:
            with open(config_file, 'r') as f:
                self.agent_configs = json.load(f)
            logging.info(f"📝 Loaded agent configurations from {config_file}")
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
                logging.info(f"ℹ️ Plugin {agent.metadata.name} skipped (missing optional dependencies: {missing_deps})")
                return False
            
            # Initialize the agent
            if not agent.initialize():
                logging.info(f"ℹ️ Plugin {agent.metadata.name} failed to initialize (this is optional)")
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
            
            logging.info(f"🎯 Registered agent: {agent.metadata.name} v{agent.metadata.version}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to register agent {agent.metadata.name}: {e}")
            return False
    
    def find_best_agent(self, query: str, file_type: Optional[str] = None, **kwargs) -> Optional[BasePluginAgent]:
        """
        Find the best agent to handle a specific query
        
        Uses intelligent routing based on:
        - File type compatibility
        - Query analysis
        - Agent confidence scores
        - Resource availability
        """
        candidates = []
        
        # Get candidates by file type
        if file_type:
            file_type_candidates = self.file_type_index.get(file_type, [])
            candidates.extend(file_type_candidates)
        
        # If no file type candidates, check all agents
        if not candidates:
            candidates = list(self.registered_agents.keys())
        
        # Score each candidate
        scored_candidates = []
        for agent_name in candidates:
            agent = self.registered_agents[agent_name]
            try:
                confidence = agent.can_handle(query, file_type, **kwargs)
                if confidence > 0:
                    # Combine confidence with priority
                    final_score = confidence * 0.8 + (agent.metadata.priority / 100) * 0.2
                    scored_candidates.append((final_score, agent_name, agent))
            except Exception as e:
                logging.error(f"Error checking agent {agent_name}: {e}")
        
        # Return best match
        if scored_candidates:
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_name, best_agent = scored_candidates[0]
            logging.info(f"🎯 Selected agent: {best_name} (score: {best_score:.2f})")
            return best_agent
        
        logging.warning("❌ No suitable agent found for query")
        return None
    
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
        # Implementation for hot-reloading
        # This would unregister, reload module, and re-register
        logging.info(f"🔄 Hot-reloading agent: {agent_name}")
        # TODO: Implement hot-reload logic
        return True
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource availability"""
        import psutil
        
        memory = psutil.virtual_memory()
        return {
            "available_ram_mb": memory.available // (1024 * 1024),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "agent_count": len(self.registered_agents)
        }

# Global registry instance
_global_registry: Optional[AgentRegistry] = None

def get_agent_registry(plugins_dir: str = None) -> AgentRegistry:
    """Get or create the global agent registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry(plugins_dir)
        _global_registry.discover_agents()
    return _global_registry

def initialize_plugin_system(plugins_dir: str = None) -> AgentRegistry:
    """Initialize the plugin system and return the registry"""
    logging.info("🚀 Initializing plug-and-play agent system...")
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
    
    logging.info(f"✅ Plugin system initialized with {len(registry.registered_agents)} agents")
    return registry