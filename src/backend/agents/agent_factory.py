"""
Agent Factory Module
====================
Creates and manages CrewAI agents for different analysis tasks.
Extracted from crew_manager.py for better maintainability.
"""

import logging
from typing import Optional

from backend.agents.model_initializer import get_model_initializer


# Singleton instance
_agent_factory: Optional['AgentFactory'] = None


class AgentFactory:
    """
    Factory for creating specialized CrewAI agents.
    Uses lazy loading to create agents only when needed.
    """
    
    def __init__(self):
        self._initializer = get_model_initializer()
        
        # Agent instances (lazy loaded)
        self._data_analyst = None
        self._rag_specialist = None
        self._reviewer = None
        self._visualizer = None
        self._reporter = None
        
        logging.info("🏭 AgentFactory created (lazy loading enabled)")
    
    @property
    def data_analyst(self):
        """Get or create the Data Analyst agent."""
        if self._data_analyst is None:
            self._data_analyst = self._create_data_analyst()
        return self._data_analyst
    
    @property
    def rag_specialist(self):
        """Get or create the RAG Specialist agent."""
        if self._rag_specialist is None:
            self._rag_specialist = self._create_rag_specialist()
        return self._rag_specialist
    
    @property
    def reviewer(self):
        """Get or create the Reviewer agent."""
        if self._reviewer is None:
            self._reviewer = self._create_reviewer()
        return self._reviewer
    
    @property
    def visualizer(self):
        """Get or create the Visualizer agent."""
        if self._visualizer is None:
            self._visualizer = self._create_visualizer()
        return self._visualizer
    
    @property
    def reporter(self):
        """Get or create the Reporter agent."""
        if self._reporter is None:
            self._reporter = self._create_reporter()
        return self._reporter
    
    def _create_data_analyst(self):
        """Create the Data Analyst agent."""
        try:
            from crewai import Agent
            
            return Agent(
                role="Senior Data Analyst",
                goal="Analyze data accurately and provide actionable insights",
                backstory="""You are an expert data analyst with years of experience
                in statistical analysis, data visualization, and business intelligence.
                You excel at finding patterns in data and explaining complex findings
                in simple terms.""",
                llm=self._initializer.primary_llm,
                tools=self._initializer.tools,
                verbose=False,
                allow_delegation=False,
                max_iter=3
            )
        except Exception as e:
            logging.error(f"Failed to create Data Analyst agent: {e}")
            return None
    
    def _create_rag_specialist(self):
        """Create the RAG Specialist agent."""
        try:
            from crewai import Agent
            
            return Agent(
                role="Document Analysis Specialist",
                goal="Extract insights from unstructured documents using RAG techniques",
                backstory="""You are an expert in document analysis and retrieval-augmented
                generation. You can efficiently process PDFs, text files, and other documents
                to answer questions and extract relevant information.""",
                llm=self._initializer.primary_llm,
                tools=self._initializer.tools,
                verbose=False,
                allow_delegation=False,
                max_iter=3
            )
        except Exception as e:
            logging.error(f"Failed to create RAG Specialist agent: {e}")
            return None
    
    def _create_reviewer(self):
        """Create the Reviewer agent."""
        try:
            from crewai import Agent
            
            return Agent(
                role="Quality Assurance Reviewer",
                goal="Review and validate analysis results for accuracy and quality",
                backstory="""You are a meticulous reviewer with expertise in statistical
                validation and quality assurance. You verify calculations, check for errors,
                and ensure analysis conclusions are well-supported by the data.""",
                llm=self._initializer.review_llm,
                verbose=False,
                allow_delegation=False,
                max_iter=2
            )
        except Exception as e:
            logging.error(f"Failed to create Reviewer agent: {e}")
            return None
    
    def _create_visualizer(self):
        """Create the Visualizer agent."""
        try:
            from crewai import Agent
            
            return Agent(
                role="Data Visualization Expert",
                goal="Create clear, informative, and beautiful data visualizations",
                backstory="""You are an expert in data visualization with deep knowledge
                of Plotly, Matplotlib, and modern visualization best practices. You create
                charts that effectively communicate data insights.""",
                llm=self._initializer.primary_llm,
                tools=self._initializer.tools,
                verbose=False,
                allow_delegation=False,
                max_iter=3
            )
        except Exception as e:
            logging.error(f"Failed to create Visualizer agent: {e}")
            return None
    
    def _create_reporter(self):
        """Create the Reporter agent."""
        try:
            from crewai import Agent
            
            return Agent(
                role="Business Report Writer",
                goal="Create comprehensive and professional analysis reports",
                backstory="""You are a skilled business analyst and technical writer.
                You excel at transforming complex data analysis into clear, actionable
                reports for stakeholders at all levels.""",
                llm=self._initializer.primary_llm,
                verbose=False,
                allow_delegation=False,
                max_iter=3
            )
        except Exception as e:
            logging.error(f"Failed to create Reporter agent: {e}")
            return None
    
    def create_custom_agent(self, role: str, goal: str, backstory: str, use_review_llm: bool = False):
        """Create a custom agent with specified parameters."""
        try:
            from crewai import Agent
            
            llm = self._initializer.review_llm if use_review_llm else self._initializer.primary_llm
            
            return Agent(
                role=role,
                goal=goal,
                backstory=backstory,
                llm=llm,
                tools=self._initializer.tools,
                verbose=False,
                allow_delegation=False,
                max_iter=3
            )
        except Exception as e:
            logging.error(f"Failed to create custom agent: {e}")
            return None

    def create_agent(self, agent_type: str, **kwargs):
        """
        Legacy wrapper for agent creation to support tests and dynamic loading.
        Fixes 'AttributeError: AgentFactory has no attribute create_agent'.
        """
        agent_map = {
            'data_analyst': self.data_analyst,
            'statistical': self.data_analyst,  # Alias
            'rag_specialist': self.rag_specialist,
            'rag': self.rag_specialist,        # Alias
            'reviewer': self.reviewer,
            'visualizer': self.visualizer,
            'reporter': self.reporter
        }
        
        agent_key = agent_type.lower()
        agent = agent_map.get(agent_key)
        
        if not agent:
            # Check for custom agent requests or raise error
            logging.warning(f"Requested unknown agent type: {agent_type}")
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        return agent


def get_agent_factory() -> AgentFactory:
    """Get the singleton AgentFactory instance."""
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = AgentFactory()
    return _agent_factory
