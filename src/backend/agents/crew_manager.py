"""
CrewAI Crew Manager - Central coordination for all agents
==========================================================
Refactored for Phase 1: Code Quality & Architecture

This module has been refactored from 1448 lines to ~300 lines.
Core functionality has been extracted to:
- model_initializer.py: Model initialization and lazy loading
- agent_factory.py: CrewAI agent creation
- analysis_executor.py: Structured data analysis execution
- rag_handler.py: Unstructured data (RAG) analysis

This file now serves as a facade/coordinator that delegates to specialized modules.
"""

import logging
import os
from typing import Dict, List, Any, Optional

from backend.core.sandbox import Sandbox
from backend.core.utils import friendly_error
from backend.core.plugin_system import get_agent_registry

# Import new modular components
from backend.agents.model_initializer import get_model_initializer
from backend.agents.agent_factory import get_agent_factory
from backend.agents.analysis_executor import get_analysis_executor
from backend.agents.rag_handler import get_rag_handler


# Singleton instance
_crew_manager_instance = None
_instance_lock = False


class CrewManager:
    """
    Central manager for CrewAI-based multi-agent data analysis system.
    
    REFACTORED ARCHITECTURE:
    - ModelInitializer: Handles lazy loading of LLM models
    - AgentFactory: Creates specialized CrewAI agents
    - AnalysisExecutor: Executes structured data analysis
    - RAGHandler: Handles unstructured data (RAG) analysis
    
    This class now acts as a coordinator/facade that delegates to these modules.
    """
    
    def __new__(cls):
        global _crew_manager_instance, _instance_lock
        
        if _crew_manager_instance is None and not _instance_lock:
            _instance_lock = True
            _crew_manager_instance = super(CrewManager, cls).__new__(cls)
            _crew_manager_instance._initialized = False
        
        return _crew_manager_instance
    
    def __init__(self):
        # Prevent multiple initialization of singleton
        if getattr(self, '_initialized', False):
            return
            
        logging.info("Initializing CrewAI Manager (refactored, lazy loading enabled)...")
        
        # Initialize lightweight components
        self.sandbox = Sandbox()
        self.plugin_registry = get_agent_registry()
        
        # Get singleton instances of modular components
        self._initializer = get_model_initializer()
        self._factory = get_agent_factory()
        self._executor = get_analysis_executor()
        self._rag = get_rag_handler()
        
        self._initialized = True
        logging.info("✅ CrewAI Manager initialized (models will load on first request)")
    
    # =========================================================================
    # PROPERTY ACCESSORS - Delegate to ModelInitializer
    # =========================================================================
    
    @property
    def llm_client(self):
        """Get the LLM client."""
        return self._initializer.llm_client
    
    @property
    def chroma_client(self):
        """Get the ChromaDB client."""
        return self._initializer.chroma_client
    
    @property
    def primary_llm(self):
        """Get the primary LLM."""
        return self._initializer.primary_llm
    
    @property
    def review_llm(self):
        """Get the review LLM."""
        return self._initializer.review_llm
    
    @property
    def intelligent_router(self):
        """Get the intelligent router."""
        return self._initializer.intelligent_router
    
    @property
    def tools(self):
        """Get analysis tools."""
        return self._initializer.tools
    
    @property
    def query_parser(self):
        """Get the query parser."""
        return self._initializer.query_parser
    
    # =========================================================================
    # AGENT ACCESSORS - Delegate to AgentFactory  
    # =========================================================================
    
    @property
    def data_analyst(self):
        """Get the Data Analyst agent."""
        return self._factory.data_analyst
    
    @property
    def rag_specialist(self):
        """Get the RAG Specialist agent."""
        return self._factory.rag_specialist
    
    @property
    def reviewer(self):
        """Get the Reviewer agent."""
        return self._factory.reviewer
    
    @property
    def visualizer(self):
        """Get the Visualizer agent."""
        return self._factory.visualizer
    
    @property
    def reporter(self):
        """Get the Reporter agent."""
        return self._factory.reporter
    
    # =========================================================================
    # BACKWARD COMPATIBILITY METHODS
    # =========================================================================
    
    def _ensure_models_initialized(self):
        """Ensure models are initialized (backward compatibility)."""
        self._initializer.ensure_initialized()
    
    def _ensure_cot_engine(self):
        """Ensure CoT engine is initialized (backward compatibility)."""
        return self._initializer.ensure_cot_engine()
    
    # =========================================================================
    # MAIN API METHODS
    # =========================================================================
    
    def analyze_structured_data(self, query: str, filename: str, **kwargs) -> Dict[str, Any]:
        """
        Handle structured data analysis queries.
        Delegates to AnalysisExecutor for the actual work.
        """
        return self._executor.analyze_structured(
            query=query,
            filename=filename,
            **kwargs
        )
    
    def analyze_unstructured_data(self, query: str, filename: str, **kwargs) -> Dict[str, Any]:
        """
        Handle unstructured data (RAG) analysis queries.
        Delegates to RAGHandler for the actual work.
        """
        return self._rag.analyze_unstructured(
            file_path=self._resolve_filepath(filename),
            query=query,
            enable_cot=kwargs.get('enable_cot', False)
        )
    
    def analyze_multiple_files(self, query: str, filenames: List[str], **kwargs) -> Dict[str, Any]:
        """
        Handle multi-file analysis with automatic join detection.
        """
        import pandas as pd
        
        analysis_id = kwargs.pop('analysis_id', None)
        
        logging.info(f"📊 Starting multi-file analysis for {len(filenames)} files")
        
        try:
            # Load all files
            dataframes = []
            
            for filename in filenames:
                filepath = self._resolve_filepath(filename)
                if not filepath:
                    return friendly_error(
                        f"File not found: {filename}",
                        "Please upload the file first"
                    )
                
                try:
                    df = pd.read_csv(filepath)
                    dataframes.append((filename, df))
                    logging.info(f"  Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")
                except Exception as e:
                    return friendly_error(
                        f"Error loading {filename}: {str(e)}",
                        "Ensure file is valid CSV format"
                    )
            
            # Auto-detect join keys and merge
            if len(dataframes) == 2:
                return self._merge_two_dataframes(dataframes, query, analysis_id, **kwargs)
            else:
                return friendly_error(
                    f"Multi-file analysis currently supports 2 files only",
                    f"Received {len(filenames)} files. Support for 3+ files coming soon."
                )
                
        except Exception as e:
            logging.error(f"Multi-file analysis error: {e}", exc_info=True)
            return friendly_error("Multi-file analysis failed", str(e))
    
    def _merge_two_dataframes(
        self, 
        dataframes: List[tuple], 
        query: str, 
        analysis_id: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Merge two dataframes and analyze."""
        import pandas as pd
        
        file1_name, df1 = dataframes[0]
        file2_name, df2 = dataframes[1]
        
        # Find common columns
        common_cols = set(df1.columns) & set(df2.columns)
        if not common_cols:
            return friendly_error(
                "No common columns found for joining",
                f"Files have no shared columns"
            )
        
        # Prioritize ID columns for joining
        join_key = None
        for col in common_cols:
            if '_id' in col.lower() or col.lower().endswith('id'):
                join_key = col
                break
        
        if not join_key:
            join_key = list(common_cols)[0]
        
        logging.info(f"  🔗 Joining on column: {join_key}")
        
        try:
            merged_df = pd.merge(df1, df2, on=join_key, how='inner')
            logging.info(f"  ✅ Merged: {len(merged_df)} rows, {len(merged_df.columns)} columns")
            
            # Save merged data temporarily
            base_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
            temp_filename = f"merged_{file1_name.replace('.csv', '')}_{file2_name}"
            temp_path = os.path.join(base_data_dir, 'uploads', temp_filename)
            merged_df.to_csv(temp_path, index=False)
            
            # Analyze merged data
            return self.analyze_structured_data(query, temp_filename, analysis_id=analysis_id, **kwargs)
            
        except Exception as e:
            return friendly_error(
                f"Error joining files: {str(e)}",
                f"Join key '{join_key}' may not be compatible between files"
            )
    
    def create_visualization(self, data_summary: str, chart_type: str = "auto") -> Dict[str, Any]:
        """Generate data visualizations."""
        try:
            self._initializer.ensure_initialized()
            
            Task = self._initializer.get_task_class()
            Crew = self._initializer.get_crew_class()
            Process = self._initializer.get_process_class()
            
            viz_task = Task(
                description=f"""
                Create a data visualization based on this data summary: {data_summary}
                
                Requirements:
                1. Choose the most appropriate chart type (or use '{chart_type}' if specified)
                2. Generate clean, professional Plotly code
                3. Ensure the visualization is interactive and informative
                4. Add appropriate titles, labels, and formatting
                5. Return the complete Python code
                """,
                agent=self.visualizer,
                expected_output="Complete Plotly visualization code with explanation"
            )
            
            crew = Crew(
                agents=[self.visualizer],
                tasks=[viz_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "success": True,
                "visualization_code": str(result),
                "chart_type": chart_type
            }
            
        except Exception as e:
            logging.error(f"Visualization creation failed: {e}")
            return friendly_error(f"Visualization failed: {str(e)}", "Check the data format")
    
    def generate_report(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive reports."""
        try:
            self._initializer.ensure_initialized()
            
            Task = self._initializer.get_task_class()
            Crew = self._initializer.get_crew_class()
            Process = self._initializer.get_process_class()
            
            report_task = Task(
                description=f"""
                Create a comprehensive professional report based on these analysis results:
                {analysis_results}
                
                The report should include:
                1. Executive Summary
                2. Key Findings
                3. Detailed Analysis
                4. Recommendations
                5. Methodology
                6. Appendix with technical details
                """,
                agent=self.reporter,
                expected_output="Professional formatted report with all sections"
            )
            
            crew = Crew(
                agents=[self.reporter],
                tasks=[report_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "success": True,
                "report": str(result),
                "format": "structured_text"
            }
            
        except Exception as e:
            logging.error(f"Report generation failed: {e}")
            return friendly_error(f"Report generation failed: {str(e)}", "Check the analysis results format")
    
    def handle_query(self, query: str, filename: str = None, filenames: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Main entry point for handling all types of queries.
        Uses intelligent routing with plugin system for extensibility.
        """
        # Normalize to list of files
        files = []
        if filenames:
            files = filenames
        elif filename:
            files = [filename]
        
        if not files:
            return friendly_error("No file provided", "Please upload a data file first")
        
        analysis_id = kwargs.get('analysis_id')
        
        # Handle forced model usage for review scenarios
        force_model = kwargs.pop('force_model', None)  # Use pop to avoid duplicate arg
        if force_model:
            return self._handle_forced_model_query(query, force_model, **kwargs)
        
        # Check for cancellation
        if analysis_id:
            from backend.core.analysis_manager import check_cancellation
            try:
                check_cancellation(analysis_id)
            except Exception:
                return {"error": "Analysis was cancelled", "status": "cancelled"}
        
        # Multi-file support
        if len(files) > 1:
            logging.info(f"🔗 Multi-file analysis requested: {files}")
            return self.analyze_multiple_files(query, files, **kwargs)
        
        # Single file analysis
        filename = files[0]
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Try plugin agents first
        plugin_agent = self.plugin_registry.find_best_agent(query, file_ext, **kwargs)
        if plugin_agent:
            logging.info(f"🔌 Using plugin agent: {plugin_agent.metadata.name}")
            try:
                result = plugin_agent.execute(query, filename=filename, **kwargs)
                if result.get("success"):
                    return result
            except Exception as e:
                logging.debug(f"Plugin agent error: {e}")
        
        # Fall back to built-in CrewAI agents
        if file_ext in ['.csv', '.json', '.xlsx', '.xls']:
            return self.analyze_structured_data(query, filename, **kwargs)
        elif file_ext in ['.pdf', '.txt', '.docx', '.pptx', '.rtf']:
            return self.analyze_unstructured_data(query, filename, **kwargs)
        else:
            return friendly_error(
                f"Unsupported file type: {filename}",
                f"Supported types: CSV, JSON, XLSX, XLS, PDF, TXT, DOCX, PPTX, RTF"
            )
    
    def _handle_forced_model_query(self, query: str, force_model: str, **kwargs) -> Dict[str, Any]:
        """Handle queries with forced model selection."""
        import time
        
        logging.info(f"🔧 Using forced model: {force_model}")
        
        try:
            start_time = time.time()
            
            review_prompt = f"""Analyze the following data analysis results and provide quality insights:

Query: {query}
Analysis Results: {kwargs.get('analysis_results', 'Results from primary analysis')}

Provide a comprehensive review covering:
1. Quality assessment of the methodology and results
2. Key insights and observations
3. Recommendations for improvement or additional analysis
4. Statistical validity considerations"""

            response = self.llm_client.generate(
                prompt=review_prompt,
                model=force_model,
                adaptive_timeout=True
            )
            
            if response and response.get('success'):
                result = response.get('response', 'No review generated')
            else:
                result = "Unable to generate review insights at this time."
            
            return {
                "result": result,
                "execution_time": time.time() - start_time,
                "model_used": force_model,
                "status": "success"
            }
        except Exception as e:
            logging.error(f"Error with forced model {force_model}: {e}")
            return friendly_error(f"Forced model error: {str(e)}", "Try a different model")
    
    def _resolve_filepath(self, filename: str) -> Optional[str]:
        """Resolve the full path to a data file."""
        base_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
        
        # Check uploads first
        uploads_path = os.path.join(base_data_dir, 'uploads', filename)
        if os.path.exists(uploads_path):
            return uploads_path
        
        # Check samples
        samples_path = os.path.join(base_data_dir, 'samples', filename)
        if os.path.exists(samples_path):
            return samples_path
        
        # Check samples subdirectories
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext == '.csv':
            csv_path = os.path.join(base_data_dir, 'samples', 'csv', filename)
            if os.path.exists(csv_path):
                return csv_path
        elif file_ext == '.json':
            json_path = os.path.join(base_data_dir, 'samples', 'json', filename)
            if os.path.exists(json_path):
                return json_path
        
        return None
    
    # =========================================================================
    # LEGACY METHOD - Kept for backward compatibility
    # =========================================================================
    
    def execute_with_review_protocol(self, task_description: str, primary_agent, review_agent=None, **kwargs) -> Dict[str, Any]:
        """
        Enhanced communication protocol between primary and review models.
        Kept for backward compatibility.
        """
        max_retries = kwargs.get('max_retries', 2)
        enable_review = kwargs.get('enable_review', False) and review_agent is not None
        
        conversation_rounds = 0
        max_conversation_rounds = 3
        
        try:
            self._initializer.ensure_initialized()
            
            Task = self._initializer.get_task_class()
            Crew = self._initializer.get_crew_class()
            Process = self._initializer.get_process_class()
            
            logging.info("🎯 Executing primary task...")
            
            primary_task = Task(
                description=task_description,
                agent=primary_agent,
                expected_output="Comprehensive analysis results"
            )
            
            primary_crew = Crew(
                agents=[primary_agent],
                tasks=[primary_task],
                process=Process.sequential,
                verbose=False
            )
            
            primary_result = primary_crew.kickoff()
            
            # Review validation (if enabled)
            review_feedback = None
            needs_retry = False
            
            if enable_review:
                logging.info("🔍 Starting review validation...")
                
                review_task = Task(
                    description=f"""
                    Review the following analysis results:
                    ORIGINAL TASK: {task_description}
                    PRIMARY RESULTS: {str(primary_result)}
                    
                    Provide quality assessment and recommendations.
                    """,
                    agent=review_agent,
                    expected_output="Structured review feedback"
                )
                
                review_crew = Crew(
                    agents=[review_agent],
                    tasks=[review_task],
                    process=Process.sequential,
                    verbose=False
                )
                
                review_result = review_crew.kickoff()
                review_feedback = str(review_result)
                
                quality_indicators = ["poor", "incomplete", "incorrect", "error"]
                needs_retry = any(ind in review_feedback.lower() for ind in quality_indicators)
            
            # Retry if needed
            retry_count = 0
            final_result = primary_result
            
            while needs_retry and retry_count < max_retries and conversation_rounds < max_conversation_rounds:
                conversation_rounds += 1
                retry_count += 1
                
                logging.info(f"🔄 Retry {conversation_rounds}/{max_conversation_rounds}...")
                
                retry_task = Task(
                    description=f"{task_description}\n\nFEEDBACK:\n{review_feedback}",
                    agent=primary_agent,
                    expected_output="Improved analysis results"
                )
                
                retry_crew = Crew(
                    agents=[primary_agent],
                    tasks=[retry_task],
                    process=Process.sequential,
                    verbose=False
                )
                
                final_result = retry_crew.kickoff()
                needs_retry = False
            
            return {
                "success": True,
                "result": str(final_result),
                "review_feedback": review_feedback,
                "retry_performed": retry_count > 0,
                "conversation_rounds": conversation_rounds
            }
            
        except Exception as e:
            logging.error(f"Communication protocol failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
