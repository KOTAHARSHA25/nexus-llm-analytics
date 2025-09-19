# CrewAI Crew Manager - Central coordination for all agents
# This replaces the old controller_agent.py with proper CrewAI implementation

from crewai import Agent, Task, Crew
from crewai.process import Process
from typing import Dict, List, Any
import logging
import os

from backend.core.crewai_base import create_base_llm, create_analysis_tools
from backend.core.sandbox import Sandbox
from backend.core.chromadb_client import ChromaDBClient
from backend.core.llm_client import LLMClient
from backend.core.utils import friendly_error
from backend.core.query_parser import AdvancedQueryParser
from backend.core.model_selector import ModelSelector

class CrewManager:
    """
    Central manager for CrewAI-based multi-agent data analysis system.
    Coordinates between Data Analyst, RAG Specialist, Reviewer, Visualizer, and Reporter.
    """
    
    def __init__(self):
        logging.info("Initializing CrewAI Manager...")
        
        # Intelligent model selection based on system resources
        primary_model, review_model, embedding_model = ModelSelector.select_optimal_models()
        logging.info(f"🤖 Selected models - Primary: {primary_model}, Review: {review_model}, Embedding: {embedding_model}")
        
        # Validate model compatibility
        for model_name, model_type in [(primary_model, "Primary"), (review_model, "Review"), (embedding_model, "Embedding")]:
            compatible, message = ModelSelector.validate_model_compatibility(model_name)
            if not compatible:
                logging.warning(f"⚠️ {model_type} model compatibility issue: {message}")
        
        # Set environment variables for CrewAI/LiteLLM - Use ollama provider format
        os.environ["OPENAI_API_KEY"] = "not-needed"
        os.environ["OPENAI_API_BASE"] = "http://localhost:11434"
        # Don't set OPENAI_MODEL_NAME - let CrewAI handle it
        
        # Initialize core components
        self.llm_client = LLMClient()
        self.sandbox = Sandbox()
        self.chroma_client = ChromaDBClient(
            persist_directory=os.getenv("CHROMADB_PERSIST_DIRECTORY", "./chroma_db")
        )
        
        # Create base LLM instances with smart model selection
        self.primary_llm = create_base_llm(primary_model)
        self.review_llm = create_base_llm(review_model)
        self.embedding_model_name = embedding_model
        
        # Create tools
        self.tools = create_analysis_tools(
            sandbox=self.sandbox,
            chroma_client=self.chroma_client,
            llm_client=self.llm_client
        )
        
        # Initialize query parser for enhanced natural language processing
        self.query_parser = AdvancedQueryParser(self.llm_client)
        
        # Initialize agents
        self._create_agents()
        
        logging.info("CrewAI Manager initialized successfully")
    
    def _create_agents(self):
        """Create all specialized agents"""
        
        # Data Analyst Agent
        self.data_analyst = Agent(
            role="Senior Data Analyst",
            goal="Analyze structured data and generate insights using statistical methods and data manipulation",
            backstory="""You are an expert data analyst with deep knowledge of pandas, numpy, and statistical analysis. 
            You excel at understanding data structures, performing complex queries, and generating meaningful insights from datasets.
            You always write clean, efficient Python code and explain your analysis clearly.""",
            llm=self.primary_llm,
            tools=self.tools,
            verbose=True,
            allow_delegation=False
        )
        
        # RAG Specialist Agent
        self.rag_specialist = Agent(
            role="RAG Information Specialist",
            goal="Retrieve and synthesize information from unstructured documents using vector similarity search",
            backstory="""You are a specialist in information retrieval and document analysis. You excel at finding relevant 
            information from large document collections and synthesizing it into coherent, accurate responses. You understand 
            context and can connect information across multiple sources.""",
            llm=self.primary_llm,
            tools=self.tools,
            verbose=True,
            allow_delegation=False
        )
        
        # Code Reviewer Agent
        self.reviewer = Agent(
            role="Senior Code Reviewer & Quality Assurance",
            goal="Review generated code for security, correctness, and best practices",
            backstory="""You are a meticulous code reviewer with expertise in security, performance, and code quality. 
            You catch potential vulnerabilities, logic errors, and suggest improvements. You ensure all code follows 
            best practices and is safe to execute.""",
            llm=self.review_llm,
            tools=[],
            verbose=True,
            allow_delegation=False
        )
        
        # Visualization Expert Agent
        self.visualizer = Agent(
            role="Data Visualization Expert",
            goal="Create compelling and informative data visualizations using Plotly",
            backstory="""You are a data visualization expert who creates beautiful, informative charts and graphs. 
            You understand which visualization types work best for different data patterns and always create 
            interactive, professional-quality visualizations using Plotly.""",
            llm=self.primary_llm,
            tools=self.tools,
            verbose=True,
            allow_delegation=False
        )
        
        # Report Writer Agent
        self.reporter = Agent(
            role="Technical Report Writer",
            goal="Compile analysis results into professional, comprehensive reports",
            backstory="""You are an expert technical writer who creates clear, comprehensive reports from data analysis results. 
            You structure information logically, highlight key insights, and present findings in a way that's accessible 
            to both technical and non-technical audiences.""",
            llm=self.primary_llm,
            tools=[],
            verbose=True,
            allow_delegation=False
        )
    
    def analyze_structured_data(self, query: str, filename: str, **kwargs) -> Dict[str, Any]:
        """Handle structured data analysis queries with enhanced NLP parsing"""
        try:
            # Load data to get column information for better parsing
            import pandas as pd
            # Look for files in both uploads and samples directories
            base_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
            filepath = None
            
            # Check uploads directory first
            uploads_path = os.path.join(base_data_dir, 'uploads', filename)
            if os.path.exists(uploads_path):
                filepath = uploads_path
            else:
                # Check samples directory
                samples_path = os.path.join(base_data_dir, 'samples', filename)
                if os.path.exists(samples_path):
                    filepath = samples_path
                else:
                    # Fallback to direct path in data directory
                    filepath = os.path.join(base_data_dir, filename)
            
            try:
                if filename.endswith('.csv'):
                    sample_data = pd.read_csv(filepath)
                elif filename.endswith('.json'):
                    sample_data = pd.read_json(filepath)
                else:
                    sample_data = None
                
                # Get available columns and data sample for query parsing
                available_columns = list(sample_data.columns) if sample_data is not None else []
                data_sample = sample_data.head(3).to_dict() if sample_data is not None else None
                
            except Exception as e:
                logging.warning(f"Could not load data sample for query parsing: {e}")
                available_columns = []
                data_sample = None
            
            # Parse the natural language query
            parsed_query = self.query_parser.parse_query(
                query, 
                available_columns=available_columns,
                data_sample=data_sample
            )
            
            logging.info(f"Parsed query intent: {parsed_query.intent.value}, confidence: {parsed_query.confidence}")
            
            # Generate analysis plan
            analysis_plan = self.query_parser.generate_analysis_plan(parsed_query)
            
            # Create enhanced analysis task with parsed information
            analysis_task = Task(
                description=f"""
                Analyze the structured data file '{filename}' based on this query: "{query}"
                
                PARSED QUERY INFORMATION:
                - Intent: {parsed_query.intent.value}
                - Relevant columns: {parsed_query.columns}
                - Filter conditions: {parsed_query.conditions}
                - Aggregation: {parsed_query.aggregation}
                - Chart type: {parsed_query.chart_type}
                - Confidence: {parsed_query.confidence:.2f}
                
                ANALYSIS PLAN:
                - Steps: {analysis_plan['steps']}
                - Estimated complexity: {analysis_plan['complexity']}
                
                EXECUTION INSTRUCTIONS:
                1. Load the data file: {filename}
                2. Focus on columns: {parsed_query.columns or 'all columns'}
                3. Apply any filter conditions: {parsed_query.conditions}
                4. Perform the {parsed_query.intent.value} operation
                5. Generate appropriate Python code
                6. Execute safely and interpret results
                
                Additional parameters: {kwargs}
                
                Provide detailed results with code, explanation, and insights.
                """,
                agent=self.data_analyst,
                expected_output="Comprehensive analysis results with code, explanation, and key findings"
            )
            
            # Create review task
            review_task = Task(
                description="""
                Review the data analysis code and results from the previous task for:
                1. Code security and safety
                2. Logical correctness
                3. Best practices compliance
                4. Result accuracy
                
                If issues are found, provide corrected code and explanation.
                """,
                agent=self.reviewer,
                expected_output="Code review with approval or corrections",
                context=[analysis_task]
            )
            
            # Create crew and execute
            crew = Crew(
                agents=[self.data_analyst, self.reviewer],
                tasks=[analysis_task, review_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "success": True,
                "result": str(result),
                "filename": filename,
                "query": query,
                "type": "structured_analysis"
            }
            
        except Exception as e:
            logging.error(f"Structured data analysis failed: {e}")
            return friendly_error(
                f"Analysis failed: {str(e)}", 
                "Check your query and data file format"
            )
    
    def analyze_unstructured_data(self, query: str, filename: str, **kwargs) -> Dict[str, Any]:
        """Handle unstructured data (RAG) queries"""
        try:
            # Create RAG task
            rag_task = Task(
                description=f"""
                Use RAG (Retrieval-Augmented Generation) to answer this query: "{query}"
                
                Steps:
                1. Search for relevant information in the document collection
                2. Retrieve the most relevant text chunks
                3. Synthesize the information to answer the user's query
                4. Provide sources and context for your response
                
                Target document: {filename}
                Additional parameters: {kwargs}
                """,
                agent=self.rag_specialist,
                expected_output="Comprehensive answer with source attribution"
            )
            
            # Create crew and execute
            crew = Crew(
                agents=[self.rag_specialist],
                tasks=[rag_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "success": True,
                "result": str(result),
                "filename": filename,
                "query": query,
                "type": "rag_analysis"
            }
            
        except Exception as e:
            logging.error(f"RAG analysis failed: {e}")
            return friendly_error(
                f"Document analysis failed: {str(e)}", 
                "Check if the document was uploaded and indexed properly"
            )
    
    def create_visualization(self, data_summary: str, chart_type: str = "auto") -> Dict[str, Any]:
        """Generate data visualizations"""
        try:
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
            return friendly_error(
                f"Visualization failed: {str(e)}", 
                "Check the data format and chart type"
            )
    
    def generate_report(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive reports"""
        try:
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
                
                Format the report in a clear, professional structure suitable for both 
                technical and business audiences.
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
            return friendly_error(
                f"Report generation failed: {str(e)}", 
                "Check the analysis results format"
            )
    
    def handle_query(self, query: str, filename: str = None, **kwargs) -> Dict[str, Any]:
        """
        Main entry point for handling all types of queries.
        Routes to appropriate analysis method based on file type and query.
        """
        if not filename:
            return friendly_error("No file provided", "Please upload a data file first")
        
        # Determine analysis type based on file extension
        if filename.lower().endswith(('.csv', '.json')):
            return self.analyze_structured_data(query, filename, **kwargs)
        elif filename.lower().endswith(('.pdf', '.txt')):
            return self.analyze_unstructured_data(query, filename, **kwargs)
        else:
            return friendly_error(
                f"Unsupported file type: {filename}", 
                "Please use CSV, JSON, PDF, or TXT files"
            )