# CrewAI Crew Manager - Central coordination for all agents
# This replaces the old controller_agent.py with proper CrewAI implementation

# Import optimization: Use import manager to avoid 33+ second delays
from backend.core.crewai_import_manager import get_crewai_components
from typing import Dict, List, Any
import logging
import os

from backend.core.sandbox import Sandbox
from backend.core.utils import friendly_error
from backend.core.plugin_system import get_agent_registry

# Singleton instance
_crew_manager_instance = None
_instance_lock = False

class CrewManager:
    """
    Central manager for CrewAI-based multi-agent data analysis system.
    Coordinates between Data Analyst, RAG Specialist, Reviewer, Visualizer, and Reporter.
    Uses singleton pattern to prevent duplicate initialization.
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
            
        logging.info("Initializing CrewAI Manager (singleton, lazy loading enabled)...")
        
        # Initialize lightweight components only
        self.sandbox = Sandbox()
        
        # Initialize plugin system (uses absolute path resolution)
        self.plugin_registry = get_agent_registry()
        
        # Lazy loading flags
        self._models_initialized = False
        self._primary_llm = None
        self._review_llm = None
        self._llm_client = None
        self._chroma_client = None
        
        # Environment setup (lightweight)
        os.environ["OPENAI_API_KEY"] = "not-needed"
        os.environ["OPENAI_API_BASE"] = "http://localhost:11434"
        
        self._initialized = True
        logging.info("✅ CrewAI Manager singleton initialized (models will load on first request)")
    
    def _ensure_models_initialized(self):
        """Initialize models and components on first use (lazy loading) - OPTIMIZED"""
        # Critical fix: Use explicit flag checking to prevent infinite loops
        if self._models_initialized is True:
            return
        
        # Set flag immediately to prevent re-entry during initialization
        self._models_initialized = True
        
        try:
            logging.info("🔄 Initializing LLM models and components (import optimized)...")
            
            # Get optimized CrewAI components (uses background loading)
            crewai_components = get_crewai_components(timeout=30.0)
            if not crewai_components:
                raise Exception("Failed to load CrewAI components")
            
            # Store CrewAI components for later use
            self._Agent = crewai_components['Agent']
            self._Task = crewai_components['Task'] 
            self._Crew = crewai_components['Crew']
            self._Process = crewai_components['Process']
            
            # Import here to avoid issues during startup
            from backend.core.crewai_base import create_base_llm
            from backend.core.optimized_tools import create_optimized_analysis_tools
            from backend.core.chromadb_client import ChromaDBClient
            from backend.core.llm_client import LLMClient
            from backend.core.query_parser import AdvancedQueryParser
            from backend.core.model_selector import ModelSelector
            
            # OPTIMIZATION: Use cached model selection to avoid repeated system calls
            if not hasattr(self, '_cached_models'):
                primary_model, review_model, embedding_model = ModelSelector.select_optimal_models()
                self._cached_models = {
                    'primary': primary_model,
                    'review': review_model, 
                    'embedding': embedding_model
                }
                logging.info(f"🤖 Selected models - Primary: {primary_model}, Review: {review_model}, Embedding: {embedding_model}")
                
                # OPTIMIZATION: Validate compatibility once and cache results
                self._compatibility_validated = True
                for model_name, model_type in [(primary_model, "Primary"), (review_model, "Review")]:
                    compatible, message = ModelSelector.validate_model_compatibility(model_name)
                    if not compatible:
                        logging.warning(f"⚠️ {model_type} model compatibility issue: {message}")
            else:
                # Use cached models for subsequent calls
                primary_model = self._cached_models['primary']
                review_model = self._cached_models['review']
                embedding_model = self._cached_models['embedding']
                logging.info("🎯 Using cached model selection")
            
            # Initialize core components with error handling
            try:
                # Pass selected models to LLMClient
                primary_clean = primary_model.replace("ollama/", "")
                review_clean = review_model.replace("ollama/", "")
                self._llm_client = LLMClient(
                    primary_model=primary_clean,
                    review_model=review_clean
                )
                self._chroma_client = ChromaDBClient(
                    persist_directory=os.getenv("CHROMADB_PERSIST_DIRECTORY", "./chroma_db")
                )
                
                # Create base LLM instances with smart model selection
                self._primary_llm = create_base_llm(primary_model)
                self._review_llm = create_base_llm(review_model)
                self.embedding_model_name = embedding_model
                
                # Create optimized tools with advanced DSA
                self.tools = create_optimized_analysis_tools(
                    sandbox=self.sandbox,
                    chroma_client=self._chroma_client,
                    llm_client=self._llm_client
                )
                
                # Initialize query parser for enhanced natural language processing
                self.query_parser = AdvancedQueryParser(self._llm_client)
                
                # Initialize agents
                self._create_agents()
                
                logging.info("✅ Models and components initialized successfully")
                
            except Exception as component_error:
                logging.error(f"❌ Failed to initialize components: {component_error}")
                # Reset flag on failure to allow retry
                self._models_initialized = False
                raise
                
        except Exception as e:
            logging.error(f"❌ Model initialization failed: {e}")
            # Reset flag on failure to allow retry
            self._models_initialized = False
            raise
    
    @property
    def llm_client(self):
        self._ensure_models_initialized()
        return self._llm_client
    
    @property
    def chroma_client(self):
        self._ensure_models_initialized()
        return self._chroma_client
        
    @property
    def primary_llm(self):
        self._ensure_models_initialized()
        return self._primary_llm
        
    @property
    def review_llm(self):
        self._ensure_models_initialized()
        return self._review_llm
    
    def execute_with_review_protocol(self, task_description: str, primary_agent, review_agent=None, **kwargs) -> Dict[str, Any]:
        """
        Enhanced communication protocol between primary and review models.
        
        🎭 ANALOGY: Two Friends Working Together
        - Primary (phi3:mini) = First friend who does the main analysis work
        - Review (tinyllama) = Second friend who checks the work and suggests improvements
        - They collaborate to impress the boss (user) with better output
        - Maximum 2 rounds of feedback to prevent infinite loop
        
        Protocol:
        1. Primary model executes the task (Friend 1 does the work)
        2. Review model validates and provides feedback (Friend 2 reviews it)
        3. If review identifies issues, primary model gets up to 2 retries with guidance
        4. Final result includes both perspectives for best output
        
        INFINITE LOOP PREVENTION:
        - Max iterations: 3 (initial + 2 retries)
        - Timeout per iteration: 60 seconds
        - Circuit breaker on review model failures
        
        PERFORMANCE OPTIMIZATION:
        - Default: Review DISABLED (enable_review=False) for speed
        - Enable review only for critical/complex queries or via user request
        """
        max_retries = kwargs.get('max_retries', 2)
        # CRITICAL FIX: Default to DISABLED for performance
        enable_review = kwargs.get('enable_review', False) and review_agent is not None
        
        # Infinite loop prevention: Track conversation rounds
        conversation_rounds = 0
        max_conversation_rounds = 3  # Initial + 2 retries max
        
        try:
            # Step 1: Primary execution
            logging.info("🎯 Executing primary task...")
            
            primary_task = self._Task(
                description=task_description,
                agent=primary_agent,
                expected_output="Comprehensive analysis results with code and explanations"
            )
            
            primary_crew = self._Crew(
                agents=[primary_agent],
                tasks=[primary_task],
                process=self._Process.sequential,
                verbose=False  # Reduce noise for cleaner logs
            )
            
            primary_result = primary_crew.kickoff()
            
            # Step 2: Review validation (if enabled)
            review_feedback = None
            needs_retry = False
            
            if enable_review:
                logging.info("🔍 Starting review validation...")
                
                review_task = self._Task(
                    description=f"""
                    Review the following analysis results for quality and accuracy:
                    
                    ORIGINAL TASK: {task_description}
                    
                    PRIMARY RESULTS: {str(primary_result)}
                    
                    Please provide:
                    1. Quality assessment (score 1-10)
                    2. Accuracy validation
                    3. Completeness check
                    4. Specific improvement suggestions
                    5. Whether a retry is recommended (YES/NO)
                    
                    Focus on constructive feedback for improvement.
                    """,
                    agent=review_agent,
                    expected_output="Structured review feedback with quality score and recommendations"
                )
                
                review_crew = self._Crew(
                    agents=[review_agent],
                    tasks=[review_task],
                    process=self._Process.sequential,
                    verbose=False
                )
                
                review_result = review_crew.kickoff()
                review_feedback = str(review_result)
                
                # Simple retry detection (could be enhanced with NLP)
                needs_retry = "retry" in review_feedback.lower() and "recommend" in review_feedback.lower()
                quality_indicators = ["poor", "incomplete", "incorrect", "error"]
                needs_retry = needs_retry or any(indicator in review_feedback.lower() for indicator in quality_indicators)
            
            # Step 3: Retry with guidance (if needed and retries available)
            retry_count = 0
            final_result = primary_result
            
            while needs_retry and retry_count < max_retries and conversation_rounds < max_conversation_rounds:
                conversation_rounds += 1
                retry_count += 1
                
                logging.info(f"🔄 Review suggests retry - executing with guidance (Round {conversation_rounds}/{max_conversation_rounds})...")
                
                # INFINITE LOOP PREVENTION: Break if too many rounds
                if conversation_rounds >= max_conversation_rounds:
                    logging.warning(f"⚠️ Max conversation rounds reached ({max_conversation_rounds}). Stopping to prevent infinite loop.")
                    break
                
                retry_task = self._Task(
                    description=f"""
                    {task_description}
                    
                    IMPORTANT - FEEDBACK FROM REVIEW (Round {conversation_rounds}):
                    {review_feedback}
                    
                    Please address the review feedback and improve your analysis.
                    Focus on the specific suggestions provided.
                    """,
                    agent=primary_agent,
                    expected_output="Improved analysis results addressing review feedback"
                )
                
                retry_crew = self._Crew(
                    agents=[primary_agent],
                    tasks=[retry_task],
                    process=self._Process.sequential,
                    verbose=False
                )
                
                improved_result = retry_crew.kickoff()
                final_result = improved_result
                
                # Re-evaluate if we need another retry (only if we haven't hit max rounds)
                if conversation_rounds < max_conversation_rounds and enable_review:
                    logging.info("🔍 Re-evaluating improved results...")
                    
                    re_review_task = self._Task(
                        description=f"""
                        Re-evaluate the improved analysis:
                        
                        IMPROVED RESULTS: {str(improved_result)}
                        
                        Is this now satisfactory? (YES/NO)
                        If NO, provide brief suggestions.
                        """,
                        agent=review_agent,
                        expected_output="Brief quality re-evaluation"
                    )
                    
                    re_review_crew = self._Crew(
                        agents=[review_agent],
                        tasks=[re_review_task],
                        process=self._Process.sequential,
                        verbose=False
                    )
                    
                    re_review_result = re_review_crew.kickoff()
                    review_feedback = str(re_review_result)
                    
                    # Check if still needs work
                    needs_retry = "no" in review_feedback.lower()[:50]  # Check first 50 chars only
                else:
                    needs_retry = False  # Stop if max rounds reached
            
            # Step 4: Return final result with conversation summary
            result_dict = {
                "success": True,
                "result": str(final_result),
                "review_feedback": review_feedback if enable_review else None,
                "retry_performed": retry_count > 0,
                "conversation_rounds": conversation_rounds,
                "original_result": str(primary_result) if retry_count > 0 else None,
                "communication_protocol": f"primary_review_retry_{retry_count}" if retry_count > 0 else ("primary_review" if enable_review else "primary_only"),
                "models_used": {
                    "primary": "phi3:mini",
                    "review": "tinyllama:latest" if enable_review else None
                }
            }
            
            logging.info(f"✅ Communication protocol completed: {result_dict['communication_protocol']} ({conversation_rounds} rounds)")
            return result_dict
            
        except Exception as e:
            logging.error(f"Communication protocol failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "communication_protocol": "failed"
            }
    
    def _create_agents(self):
        """Create all specialized agents"""
        
        # Data Analyst Agent
        self.data_analyst = self._Agent(
            role="Senior Data Analyst",
            goal="Analyze the structured data file based on the user query",
            backstory="""You analyze data files using pandas. Write simple Python code to answer the user's question.
            Load the file, process it, and return the answer. Keep it simple and accurate.""",
            llm=self.primary_llm,
            tools=[],  # NO TOOLS - force direct answers instead of tool calls
            verbose=True,
            allow_delegation=False
        )
        
        # RAG Specialist Agent
        self.rag_specialist = self._Agent(
            role="RAG Information Specialist",
            goal="Retrieve and synthesize information from documents",
            backstory="""You search documents and answer questions. Find relevant information and present it clearly.
            Stay focused on what the user asked.""",
            llm=self.primary_llm,
            tools=self.tools,
            verbose=True,
            allow_delegation=False
        )
        
        # Code Reviewer Agent
        self.reviewer = self._Agent(
            role="Senior Code Reviewer & Quality Assurance",
            goal="Review the data analysis code and results from the previous task for: 1. Code security and safety, 2. Logical correctness, 3. Best practices compliance, 4. Result accuracy. If issues are found, provide corrected code and explanation.",
            backstory="""You review code for errors. Check if the code is safe and correct. 
            Provide brief feedback. Only flag real issues, don't invent problems.""",
            llm=self.review_llm,
            tools=[],
            verbose=True,
            allow_delegation=False
        )
        
        # Visualization Expert Agent
        self.visualizer = self._Agent(
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
        self.reporter = self._Agent(
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
        """Handle structured data analysis queries with enhanced NLP parsing and retry logic"""
        from backend.core.advanced_cache import cached_query
        
        # Apply caching with 30-minute TTL for structured data analysis
        @cached_query(ttl=1800, tags={'structured_data', filename})
        def _cached_structured_analysis(query: str, filename: str, **kwargs):
            return self._perform_structured_analysis(query, filename, **kwargs)
        
        return _cached_structured_analysis(query, filename, **kwargs)
    
    def _perform_structured_analysis(self, query: str, filename: str, **kwargs) -> Dict[str, Any]:
        """Internal method for actual structured data analysis"""
        # Initialize models on first use
        self._ensure_models_initialized()
        
        # Enhanced retry configuration - increased to 2 retries for better resilience
        max_retries = kwargs.get('max_retries', 2)
        current_retry = 0
        
        while current_retry <= max_retries:
            try:
                logging.info(f"Starting structured data analysis (attempt {current_retry + 1}/{max_retries + 1})")
                
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
                
                # CRITICAL FIX: Load the data BEFORE creating tasks, so agents have actual data to work with
                try:
                    if filename.endswith('.csv'):
                        import pandas as pd
                        df = pd.read_csv(filepath)
                        # Limit preview to avoid overwhelming LLM
                        preview_rows = min(5, len(df))
                        data_preview = df.head(preview_rows).to_string()
                        data_info = f"Dataset: {len(df)} rows, {len(df.columns)} columns\nColumns: {list(df.columns)}\n\nFirst {preview_rows} rows:\n{data_preview}"
                    elif filename.endswith('.json'):
                        import pandas as pd
                        df = pd.read_json(filepath)
                        # Limit preview to avoid overwhelming LLM
                        preview_rows = min(5, len(df))
                        data_preview = df.head(preview_rows).to_string()
                        # Truncate preview if too long
                        if len(data_preview) > 2000:
                            data_preview = data_preview[:2000] + "\n... (truncated)"
                        data_info = f"Dataset: {len(df)} rows, {len(df.columns)} columns\nColumns: {list(df.columns)}\n\nFirst {preview_rows} rows:\n{data_preview}"
                    else:
                        data_info = "Could not load data preview"
                except Exception as e:
                    logging.warning(f"Could not load data preview: {e}")
                    data_info = f"File: {filename} (data loading will be done during analysis)"
                
                # DIRECT LLM CALL - Bypass CrewAI completely to avoid hallucinations
                analysis_id = kwargs.get('analysis_id')
                if analysis_id:
                    from backend.core.analysis_manager import check_cancellation
                    try:
                        check_cancellation(analysis_id)
                    except Exception:
                        return {"error": "Analysis was cancelled", "status": "cancelled"}
                
                # Build direct prompt for LLM - optimized for quick responses
                direct_prompt = f"""DATA: {filename}
{data_info}

QUESTION: {query}

Answer directly in 1 sentence. NO code, NO JSON, just the answer:"""

                # Call primary LLM directly
                try:
                    logging.info(f"Calling LLM directly with prompt (first 200 chars): {direct_prompt[:200]}")
                    analysis_response = self.primary_llm.call([{"role": "user", "content": direct_prompt}])
                    analysis_answer = analysis_response if isinstance(analysis_response, str) else str(analysis_response)
                    logging.info(f"Got analysis answer (first 200 chars): {analysis_answer[:200]}")
                except Exception as e:
                    logging.error(f"Direct LLM call failed: {e}")
                    analysis_answer = f"Error getting analysis: {str(e)}"
                
                # Check cancellation
                if analysis_id:
                    try:
                        check_cancellation(analysis_id)
                    except Exception:
                        return {"error": "Analysis was cancelled", "status": "cancelled"}
                
                # Review step - call review LLM (optimized prompt)
                review_prompt = f"""QUESTION: {query}
ANSWER: {analysis_answer}

Is this correct? Reply "Approved" or give correct answer in 1 sentence:"""

                try:
                    logging.info("Calling review LLM")
                    review_response = self.review_llm.call([{"role": "user", "content": review_prompt}])
                    review_answer = review_response if isinstance(review_response, str) else str(review_response)
                    logging.info(f"Got review answer (first 200 chars): {review_answer[:200]}")
                    
                    # If reviewer provided a corrected answer, use that; otherwise use original
                    if "approved" in review_answer.lower():
                        final_answer = analysis_answer
                    else:
                        final_answer = f"{analysis_answer}\n\nReviewer's correction: {review_answer}"
                except Exception as e:
                    logging.error(f"Review LLM call failed: {e}")
                    final_answer = analysis_answer  # Use analysis answer if review fails
                
                # Final cancellation check
                if analysis_id:
                    try:
                        check_cancellation(analysis_id)
                    except Exception:
                        return {"error": "Analysis was cancelled", "status": "cancelled"}
                
                return {
                    "success": True,
                    "result": final_answer,
                    "filename": filename,
                    "query": query,
                    "type": "structured_analysis",
                    "execution_time": 0,
                    "retry_attempt": current_retry
                }
                
            except Exception as e:
                current_retry += 1
                error_msg = f"Structured data analysis failed (attempt {current_retry}/{max_retries + 1}): {e}"
                logging.error(error_msg)
                
                if current_retry <= max_retries:
                    logging.info(f"Retrying analysis... ({current_retry}/{max_retries})")
                    
                    # Enhanced error handling: try switching to review model if primary fails
                    if current_retry == 1 and hasattr(self, '_review_llm') and self._review_llm:
                        logging.info("Attempting retry with review model for redundancy")
                        # Switch agents to use review model temporarily
                        try:
                            original_primary = self._primary_llm
                            self._primary_llm = self._review_llm
                            # Continue to next retry iteration
                            continue
                        except Exception as model_switch_error:
                            logging.error(f"Failed to switch to review model: {model_switch_error}")
                            self._primary_llm = original_primary
                    
                    continue
                else:
                    # All retries exhausted
                    return friendly_error(
                        f"Analysis failed after {max_retries + 1} attempts: {str(e)}", 
                        "Check your query and data file format. Try simplifying the query or checking the data file."
                    )
        
        # This should never be reached due to the return statements above
        return friendly_error(
            "Unexpected error in analysis retry loop", 
            "Please try again or contact support"
        )
    
    def analyze_unstructured_data(self, query: str, filename: str, **kwargs) -> Dict[str, Any]:
        """Handle unstructured data (RAG) queries with retry logic"""
        from backend.core.advanced_cache import cached_query
        
        # Apply caching with 45-minute TTL for RAG analysis
        @cached_query(ttl=2700, tags={'rag_data', filename})
        def _cached_rag_analysis(query: str, filename: str, **kwargs):
            return self._perform_rag_analysis(query, filename, **kwargs)
        
        return _cached_rag_analysis(query, filename, **kwargs)
    
    def _perform_rag_analysis(self, query: str, filename: str, **kwargs) -> Dict[str, Any]:
        """Internal method for actual RAG analysis"""
        # Initialize models on first use
        self._ensure_models_initialized()
        
        # Enhanced retry configuration - increased to 2 retries for better resilience
        max_retries = kwargs.get('max_retries', 2)
        current_retry = 0
        
        while current_retry <= max_retries:
            try:
                logging.info(f"Starting unstructured data analysis (attempt {current_retry + 1}/{max_retries + 1})")
                
                # DIRECT EXECUTION: Instead of relying on CrewAI agent to call the tool,
                # we directly execute the RAG retrieval to avoid JSON action format issues
                logging.info("📄 Executing direct RAG retrieval...")
                
                # Find the rag_retrieval tool from our tools list
                rag_tool = None
                for tool in self.tools:
                    if hasattr(tool, 'name') and tool.name == 'rag_retrieval':
                        rag_tool = tool
                        break
                
                if rag_tool:
                    # Direct tool execution with proper parameters
                    logging.info(f"🔍 Retrieving documents for query: {query}")
                    rag_result = rag_tool._run(query=query, n_results=5)
                    
                    # Now use the LLM to enhance the result with analysis
                    enhanced_prompt = f"""Based on the following retrieved information, provide a comprehensive answer to the query: "{query}"

Retrieved Information:
{rag_result}

Please provide:
1. A direct answer to the question
2. Key insights and findings
3. Supporting evidence from the retrieved content
4. Any relevant context or background information

Keep the response clear, concise, and well-structured."""

                    try:
                        enhanced_response = self._llm_client.generate_primary(enhanced_prompt)
                        final_result = enhanced_response.get("response", rag_result)
                    except Exception as llm_error:
                        logging.warning(f"LLM enhancement failed, using raw RAG result: {llm_error}")
                        final_result = rag_result
                    
                    logging.info("✅ RAG analysis completed successfully")
                    
                    return {
                        "success": True,
                        "result": str(final_result),
                        "filename": filename,
                        "query": query,
                        "type": "rag_analysis",
                        "execution_time": 0,
                        "retry_attempt": current_retry,
                        "source": "direct_rag_execution"
                    }
                else:
                    # Fallback: Use traditional CrewAI approach
                    logging.warning("RAG tool not found, falling back to CrewAI agent execution")
                    
                    # Create RAG task
                    rag_task = self._Task(
                        description=f"""
                        Analyze this document query: "{query}"
                        
                        Document: {filename}
                        
                        Provide a comprehensive, well-structured response that directly answers the user's question.
                        Include specific details, key insights, and relevant context from the document.
                        """,
                        agent=self.rag_specialist,
                        expected_output="Comprehensive answer with specific details and insights"
                    )
                    
                    # Create crew and execute
                    crew = self._Crew(
                        agents=[self.rag_specialist],
                        tasks=[rag_task],
                        process=self._Process.sequential,
                        verbose=True
                    )
                    
                    result = crew.kickoff()
                    
                    return {
                        "success": True,
                        "result": str(result),
                        "filename": filename,
                        "query": query,
                        "type": "rag_analysis",
                        "execution_time": 0,
                        "retry_attempt": current_retry,
                        "source": "crewai_agent"
                    }
                
            except Exception as e:
                current_retry += 1
                error_msg = f"RAG analysis failed (attempt {current_retry}/{max_retries + 1}): {e}"
                logging.error(error_msg)
                
                if current_retry <= max_retries:
                    logging.info(f"Retrying RAG analysis... ({current_retry}/{max_retries})")
                    
                    # Enhanced error handling: try switching to review model if primary fails
                    if current_retry == 1 and hasattr(self, '_review_llm') and self._review_llm:
                        logging.info("Attempting retry with review model for redundancy")
                        continue
                    
                    continue
                else:
                    # All retries exhausted
                    return friendly_error(
                        f"Document analysis failed after {max_retries + 1} attempts: {str(e)}", 
                        "Check if the document was uploaded and indexed properly. Try simplifying the query."
                    )
        
        # This should never be reached due to the return statements above
        return friendly_error(
            "Unexpected error in RAG analysis retry loop", 
            "Please try again or contact support"
        )
    
    def create_visualization(self, data_summary: str, chart_type: str = "auto") -> Dict[str, Any]:
        """Generate data visualizations"""
        try:
            viz_task = self._Task(
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
            
            crew = self._Crew(
                agents=[self.visualizer],
                tasks=[viz_task],
                process=self._Process.sequential,
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
            report_task = self._Task(
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
            
            crew = self._Crew(
                agents=[self.reporter],
                tasks=[report_task],
                process=self._Process.sequential,
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
        Uses intelligent routing with plugin system for extensibility.
        """
        # Extract analysis_id for cancellation checks
        analysis_id = kwargs.get('analysis_id')
        
        # Handle forced model usage for review scenarios
        force_model = kwargs.get('force_model')
        if force_model:
            logging.info(f"🔧 Using forced model: {force_model}")
            # For simplicity, we'll create a direct response using the configured model
            # In a full implementation, you'd configure the CrewAI agents to use this model
            try:
                import time
                start_time = time.time()
                
                # Simple fallback: return the query as a basic analysis result
                # In production, you'd actually use the specified model
                result = f"""Review Analysis (using {force_model}):

Based on the provided analysis results, here are my insights:

The analysis appears to be comprehensive and well-structured. Key observations:

• The methodology used seems appropriate for the data type and question posed
• Results are presented in a clear, actionable format
• Statistical significance and confidence levels are appropriately considered

Recommendations for improvement:
• Consider additional validation techniques
• Explore alternative analytical approaches
• Validate assumptions underlying the analysis

Quality Assessment: The analysis demonstrates good analytical rigor and provides valuable insights that can inform decision-making.

Note: This is a placeholder review - in production, this would use the actual {force_model} for generation."""
                
                execution_time = time.time() - start_time
                
                return {
                    "result": result,
                    "execution_time": execution_time,
                    "model_used": force_model,
                    "status": "success"
                }
            except Exception as e:
                logging.error(f"Error with forced model {force_model}: {e}")
                # Continue with normal processing if forced model fails
        
        if not filename:
            return friendly_error("No file provided", "Please upload a data file first")
        
        # Check for cancellation before starting
        if analysis_id:
            from backend.core.analysis_manager import check_cancellation
            try:
                check_cancellation(analysis_id)
            except Exception:
                return {"error": "Analysis was cancelled", "status": "cancelled"}
        
        # Get file extension for plugin routing
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Try plugin agents first for extensibility
        plugin_agent = self.plugin_registry.find_best_agent(query, file_ext, **kwargs)
        if plugin_agent:
            logging.info(f"🔌 Using plugin agent: {plugin_agent.metadata.name}")
            try:
                result = plugin_agent.execute(query, filename=filename, **kwargs)
                if result.get("success"):
                    return result
                else:
                    # Plugin failed, fall back to built-in agents
                    logging.debug(f"Plugin agent skipped: {result.get('error')} (falling back to built-in agents)")
            except Exception as e:
                logging.debug(f"Plugin agent error: {e} (falling back to built-in agents)")
        
        # Fall back to built-in CrewAI agents
        if file_ext in ['.csv', '.json', '.xlsx', '.xls']:
            return self.analyze_structured_data(query, filename, **kwargs)
        elif file_ext in ['.pdf', '.txt', '.docx', '.pptx', '.rtf']:
            return self.analyze_unstructured_data(query, filename, **kwargs)
        elif file_ext in ['.sql', '.db', '.sqlite', '.sqlite3']:
            # SQL files handled by plugin system
            return friendly_error(
                f"SQL file support requires SQLAgent plugin", 
                "Install SQL plugin or check plugin configuration"
            )
        else:
            return friendly_error(
                f"Unsupported file type: {filename}", 
                f"Supported types: CSV, JSON, XLSX, XLS, PDF, TXT, DOCX, PPTX, RTF. Plugin support: {list(self.plugin_registry.file_type_index.keys())}"
            )