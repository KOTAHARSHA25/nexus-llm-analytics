# Enhanced SQL Agent Plugin
# Comprehensive SQL database analysis and query generation system
# Supports SQLite, PostgreSQL, MySQL, and SQL Server connections

import sys
import os
import re
import logging
import sqlite3
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import tempfile

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

try:
    from backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability
    from backend.core.llm_client import LLMClient
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback for testing execution where backend.core might not be in path
    try:
        from src.backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability
        from src.backend.core.llm_client import LLMClient
    except ImportError:
        pass

# Optional imports for enhanced functionality
try:
    import sqlalchemy
    from sqlalchemy import create_engine, MetaData, inspect, text
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    sqlalchemy = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

class SQLAgent(BasePluginAgent):
    """
    Enhanced SQL Database Analysis Agent
    
    Capabilities:
    - Advanced SQL query generation from natural language
    - "Chat with your Data": Load CSV/Excel into in-memory SQL for querying
    - Comprehensive database schema analysis
    - Intelligent query optimization suggestions
    - Data relationship discovery and mapping
    """
    
    def get_metadata(self) -> AgentMetadata:
        """Define agent metadata and capabilities"""
        return AgentMetadata(
            name="EnhancedSQLAgent",
            version="2.0.0", 
            description="Comprehensive SQL database analysis, query generation, and optimization agent with 'Talk to Data' support",
            author="Nexus LLM Analytics Team",
            capabilities=[
                AgentCapability.SQL_QUERYING,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.VISUALIZATION,
                AgentCapability.REPORTING
            ],
            file_types=[".sql", ".db", ".sqlite", ".sqlite3", ".mysql", ".psql", ".csv", ".xlsx", ".json"],
            dependencies=["sqlite3", "sqlalchemy", "pandas"],
            min_ram_mb=512,
            max_timeout_seconds=300,
            priority=85 
        )
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the enhanced SQL agent"""
        try:
            # Database connections - Init these FIRST so they exist even if LLM fails
            self.connections = {}
            self.engines = {} 
            self.connections = {}
            self.active_db_type = "sqlite"
            self.active_connection_id = "default"

            # Configuration
            self.database_url = self.config.get("database_url", "sqlite:///:memory:") 
            self.query_timeout = self.config.get("query_timeout", 60)
            self.max_results = self.config.get("max_results", 10000)
            
            # Initialize LLM Client (Optional - for graceful degradation)
            try:
                self.llm_client = LLMClient()
                logging.debug("SQLAgent: LLM Client initialized")
            except Exception as e:
                logging.warning(f"SQLAgent: LLM Client failed to init ({e}). SQL generation will be disabled.")
                self.llm_client = None
            
            # Initialize default in-memory SQLite connection for "Chat with Data"
            self._init_memory_connection()
            
            # Initialize External Connection if configured
            if self.database_url and "sqlite:///:memory:" not in self.database_url:
                if HAS_SQLALCHEMY:
                    try:
                        engine = create_engine(self.database_url)
                        self.engines['external'] = engine
                        self.connections['external'] = engine.connect()
                        self.active_connection_id = "external" # Default to external if configured
                        logging.info(f"SQLAgent: Connected to external DB: {self.database_url}")
                    except Exception as e:
                        logging.error(f"SQLAgent: Failed to connect to external DB: {e}")
                        # Fallback to memory
                        self.active_connection_id = "default"
                else:
                    logging.warning("SQLAgent: External DB configured but SQLAlchemy missing.")
            
            self.initialized = True
            logging.debug(f"SQLAgent initialized: timeout={self.query_timeout}s")
            
            return True
            
        except Exception as e:
            logging.error(f"SQLAgent initialization failed: {e}")
            return False
            
    def _init_memory_connection(self):
        """Initialize default in-memory SQLite connection"""
        try:
            if HAS_SQLALCHEMY:
                engine = create_engine("sqlite:///:memory:", echo=False)
                self.engines['default'] = engine
                self.connections['default'] = engine.connect()
                logging.debug("In-memory SQLite engine initialized")
            else:
                self.connections['default'] = sqlite3.connect(":memory:", check_same_thread=False)
                logging.debug("In-memory SQLite connection initialized")
        except Exception as e:
            logging.warning(f"Default connection setup failed: {e}")

    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs) -> float:
        """
        Determine if this agent can handle the query.
        Handles explicit SQL queries AND 'Ask data' queries if CSV/Excel provided.
        """
        if not self.initialized:
            return 0.0
            
        confidence = 0.0
        query_lower = query.lower()
        
        # Explicit SQL Intent
        explicit_sql = ["sql", "database", "query", "select *", "join tables"]
        if any(x in query_lower for x in explicit_sql):
            confidence += 0.4

        # File Handling
        if file_type:
            ft = file_type.lower()
            if not ft.startswith('.'): ft = '.' + ft
            
            if ft in [".sql", ".db", ".sqlite"]:
                 confidence += 0.5
            elif ft in [".csv", ".xlsx", ".json"]:
                 # If CSV/Excel, only handle if query looks like a specific data question
                 # that requires SQL-like aggregation (Count, Sum, Group By)
                 agg_terms = ["count", "sum", "average", "group by", "how many", "total"]
                 if any(x in query_lower for x in agg_terms):
                     confidence += 0.2
        
        return min(confidence, 0.9) # Cap at 0.9 to let specialized agents (Financial) take precedence if very specific

    def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Execute SQL analysis"""
        try:
            if not self.initialized and not self.initialize():
                 return {"success": False, "error": "SQL Agent failed to initialize", "agent": "SQLAgent"}

            # 1. Load Data into SQL if provided (and not already a SQL DB)
            table_name = "analyzed_data"
            if data is not None and HAS_PANDAS and isinstance(data, pd.DataFrame):
                self._load_dataframe_to_sql(data, table_name)
            
            filename = kwargs.get('filename', 'data')
            
            # 2. Parse Intent
            intent = self._parse_query_intent(query)
            
            if intent == "schema":
                return self._analyze_schema(table_name)
            elif intent == "optimize":
                return self._optimize_query(query)
            else:
                # Default: Generate and Execute SQL
                # Step 1: Generate SQL from Natural Language
                # If query is already SQL, skip generation
                if self._is_raw_sql(query):
                    sql_query = query
                else:
                    generation_result = self._generate_sql_query(query, table_name, data)
                    if not generation_result["success"]:
                        return generation_result
                    sql_query = generation_result["result"]["generated_sql"]

                # Step 2: Execute SQL
                return self._execute_sql_query(sql_query)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"SQL agent execution failed: {str(e)}",
                "agent": "SQLAgent"
            }
    
    def _load_dataframe_to_sql(self, df: pd.DataFrame, table_name: str):
        """Load a DataFrame into the in-memory SQLite database"""
        try:
            # ALWAYS load uploaded data into the memory connection, to avoid polluting external DBs
            conn = self.connections['default'] # Force 'default' (memory)
            
            if HAS_SQLALCHEMY:
                df.to_sql(table_name, con=self.engines['default'], if_exists='replace', index=False)
            else:
                df.to_sql(table_name, con=conn, if_exists='replace', index=False)
            
            # Switch active context to memory since we are working with uploaded data
            self.active_connection_id = 'default' 
            logging.info(f"Loaded DataFrame into SQL table '{table_name}' (Memory Context)")
        except Exception as e:
            logging.error(f"Failed to load DF to SQL: {e}")
            raise

    def _is_raw_sql(self, query: str) -> bool:
        """Check if query is likely raw SQL"""
        q = query.strip().upper()
        return q.startswith("SELECT") or q.startswith("WITH") or q.startswith("PRAGMA")

    def _parse_query_intent(self, query: str) -> str:
        q = query.lower()
        if "schema" in q or "describe table" in q: return "schema"
        if "optimize" in q or "explain query" in q: return "optimize"
        return "query"

    def _analyze_schema(self, table_name: str = None) -> Dict[str, Any]:
        """Analyze database schema"""
        try:
            schema_info = {}
            if HAS_SQLALCHEMY:
                inspector = inspect(self.engines['default'])
                tables = inspector.get_table_names()
                for table in tables:
                    columns = inspector.get_columns(table)
                    schema_info[table] = [{"name": c["name"], "type": str(c["type"])} for c in columns]
            else:
                # SQLite fallback
                cur = self.connections['default'].cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cur.fetchall()
                for table in tables:
                    t_name = table[0]
                    cur.execute(f"PRAGMA table_info({t_name})")
                    columns = cur.fetchall()
                    schema_info[t_name] = [{"name": c[1], "type": c[2]} for c in columns]

            return {
                "success": True,
                "result": {
                    "schema": schema_info,
                    "summary": f"Database contains {len(schema_info)} tables."
                },
                "agent": "SQLAgent",
                "operation": "schema_analysis"
            }
        except Exception as e:
             return {"success": False, "error": f"Schema analysis failed: {str(e)}", "agent": "SQLAgent"}

    def _generate_sql_query(self, query: str, table_name: str, data: Any) -> Dict[str, Any]:
        """Generate SQL query using an LLM"""
        try:
            if not self.llm_client:
                 return {
                    "success": False, 
                    "error": "LLM capabilities unavailable. Cannot generate SQL from natural language.",
                    "agent": "SQLAgent"
                }

            # 1. Get Schema Context
            if data is not None and isinstance(data, pd.DataFrame):
                columns = ", ".join([f"{col} ({dtype})" for col, dtype in data.dtypes.items()])
                schema_context = f"Table '{table_name}' has columns: {columns}"
            else:
                # Try to fetch from DB
                schema_res = self._analyze_schema()
                schema_context = f"Database Schema: {schema_res.get('result', {}).get('schema', 'Unknown')}"

            # 2. Construct Prompt
            prompt = f"""
            You are an expert SQL generator. Convert the following natural language request into a valid SQL query (SQLite dialect).
            
            Context:
            {schema_context}
            
            Request: "{query}"
            
            Rules:
            1. Return ONLY the raw SQL query. No markdown, no explanation.
            2. Use SQLite syntax.
            3. Do not use potentially dangerous commands (DROP, DELETE, UPDATE). SELECT only.
            """
            
            # 3. Call LLM
            response = self.llm_client.generate_response(prompt, temperature=0.1)
            
            # 4. Clean formatting
            sql = response.replace("```sql", "").replace("```", "").strip()
            
            return {
                "success": True,
                "result": {
                    "generated_sql": sql,
                    "explanation": "Generated from natural language request."
                },
                "agent": "SQLAgent"
            }
        except Exception as e:
            return {"success": False, "error": f"Query generation failed: {str(e)}", "agent": "SQLAgent"}

    def _execute_sql_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute SQL query safely"""
        try:
            # Security Check
            # Security Check - BLOCK DESTRUCTIVE OPERATIONS ONLY
            # User requirement: Allow operations except those that "harm" (delete/remove) the DB.
            forbidden = ["DROP ", "DELETE ", "TRUNCATE "]
            q_upper = query.upper()
            if any(cmd in q_upper for cmd in forbidden):
                 return {"success": False, "error": "Security Alert: Destructive operations (DROP, DELETE, TRUNCATE) are prohibited.", "agent": "SQLAgent"}

            # Determine connection to use
            # If we recently loaded data, active_connection_id might be 'default' (memory)
            # If default is 'external' but we are querying uploaded data, we must ensure we look at memory.
            # But the query context usually implies intent.
            
            target_conn_id = self.active_connection_id
            
            # Safe execution
            if target_conn_id in self.connections:
                conn = self.connections[target_conn_id]
            else:
                conn = self.connections['default'] # Fallback
            
            # Additional check: If using external DB, ensure read-only transaction if possible
            # (Difficult to enforce strictly at driver level without user permissions, but regex catch is first line of defense)
            
            if HAS_PANDAS:
                df_result = pd.read_sql_query(query, conn)
                results = df_result.to_dict(orient='records')
                columns = list(df_result.columns)
            else:
                cur = conn.cursor()
                cur.execute(query)
                columns = [description[0] for description in cur.description]
                results = [dict(zip(columns, row)) for row in cur.fetchall()]
            
            return {
                "success": True,
                "result": {
                    "sql_query": query,
                    "results": results[:100], # Limit return size
                    "row_count": len(results),
                    "columns": columns
                },
                "agent": "SQLAgent",
                "operation": "query_execution"
            }
        except Exception as e:
            return {"success": False, "error": f"Query execution failed: {str(e)}", "agent": "SQLAgent"}

    def _optimize_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Provide optimization suggestions (stub - can use LLM later)"""
        # Could call LLM here too, but simple rules are fast
        suggestions = []
        if "SELECT *" in query.upper():
            suggestions.append("Select specific columns instead of * to reduce load.")
        if "LIKE '%...%'" in query.upper():
            suggestions.append("Leading wildcards prevent index usage.")
            
        return {
            "success": True,
            "result": {
                "original_query": query,
                "suggestions": suggestions if suggestions else ["Query looks reasonable."]
            },
            "agent": "SQLAgent",
            "operation": "query_optimization"
        }