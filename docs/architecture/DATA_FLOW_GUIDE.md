# 🔄 Complete Data Flow Guide - Nexus LLM Analytics

This document explains the **end-to-end data flow** for different data types through the entire system, including exact file names and function calls.

---

## 📊 Table of Contents

1. [CSV/Structured Data Flow](#1-csvstructured-data-flow)
2. [PDF/Document Data Flow (RAG)](#2-pdfdocument-data-flow-rag)
3. [Text Input Flow](#3-text-input-flow)
4. [Multi-File Analysis Flow](#4-multi-file-analysis-flow)
5. [Visualization Flow](#5-visualization-flow)
6. [Report Generation Flow](#6-report-generation-flow)

---

## 1. CSV/Structured Data Flow

### **User uploads CSV file: `sales_data.csv`**

```
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND (Next.js/React)                                        │
└─────────────────────────────────────────────────────────────────┘

1. User Action: Upload CSV file
   📁 File: src/frontend/app/page.tsx
   - Component: <FileUpload> 
   - Handles: File selection, validation
   - Max Size: 10MB
   
2. File Upload API Call
   📁 File: src/frontend/app/page.tsx (handleFileUpload function)
   - POST → http://localhost:8000/upload/
   - FormData: { file: sales_data.csv }
   
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND - FILE UPLOAD (FastAPI)                                 │
└─────────────────────────────────────────────────────────────────┘

3. Upload Endpoint Receives File
   📁 File: src/backend/api/upload.py
   - Endpoint: @router.post("/")
   - Function: async def upload_file(file: UploadFile)
   
4. File Validation & Storage
   📁 File: src/backend/api/upload.py
   - Validates: Extension (.csv), size
   - Saves to: data/uploads/sales_data.csv
   - Returns: { success: true, filename: "sales_data.csv" }

5. Optional: Auto-Index for RAG
   📁 File: src/backend/api/upload.py
   - Calls: initialize_rag_if_needed()
   - For structured data: Skipped (RAG for documents only)

┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND - QUERY SUBMISSION                                     │
└─────────────────────────────────────────────────────────────────┘

6. User Asks Question
   📁 File: src/frontend/components/query-input.tsx
   - User types: "What is the total revenue?"
   - Component: <QueryInput>
   - Triggers: onQuery callback

7. Analysis Request
   📁 File: src/frontend/app/page.tsx (handleQuery function)
   - POST → http://localhost:8000/analyze
   - Body: {
       query: "What is the total revenue?",
       filename: "sales_data.csv"
     }

┌─────────────────────────────────────────────────────────────────┐
│ BACKEND - ANALYSIS ROUTING                                      │
└─────────────────────────────────────────────────────────────────┘

8. Analyze Endpoint
   📁 File: src/backend/api/analyze.py
   - Endpoint: @router.post("/")
   - Function: async def analyze_query(request: AnalyzeRequest)
   - Creates: analysis_id for tracking

9. Get AnalysisService Singleton
   📁 File: src/backend/services/analysis_service.py
   - Function: get_analysis_service()
   - Returns: Single shared AnalysisService instance
   - Prevents: Multiple costly initializations

10. Route via Analysis Service
    📁 File: src/backend/api/analyze.py
    - Calls: service.analyze(
        query="What is the total revenue?",
        context={"filename": "sales_data.csv"}
      )

┌─────────────────────────────────────────────────────────────────┐
│ PLUGIN SYSTEM ORCHESTRATION                                     │
└─────────────────────────────────────────────────────────────────┘

11. Plugin Registry Routes Query
    📁 File: src/backend/core/plugin_system.py
    - Function: registry.route_query()
    - Detects: File extension = .csv
    - Routes to: DataAnalystAgent (highest confidence)

12. Structured Data Analysis
    📁 File: src/backend/plugins/data_analyst_agent.py
    - Function: execute()
    - Applies: Data optimization
    - Calls: _execute_with_cot() or _execute_direct()

13. Data Optimization
    📁 File: utils/data_optimizer.py
    - Class: DataOptimizer
    - Function: optimize_for_llm(filepath)
    - Actions:
      • Loads CSV: pandas.read_csv()
      • Handles nested data: Flattens JSON columns
      • Samples rows: First 5 rows for LLM
      • Calculates stats: df.describe()
      • Returns: {
          preview: "structured summary",
          stats: {...},
          sample: [{row1}, {row2}...],
          total_rows: 1000
        }

14. Query Parsing (NLP)
    📁 File: src/backend/agents/query_parser.py
    - Class: QueryParser
    - Function: parse_query(query, available_columns, data_sample)
    - Uses: Intent classification (aggregate, filter, correlation, etc.)
    - Returns: ParsedQuery object with:
      • intent: QueryIntent.AGGREGATE
      • entities: ["revenue", "total"]
      • confidence: 0.95

15. Analysis Plan Generation
    📁 File: src/backend/agents/query_parser.py
    - Function: generate_analysis_plan(parsed_query)
    - Returns: {
        steps: ["Load data", "Sum revenue column", "Format result"],
        code_template: "df['revenue'].sum()"
      }

┌─────────────────────────────────────────────────────────────────┐
│ LLM ANALYSIS                                                    │
└─────────────────────────────────────────────────────────────────┘

16. LLM Call via Plugin Agent
    📁 File: src/backend/plugins/data_analyst_agent.py
    - Function: execute() → _execute_direct()
    - Creates prompt with:
      • Query: "What is the total revenue?"
      • Data preview: First 5 rows
      • Pre-calculated stats: df.describe()
      • Available columns: ["date", "product", "revenue"]

17. LLM Client
    📁 File: src/backend/core/llm_client.py
    - Class: LLMClient
    - Function: generate_primary(prompt)
    - Model: phi3:mini (or auto-selected)
    - With: Circuit breaker protection

18. Model Selection
    📁 File: src/backend/core/model_selector.py
    - Class: ModelSelector
    - Function: select_optimal_models()
    - Logic:
      • Checks: Available RAM
      • < 4GB → tinyllama
      • 4-8GB → phi3:mini
      • > 8GB → llama3.1:8b

19. Ollama API Call
    📁 File: src/backend/core/llm_client.py
    - POST → http://localhost:11434/api/generate
    - Body: {
        model: "phi3:mini",
        prompt: "...",
        stream: false
      }
    - Returns: { response: "Total revenue is $125,450" }

20. Circuit Breaker (Resilience)
    📁 File: src/backend/core/circuit_breaker.py
    - Class: CircuitBreaker
    - Monitors: LLM call failures
    - States: CLOSED → OPEN → HALF_OPEN
    - Protects: System from cascading failures

┌─────────────────────────────────────────────────────────────────┐
│ CACHING LAYER                                                   │
└─────────────────────────────────────────────────────────────────┘

21. Advanced Cache Check
    📁 File: src/backend/core/advanced_cache.py
    - Decorator: @cached_query(ttl=1800, tags={'structured_data'})
    - Cache key: hash(query + filename + file_hash)
    - Storage: In-memory dictionary
    - TTL: 30 minutes
    - Hit: Returns cached result immediately
    - Miss: Proceeds to LLM

22. Cache Invalidation
    📁 File: src/backend/core/advanced_cache.py
    - Function: invalidate_cache(tags=['structured_data'])
    - Triggers:
      • File content changes (file_hash differs)
      • Manual cache clear
      • TTL expires

┌─────────────────────────────────────────────────────────────────┐
│ RESPONSE FORMATTING                                             │
└─────────────────────────────────────────────────────────────────┘

23. Result Packaging
    📁 File: src/backend/services/analysis_service.py
    - Function: analyze()
    - Returns: {
        success: true,
        result: "Total revenue is $125,450",
        agent: "DataAnalyst",
        metadata: {filename, execution_time},
        type: "analysis_result"
      }

24. Send to Frontend
    📁 File: src/backend/api/analyze.py
    - Returns JSON response
    - Status: 200 OK

┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND - DISPLAY RESULTS                                      │
└─────────────────────────────────────────────────────────────────┘

25. Results Component
    📁 File: src/frontend/components/results-display.tsx
    - Receives: Analysis result
    - Displays:
      • Answer: "Total revenue is $125,450"
      • Execution time: 2.5s
      • Code used: df['revenue'].sum()

26. Auto-Generate Review Insights
    📁 File: src/frontend/components/results-display.tsx
    - Function: generateReviewInsights()
    - POST → http://localhost:8000/analyze/review-insights
    - Uses: Review model (phi3:mini)
    - Shows: Quality assessment, suggestions

27. Auto-Generate Visualization
    📁 File: src/frontend/components/results-display.tsx
    - Function: generateVisualization()
    - POST → http://localhost:8000/visualize/goal-based
    - Shows: Smart chart suggestions

```

---

## 2. PDF/Document Data Flow (RAG)

### **User uploads PDF document: `research_paper.pdf`**

```
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND - FILE UPLOAD                                          │
└─────────────────────────────────────────────────────────────────┘

1. User uploads PDF
   📁 File: src/frontend/app/page.tsx
   - POST → http://localhost:8000/upload/
   - FormData: { file: research_paper.pdf }

┌─────────────────────────────────────────────────────────────────┐
│ BACKEND - UPLOAD & RAG INDEXING                                │
└─────────────────────────────────────────────────────────────────┘

2. Upload Endpoint
   📁 File: src/backend/api/upload.py
   - Saves to: data/uploads/research_paper.pdf
   - Detects: Extension = .pdf (unstructured)

3. RAG Initialization Trigger
   📁 File: src/backend/api/upload.py
   - Function: initialize_rag_if_needed()
   - Calls: rag_tool.initialize_rag_for_file(filepath)

4. Document Processing
   📁 File: src/backend/tools/rag_tool.py
   - Class: RAGTool
   - Function: initialize_rag_for_file()
   
5. Text Extraction
   📁 File: src/backend/tools/rag_tool.py
   - Library: PyMuPDF (fitz)
   - Actions:
     • Opens PDF: fitz.open(filepath)
     • Extracts text: page.get_text()
     • Combines pages: full_text

6. Text Chunking
   📁 File: src/backend/tools/rag_tool.py
   - Splits text into chunks:
     • Size: 500 characters
     • Overlap: 50 characters
   - Purpose: Better retrieval granularity

7. Vector Embedding
   📁 File: src/backend/tools/rag_tool.py
   - Uses: sentence-transformers
   - Model: all-MiniLM-L6-v2
   - Converts: Text chunks → 384-dim vectors

8. ChromaDB Storage
   📁 File: src/backend/tools/rag_tool.py
   - Creates collection: research_paper_pdf_collection
   - Stores:
     • Documents: Text chunks
     • Embeddings: Vectors
     • Metadata: {filename, page_num, chunk_id}
   - Location: data/chroma_db/

┌─────────────────────────────────────────────────────────────────┐
│ QUERY PROCESSING (RAG)                                          │
└─────────────────────────────────────────────────────────────────┘

9. User Asks Question
   📁 File: src/frontend/app/page.tsx
   - Query: "What are the key findings of this research?"
   - POST → http://localhost:8000/analyze

10. Analyze Endpoint Routes
    📁 File: src/backend/api/analyze.py
    - Calls: service.analyze(query, context)

11. Plugin Registry Routes to RAG
    📁 File: src/backend/core/plugin_system.py
    - Function: route_query()
    - Detects: File extension = .pdf
    - Routes to: RAGAgent (highest confidence for documents)

12. RAG Agent Analysis
    📁 File: src/backend/plugins/rag_agent.py
    - Function: execute()
    - Uses: DocumentIndexer for vector search
    - Calls: _query_documents()

13. Vector Similarity Search
    📁 File: src/backend/tools/rag_tool.py
    - Function: _run(query, n_results=3)
    - Steps:
      1. Embed query: "What are key findings?" → vector
      2. ChromaDB search: Find 3 most similar chunks
      3. Returns: Relevant text excerpts with scores

14. Context-Aware LLM Call
    📁 File: src/backend/plugins/rag_agent.py
    - Builds prompt:
      • User query: "What are key findings?"
      • Retrieved context: [chunk1, chunk2, chunk3]
      • Instruction: "Answer based on context"
    - Calls: llm_client.generate_primary(enhanced_prompt)

15. LLM Response
    📁 File: src/backend/core/llm_client.py
    - Model: phi3:mini
    - Returns: "Key findings include: 1) X, 2) Y, 3) Z..."

16. Result with Sources
    📁 File: src/backend/plugins/rag_agent.py
    - Returns: {
        result: "Key findings include...",
        type: "rag_analysis",
        sources: ["Page 5, Chunk 12", "Page 8, Chunk 23"]
      }

```

---

## 3. Text Input Flow

### **User pastes text directly: "my name is harsha"**

```
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND - TEXT INPUT                                           │
└─────────────────────────────────────────────────────────────────┘

1. User Switches to Text Input Tab
   📁 File: src/frontend/app/page.tsx
   - Component: <Tabs> with value="text"
   - Shows: <Textarea> for text input

2. User Pastes Text
   📁 File: src/frontend/app/page.tsx
   - State: textInput = "my name is harsha and i study at mlrit"
   - onChange: setTextInput(e.target.value)

3. User Asks Question
   📁 File: src/frontend/app/page.tsx
   - Query: "What information can you extract?"
   - POST → http://localhost:8000/analyze
   - Body: {
       query: "What information can you extract?",
       text_data: "my name is harsha and i study at mlrit"
     }

┌─────────────────────────────────────────────────────────────────┐
│ BACKEND - DIRECT TEXT ANALYSIS                                 │
└─────────────────────────────────────────────────────────────────┘

4. Analyze Endpoint Receives Text
   📁 File: src/backend/api/analyze.py
   - Function: analyze_query(request: AnalyzeRequest)
   - Detects: request.text_data is present
   - Skips: File-based routing

5. Direct LLM Analysis
   📁 File: src/backend/api/analyze.py
   - No file creation needed
   - Builds prompt directly:
     • User query: "What information can you extract?"
     • Text content: "my name is harsha..."
   - Calls: llm_client.generate_primary(analysis_prompt)

6. LLM Extracts Information
   📁 File: src/backend/core/llm_client.py
   - Model: phi3:mini
   - Returns: "Extracted info: Name=Harsha, Institution=MLRIT, Course=AIML"

7. Response to Frontend
   📁 File: src/backend/api/analyze.py
   - Returns: {
       success: true,
       result: "Extracted info: Name=Harsha...",
       filename: "Text Input",
       type: "text_analysis"
     }

```

---

## 4. Multi-File Analysis Flow

### **User uploads 2 CSVs: `customers.csv` + `orders.csv`**

```
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND - MULTI-FILE UPLOAD                                    │
└─────────────────────────────────────────────────────────────────┘

1. User Selects Multiple Files
   📁 File: src/frontend/app/page.tsx
   - Component: <FileUpload multiple={true}>
   - Files: [customers.csv, orders.csv]

2. Upload Both Files
   📁 File: src/frontend/app/page.tsx
   - Two POST calls → http://localhost:8000/upload/
   - Stores: Both in data/uploads/

3. Multi-File Query
   📁 File: src/frontend/app/page.tsx
   - Query: "Show customer order totals"
   - POST → http://localhost:8000/analyze
   - Body: {
       query: "Show customer order totals",
       filenames: ["customers.csv", "orders.csv"]
     }

┌─────────────────────────────────────────────────────────────────┐
│ BACKEND - MULTI-FILE HANDLING                                   │
└─────────────────────────────────────────────────────────────────┘

4. Analyze Endpoint
   📁 File: src/backend/api/analyze.py
   - Receives: filenames array (length > 1)
   - Calls: service.analyze(query, context={filenames})

5. Plugin Registry Routes Multi-File
   📁 File: src/backend/core/plugin_system.py
   - Function: route_query()
   - Detects: len(files) > 1
   - Routes to: DataAnalystAgent with multi-file context

6. Load Both Files
   📁 File: src/backend/plugins/data_analyst_agent.py
   - Function: execute()
   - Loads: 
     • df1 = pd.read_csv("customers.csv")
     • df2 = pd.read_csv("orders.csv")

7. Detect Join Columns
   📁 File: src/backend/plugins/data_analyst_agent.py
   - Finds common columns: ["customer_id"]
   - Prioritizes: Columns with "_id" suffix

8. Merge DataFrames
   📁 File: src/backend/plugins/data_analyst_agent.py
   - Executes: merged_df = pd.merge(df1, df2, on='customer_id')
   - Saves temp file: data/uploads/merged_customers_orders.csv

9. Analyze Merged Data
   📁 File: src/backend/plugins/data_analyst_agent.py
   - Calls: _execute_direct() or _execute_with_cot()
   - Proceeds with standard structured analysis flow

```

---

## 5. Visualization Flow

### **User clicks "Generate Chart" for `sales.csv`**

```
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND - VISUALIZATION REQUEST                                │
└─────────────────────────────────────────────────────────────────┘

1. User Requests Chart
   📁 File: src/frontend/components/results-display.tsx
   - Function: generateVisualization()
   - Auto-triggered after analysis

2. Get Chart Suggestions
   📁 File: src/frontend/components/results-display.tsx
   - POST → http://localhost:8000/visualize/suggestions
   - Body: { filename: "sales.csv" }

┌─────────────────────────────────────────────────────────────────┐
│ BACKEND - CHART SUGGESTION ENGINE                               │
└─────────────────────────────────────────────────────────────────┘

3. Suggestions Endpoint
   📁 File: src/backend/api/visualize.py
   - Endpoint: @router.post("/suggestions")
   - Function: suggest_charts()

4. Data Profiler Analysis
   📁 File: src/backend/core/data_profiler.py
   - Class: DataProfiler
   - Function: analyze_data(df)
   - Detects:
     • Temporal columns: ["date", "month", "timestamp"]
     • Numeric columns: ["revenue", "quantity", "price"]
     • Categorical columns: ["product", "region", "category"]
     • Relationships: Correlations between columns

5. Template Matcher
   📁 File: src/backend/visualization/chart_templates.py
   - Class: ChartTemplateEngine
   - Function: match_template()
   - Logic:
     • Has temporal + numeric → TIME_SERIES template
     • Has categories + numeric → BAR/PIE template
     • Has two numerics → SCATTER template
     • Multi-category → GROUPED_BAR template

6. Priority Scoring
   📁 File: src/backend/visualization/chart_recommender.py
   - Class: ChartRecommender
   - Function: recommend_charts()
   - Scores based on:
     • Data type compatibility: 0-10
     • Column count match: 0-10
     • Use case relevance: 0-10
   - Returns: Top 5 suggestions sorted by score

7. Response with Suggestions
   📁 File: src/backend/api/visualize.py
   - Returns: {
       suggestions: [
         {type: "line", priority_score: 9.5, reasoning: "..."},
         {type: "bar", priority_score: 8.0, reasoning: "..."},
         ...
       ]
     }

┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND - CHART GENERATION                                     │
└─────────────────────────────────────────────────────────────────┘

8. Display Suggestions
   📁 File: src/frontend/components/results-display.tsx
   - Shows: Top 3 chart recommendations
   - User sees: Reasoning, use cases, priority scores

9. Generate Goal-Based Chart
   📁 File: src/frontend/components/results-display.tsx
   - POST → http://localhost:8000/visualize/goal-based
   - Body: {
       filename: "sales.csv",
       goal: "Show revenue trends over time",
       library: "plotly"
     }

┌─────────────────────────────────────────────────────────────────┐
│ BACKEND - CHART CREATION                                        │
└─────────────────────────────────────────────────────────────────┘

10. Goal-Based Endpoint
    📁 File: src/backend/api/visualize.py
    - Endpoint: @router.post("/goal-based")
    - Function: generate_goal_based_chart()

11. Template Selection
    📁 File: src/backend/visualization/chart_templates.py
    - Selects: TIME_SERIES template (for "trends over time")
    - Maps columns:
      • x_axis: "date" (temporal)
      • y_axis: "revenue" (numeric)

12. Plotly Chart Generation
    📁 File: src/backend/visualization/plotly_generator.py
    - Class: PlotlyChartGenerator
    - Function: generate_chart()
    - Creates: plotly.graph_objects.Figure
    - Applies:
      • Template styling
      • Colors, labels, titles
      • Responsive layout

13. Chart Serialization
    📁 File: src/backend/api/visualize.py
    - Converts: fig.to_json() → JSON string
    - Returns: {
        visualization: {
          figure_json: "{...plotly spec...}",
          chart_type: "line"
        },
        selected_chart: {type: "line", priority_score: 9.5}
      }

┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND - CHART DISPLAY                                        │
└─────────────────────────────────────────────────────────────────┘

14. Chart Viewer Component
    📁 File: src/frontend/components/chart-viewer.tsx
    - Receives: figure_json
    - Uses: Plotly.js library
    - Renders: Interactive chart with:
      • Zoom/pan controls
      • Download as PNG
      • Fullscreen mode

```

---

## 6. Report Generation Flow

### **User clicks "Download Report"**

```
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND - REPORT REQUEST                                       │
└─────────────────────────────────────────────────────────────────┘

1. Export Report Button
   📁 File: src/frontend/components/results-display.tsx
   - Button: "Export Report"
   - onClick: handleExportReport()

2. Report Generation Request
   📁 File: src/frontend/components/results-display.tsx
   - POST → http://localhost:8000/generate-report/
   - Body: {
       results: [analysisResult1, analysisResult2],
       format_type: "pdf",  // or "excel" or "both"
       title: "Q1 Sales Analysis",
       include_methodology: true,
       include_raw_data: true
     }

┌─────────────────────────────────────────────────────────────────┐
│ BACKEND - REPORT GENERATION                                     │
└─────────────────────────────────────────────────────────────────┘

3. Report Endpoint
   📁 File: src/backend/api/report.py
   - Endpoint: @router.post("/")
   - Function: generate_report()

4. Enhanced Report Manager
   📁 File: src/backend/core/enhanced_reports.py
   - Class: EnhancedReportManager
   - Function: generate_report(analysis_results, format_type)

5. PDF Report Generation
   📁 File: src/backend/core/enhanced_reports.py
   - Class: PDFReportGenerator
   - Library: ReportLab
   - Creates sections:
     • _create_title_page(): Cover with title, date, logo
     • _create_executive_summary(): Key findings overview
     • _create_table_of_contents(): Navigable TOC
     • _create_analysis_section(): Individual analysis results
     • _create_methodology_section(): Technical details
     • _create_appendix(): Raw data summaries

6. Excel Report Generation
   📁 File: src/backend/core/enhanced_reports.py
   - Class: ExcelReportGenerator
   - Library: openpyxl
   - Creates sheets:
     • _create_summary_sheet(): Executive summary
     • _create_results_sheets(): One sheet per analysis
     • _create_data_sheet(): Raw data tables

7. Chart Embedding (if visualizations exist)
   📁 File: src/backend/core/enhanced_reports.py
   - Extracts: visualization.figure_json from results
   - Converts: Plotly JSON → Image (for PDF)
   - Embeds: In appropriate sections

8. Save Report Files
   📁 File: src/backend/api/report.py
   - Saves to: data/reports/nexus_report_20251028_143022.pdf
   - Copies to: src/backend/data/ (for download endpoint)

9. Return File Path
   📁 File: src/backend/api/report.py
   - Returns: {
       success: true,
       report_path: "data/reports/nexus_report_20251028_143022.pdf",
       format: "pdf",
       analysis_count: 2
     }

┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND - REPORT DOWNLOAD                                      │
└─────────────────────────────────────────────────────────────────┘

10. Download Report
    📁 File: src/frontend/components/results-display.tsx
    - GET → http://localhost:8000/generate-report/download-report
    - Receives: Binary PDF/Excel file

11. Browser Download
    📁 File: src/frontend/components/results-display.tsx
    - Creates: Blob from response
    - Triggers: Browser download dialog
    - Filename: nexus_report_20251028_143022.pdf

```

---

## 🔄 Cross-Cutting Concerns

### **Error Handling** (All Flows)

```
📁 File: src/backend/core/error_handling.py
- Function: friendly_error()
- Returns: User-friendly error messages with suggestions

Example:
  Error: "Connection refused to Ollama"
  Friendly: "AI service unavailable. Start Ollama: 'ollama serve'"
```

### **Circuit Breaker** (All LLM Calls)

```
📁 File: src/backend/core/circuit_breaker.py
- Class: CircuitBreaker
- Monitors: Failure rates
- States: CLOSED → OPEN (stops calls) → HALF_OPEN (retry)
- Prevents: Cascading failures
```

### **Caching** (All Analysis)

```
📁 File: src/backend/core/advanced_cache.py
- Decorator: @cached_query(ttl, tags)
- Storage: In-memory dictionary
- TTL:
  • Structured data: 30 minutes
  • RAG analysis: 45 minutes
  • Visualizations: 60 minutes
- Invalidation: On file content change
```

### **Analysis Tracking** (All Queries)

```
📁 File: src/backend/core/analysis_manager.py
- Class: AnalysisManager
- Functions:
  • start_analysis() → Creates analysis_id
  • update_analysis_stage() → Tracks progress
  • complete_analysis() → Marks done
  • cancel_analysis() → User cancellation
- Purpose: Real-time progress tracking, cancellation
```

### **Model Selection** (All LLM Operations)

```
📁 File: src/backend/core/model_selector.py
- Class: ModelSelector
- Function: select_optimal_models()
- Logic:
  • Detects: Available system RAM
  • Checks: Installed Ollama models
  • Selects:
    - Low RAM (< 4GB): tinyllama
    - Medium RAM (4-8GB): phi3:mini
    - High RAM (> 8GB): llama3.1:8b
- Updates: User preferences automatically
```

### **User Preferences** (Persistent Config)

```
📁 File: src/backend/core/user_preferences.py
- Class: UserPreferencesManager
- Storage: config/user_preferences.json
- Contains:
  • primary_model: "phi3:mini"
  • review_model: "phi3:mini"
  • enable_review: false
  • auto_optimize_data: true
- Loaded: On backend startup
- Updated: Via API or setup wizard
```

---

## 📊 File System Structure

```
data/
├── uploads/              ← User-uploaded files
│   ├── sales_data.csv
│   ├── research_paper.pdf
│   └── text_input_20251028_143022.txt
│
├── samples/              ← Sample datasets for testing
│   ├── test_sales_monthly.csv
│   ├── test_employee_data.csv
│   └── test_iot_sensor.csv
│
├── chroma_db/            ← RAG vector database
│   └── research_paper_pdf_collection/
│       ├── embeddings
│       └── metadata
│
├── reports/              ← Generated reports
│   ├── nexus_report_20251028_143022.pdf
│   └── nexus_report_20251028_143022.xlsx
│
├── audit/                ← Audit logs
│   └── audit_log.jsonl
│
└── history/              ← Query history
    └── session_abc123.json

config/
└── user_preferences.json ← User configuration

src/
├── backend/
│   ├── main.py              ← FastAPI app entry point
│   ├── api/                 ← REST API endpoints
│   │   ├── analyze.py       ← /analyze endpoint
│   │   ├── upload.py        ← /upload endpoint
│   │   ├── visualize.py     ← /visualize/* endpoints
│   │   ├── report.py        ← /generate-report endpoint
│   │   └── models.py        ← /models/* endpoints
│   ├── agents/              ← Agent infrastructure
│   │   └── model_initializer.py ← Model setup
│   ├── services/            ← Service layer
│   │   └── analysis_service.py ← Main orchestrator
│   ├── plugins/             ← Plugin agents
│   │   ├── data_analyst_agent.py ← CSV/JSON analysis
│   │   ├── rag_agent.py     ← PDF/document RAG
│   │   ├── statistical_agent.py ← Statistical analysis
│   │   ├── financial_agent.py ← Financial metrics
│   │   ├── ml_insights_agent.py ← ML patterns
│   │   ├── time_series_agent.py ← Time series
│   │   ├── sql_agent.py     ← SQL generation
│   │   ├── visualizer_agent.py ← Charts
│   │   ├── reporter_agent.py ← Reports
│   │   └── reviewer_agent.py ← Review
│   ├── core/                ← Core services
│   │   ├── plugin_system.py ← Agent registry & routing
│   │   ├── llm_client.py    ← Ollama API client
│   │   ├── model_selector.py← Smart model selection
│   │   ├── self_correction_engine.py ← CoT correction
│   │   ├── circuit_breaker.py← Resilience
│   │   ├── advanced_cache.py← Caching layer
│   │   ├── analysis_manager.py← Analysis tracking
│   │   ├── user_preferences.py← Config management
│   │   ├── document_indexer.py ← RAG indexing
│   │   ├── chromadb_client.py ← Vector DB
│   │   ├── sandbox.py       ← Secure execution
│   │   └── query_parser.py  ← Query parsing
│   └── visualization/       ← Chart generation
│       ├── dynamic_charts.py← Chart templates
│       └── scaffold.py      ← Visualization scaffolding
│
├── frontend/
│   ├── app/
│   │   └── page.tsx         ← Main page component
│   ├── components/
│   │   ├── results-display.tsx← Results UI
│   │   ├── query-input.tsx  ← Query input
│   │   ├── chart-viewer.tsx ← Chart display
│   │   └── model-settings.tsx← Settings UI
│   └── lib/
│       └── config.ts        ← API endpoint config
│
└── utils/
    └── data_optimizer.py    ← Data preprocessing

tests/
├── visualization/
│   └── test_report_generation.py
└── unit/
    └── test_llm_client.py
```

---

## 🎯 Key Takeaways

### **1. Routing Decision Points**

| File Type | Detected In | Routes To | Key File |
|-----------|-------------|-----------|----------|
| .csv, .json, .xlsx | plugin_system.py | DataAnalystAgent | data_analyst_agent.py |
| .pdf, .txt, .docx | plugin_system.py | RAGAgent | rag_agent.py |
| text_data param | analyze.py | DataAnalystAgent | data_analyst_agent.py |
| Multiple files | analyze.py | DataAnalystAgent (multi-file) | data_analyst_agent.py |

### **2. Performance Optimizations**

- **Singleton Pattern**: analysis_service.py prevents multiple service instances
- **Caching**: advanced_cache.py stores results for 30-45 minutes
- **Data Optimization**: data_optimizer.py sends only 5 rows to LLM
- **Circuit Breaker**: circuit_breaker.py prevents cascading failures
- **Plugin Routing**: Direct agent execution via plugin_system.py

### **3. Smart Features**

- **Auto Model Selection**: model_selector.py picks model based on RAM
- **Query Parsing**: query_parser.py understands intent (aggregate, filter, etc.)
- **Chart Recommendations**: dynamic_charts.py suggests best visualizations
- **Multi-File Joins**: data_analyst_agent.py automatically merges related CSVs
- **RAG Indexing**: document_indexer.py auto-indexes documents on upload
- **Self-Correction**: self_correction_engine.py refines analysis with CoT

### **4. User Experience Flow**

```
Upload File → Auto-Index (if PDF) → Ask Question → Plugin Routing 
→ Cache Check → Agent Execution → Self-Correction (if enabled)
→ Auto-Visualize → Auto-Review → Display Results → Export Report
```

All steps are **automatic** except "Upload File" and "Ask Question"!

---

## 📝 Quick Reference

**Want to trace a specific flow?** Start here:

- **CSV Analysis**: `analyze.py` → `analysis_service.py` → `plugin_system.py` → `data_analyst_agent.py`
- **PDF Analysis**: `analyze.py` → `analysis_service.py` → `plugin_system.py` → `rag_agent.py`
- **Visualization**: `results-display.tsx` → `visualize.py` → `dynamic_charts.py`
- **Report Export**: `results-display.tsx` → `report.py` → `enhanced_reports.py`
- **Model Selection**: `model_selector.py` → User prefs loaded from `user_preferences.py`

**Need to modify behavior?**

- Change model: `config/user_preferences.json` or Settings UI
- Add chart type: `visualization/dynamic_charts.py`
- Modify caching: `core/advanced_cache.py` (change TTL)
- Custom error messages: `core/error_handling.py`
- Add new agent: Create plugin in `plugins/` folder (auto-discovered)

---

**Last Updated**: December 22, 2025
**Version**: 2.0 (Updated to reflect Plugin Architecture)
**Maintained By**: Nexus LLM Analytics Team
