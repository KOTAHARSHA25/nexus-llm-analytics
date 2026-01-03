# Nexus LLM Analytics - Project Architecture & Data Flow

## 🏗️ **System Architecture Overview**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NEXUS LLM ANALYTICS PLATFORM                    │
├─────────────────────────────────────────────────────────────────────┤
│                         Frontend Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │   Next.js UI    │  │  React Hook     │  │  WebSocket      │    │
│  │   Dashboard     │  │  State Mgmt     │  │  Real-time      │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                      API Gateway Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │   FastAPI       │  │  Rate Limiting  │  │  Error Handling │    │
│  │   Router        │  │  & Security     │  │  & Validation   │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                    Core Processing Layer                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │ Analysis        │  │  Plugin System  │  │  Optimization   │    │
│  │ Service         │  │  Extensible     │  │  Performance    │    │
│  │ Orchestrator    │  │  Agent Registry │  │  Memory Mgmt    │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                      Agent Ecosystem                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │  Data Analysis  │  │  RAG Specialist │  │  Visualization  │    │
│  │  Agent          │  │  Agent          │  │  Agent          │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │  SQL Plugin     │  │  Review Agent   │  │  Report Gen     │    │
│  │  Agent          │  │                 │  │  Agent          │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                        Data Layer                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │   ChromaDB      │  │  File Storage   │  │  Vector         │    │
│  │   Document      │  │  Upload/Export  │  │  Embeddings     │    │
│  │   Collections   │  │  Management     │  │  Processing     │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                       Model Layer                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │    Ollama       │  │   Model         │  │   Adaptive      │    │
│  │    LLM Models   │  │   Selection     │  │   Timeout       │    │
│  │    Integration  │  │   Strategy      │  │   Management    │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔄 **Data Flow Architecture**

### **1. Request Processing Flow**

```
User Request → Frontend → API Gateway → AnalysisService → Agent Selection → Processing → Response

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   User      │───▶│  Frontend   │───▶│  FastAPI    │───▶│  Analysis   │
│  Interface  │    │   Next.js   │    │   Gateway   │    │  Service    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                  │
                                                                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Response   │◀───│  Agent      │◀───│  Plugin     │◀───│  Intelligent│
│  Formatted  │    │  Execution  │    │  Registry   │    │   Routing   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### **2. File Processing Pipeline**

```
File Upload → Validation → Storage → Analysis → Vector Processing → Results

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  File       │───▶│  Security   │───▶│  Storage    │
│  Upload     │    │  Validation │    │  Manager    │
└─────────────┘    └─────────────┘    └─────────────┘
                                               │
                                               ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Results    │◀───│  Agent      │◀───│  Format     │
│  Return     │    │  Processing │    │  Detection  │
└─────────────┘    └─────────────┘    └─────────────┘
                           │
                           ▼
                   ┌─────────────┐
                   │  ChromaDB   │
                   │  Vector     │
                   │  Storage    │
                   └─────────────┘
```

## 🗂️ **Directory Structure & Components**

### **Core Application Structure**
```
nexus-llm-analytics/
├── src/
│   ├── backend/                 # Python FastAPI Backend
│   │   ├── services/           # Service Layer
│   │   │   ├── analysis_service.py # Central orchestrator
│   │   │   └── history_manager.py  # History tracking
│   │   ├── agents/             # Agent Infrastructure
│   │   │   └── model_initializer.py # Model setup
│   │   ├── api/                # REST API Endpoints
│   │   │   ├── analyze.py          # Analysis endpoints
│   │   │   ├── upload.py           # File upload handling
│   │   │   ├── visualize.py        # Visualization endpoints
│   │   │   ├── report.py           # Report generation
│   │   │   └── models.py           # Model management
│   │   ├── core/               # Core Infrastructure
│   │   │   ├── plugin_system.py    # Plug-and-play agents
│   │   │   ├── optimizers.py       # Performance optimization
│   │   │   ├── llm_client.py       # LLM integration
│   │   │   ├── chromadb_client.py  # Vector database
│   │   │   ├── config.py           # Configuration management
│   │   │   ├── sandbox.py          # Security sandbox
│   │   │   └── error_handling.py   # Comprehensive error mgmt
│   │   └── main.py             # FastAPI application entry
│   └── frontend/               # Next.js Frontend
│       ├── components/             # React components
│       ├── hooks/                  # Custom React hooks
│       ├── pages/                  # Next.js pages
│       └── styles/                 # Styling
├── plugins/                    # Extensible Agent Plugins
│   ├── data_analyst_agent.py   # Data analysis specialist
│   ├── rag_agent.py           # RAG processing specialist
│   ├── visualizer_agent.py    # Chart generation
│   └── sql_agent.py           # SQL analysis plugin
├── data/                       # Data Storage
│   ├── uploads/                   # User uploaded files
│   ├── exports/                   # Generated reports
│   └── samples/                   # Sample datasets
├── chroma_db/                  # ChromaDB Vector Storage
├── reports/                    # Generated analysis reports
├── logs/                       # Application logs
└── tests/                      # Comprehensive test suite
```

## 🔌 **Plugin System Architecture**

### **Extensible Agent Framework**
```
┌─────────────────────────────────────────────────────────────────┐
│                    PLUGIN SYSTEM                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │  BasePlugin     │    │  Agent Registry │    │  Auto       │  │
│  │  Agent          │───▶│  Discovery      │───▶│  Discovery  │  │
│  │  Abstract Class │    │  & Management   │    │  System     │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    INTELLIGENT ROUTING                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │  Query Analysis │    │  Capability     │    │  Best Agent │  │
│  │  & Intent       │───▶│  Matching       │───▶│  Selection  │  │
│  │  Detection      │    │  Algorithm      │    │  & Scoring  │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      PLUGIN EXAMPLES                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │   SQL Agent     │    │   Future:       │    │   Future:   │  │
│  │   Database      │    │   PDF Agent     │    │   API Agent │  │
│  │   Analysis      │    │   Document      │    │   External  │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 💾 **Data Management Architecture**

### **Storage & Processing Flow**
```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                 │
├─────────────────────────────────────────────────────────────────┤
│  File Storage:                                                  │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   data/uploads/ │    │  data/exports/  │                    │
│  │   User Files    │    │  Generated      │                    │
│  │   (.csv,.pdf,   │    │  Reports        │                    │
│  │   .json,.txt)   │    │  (.pdf,.json)   │                    │
│  └─────────────────┘    └─────────────────┘                    │
├─────────────────────────────────────────────────────────────────┤
│  Vector Database:                                               │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   ChromaDB      │    │   Embeddings    │                    │
│  │   Collections   │    │   Processing    │                    │
│  │   Document      │    │   Vector        │                    │
│  │   Chunks        │    │   Similarity    │                    │
│  └─────────────────┘    └─────────────────┘                    │
├─────────────────────────────────────────────────────────────────┤
│  Processing Pipeline:                                           │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Sandbox       │    │   Optimization  │                    │
│  │   Secure Code   │    │   Memory &      │                    │
│  │   Execution     │    │   Performance   │                    │
│  └─────────────────┘    └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 **Processing Flow Details**

### **1. Multi-Agent Orchestration**
```
AnalysisService (Singleton)
├── Intelligent Query Routing
│   ├── Agent Registry Integration
│   ├── Query Complexity Analysis
│   └── Capability Matching
├── Service Controller
│   ├── Data Analyst Plugin
│   ├── RAG Specialist Plugin
│   ├── Visualization Plugin
│   └── Review Plugin
└── Result Synthesis
    ├── Quality Review
    ├── Format Standardization
    └── Response Generation
```

### **2. Request Processing Pipeline**

**Structured Data (CSV, JSON):**
```
Upload → Validation → Data Agent → Analysis → Visualization → Review → Report
```

**Unstructured Data (PDF, TXT):**
```
Upload → RAG Processing → ChromaDB → Vector Search → Summary → Report
```

**SQL/Database Files:**
```
Upload → Plugin Detection → SQL Agent → Schema Analysis → Query Generation → Results
```

### **3. Security & Sandboxing**
```
┌─────────────────────────────────────────────────────────────────┐
│                     SECURITY LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Input Validation:                                              │
│  ├── File Type Validation                                       │
│  ├── Size Limits (configurable)                                 │
│  ├── Content Sanitization                                       │
│  └── Malware Scanning                                           │
├─────────────────────────────────────────────────────────────────┤
│  Execution Sandbox:                                             │
│  ├── Restricted Code Execution                                  │
│  ├── Memory Limits                                              │
│  ├── CPU Time Limits                                            │
│  └── Safe Module Imports Only                                   │
├─────────────────────────────────────────────────────────────────┤
│  Rate Limiting:                                                 │
│  ├── API Request Limits                                         │
│  ├── File Upload Limits                                         │
│  └── Model Usage Throttling                                     │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 **Model Integration Architecture**

### **LLM Management System**
```
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  Model Selection Strategy:                                      │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Intelligent   │    │   Resource      │                    │
│  │   Model         │───▶│   Aware         │                    │
│  │   Selection     │    │   Allocation    │                    │
│  └─────────────────┘    └─────────────────┘                    │
├─────────────────────────────────────────────────────────────────┤
│  Supported Models:                                              │
│  ├── llama3.1:8b     (High-quality analysis)                   │
│  ├── phi3:mini       (Balanced performance)                    │
│  ├── tinyllama       (Low-resource environments)               │
│  └── nomic-embed-text (Vector embeddings)                      │
├─────────────────────────────────────────────────────────────────┤
│  Adaptive Timeout Management:                                   │
│  ├── RAM-based timeout calculation                              │
│  ├── Model complexity awareness                                 │
│  ├── Historical performance tracking                            │
│  └── Graceful degradation strategies                            │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 **Performance Optimization Architecture**

### **Multi-Level Optimization System**
```
┌─────────────────────────────────────────────────────────────────┐
│                  OPTIMIZATION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  Memory Optimization:                                           │
│  ├── System Resource Monitoring                                 │
│  ├── Process Memory Analysis                                    │
│  ├── Cleanup Recommendations                                    │
│  └── Model Compatibility Assessment                             │
├─────────────────────────────────────────────────────────────────┤
│  Performance Optimization:                                      │
│  ├── LRU Caching (O(1) lookups)                                │
│  ├── Heap-based Document Ranking (O(log n))                    │
│  ├── Parallel Processing                                        │
│  └── Query Intent Detection                                     │
├─────────────────────────────────────────────────────────────────┤
│  Startup Optimization:                                          │
│  ├── Background Component Loading                               │
│  ├── Lazy Initialization                                        │
│  ├── Import Management                                          │
│  └── Singleton Pattern Usage                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 **Real-time Communication Architecture**

### **WebSocket Integration**
```
Frontend ←→ WebSocket Manager ←→ Analysis Progress ←→ Live Updates

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Hook    │    │   WebSocket     │    │   Analysis      │
│   useWebSocket  │◄──►│   Manager       │◄──►│   Progress      │
│                 │    │   (Optional)    │    │   Tracking      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🧪 **Testing Architecture**

### **Comprehensive Test Coverage**
```
├── Unit Tests
│   ├── Agent Function Tests
│   ├── API Endpoint Tests
│   └── Core Module Tests
├── Integration Tests
│   ├── End-to-End Workflows
│   ├── Plugin System Tests
│   └── Database Integration
├── Performance Tests
│   ├── Load Testing
│   ├── Memory Usage Analysis
│   └── Response Time Monitoring
└── Security Tests
    ├── Input Validation Tests
    ├── Sandbox Escape Tests
    └── Rate Limiting Tests
```

## 🚀 **Development & Deployment Architecture**

### **Development Environment**
```
┌─────────────────────────────────────────────────────────────────┐
│                 DEVELOPMENT SETUP                               │
├─────────────────────────────────────────────────────────────────┤
│  Backend Development:                                           │
│  ├── Python 3.12+ Virtual Environment                          │
│  ├── FastAPI with Hot Reload                                    │
│  ├── Ollama for Local LLM Testing                               │
│  └── ChromaDB for Vector Storage                                │
├─────────────────────────────────────────────────────────────────┤
│  Frontend Development:                                          │
│  ├── Next.js with TypeScript                                    │
│  ├── React Development Server                                   │
│  ├── TailwindCSS for Styling                                    │
│  └── Component-based Architecture                               │
├─────────────────────────────────────────────────────────────────┤
│  Configuration Management:                                      │
│  ├── Environment Variables (.env)                               │
│  ├── YAML Configuration Files                                   │
│  ├── Runtime Configuration Validation                           │
│  └── Development vs Production Settings                         │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 **Key Architectural Decisions**

### **1. Plugin-First Architecture**
- **Extensible by design** - New agents can be added without code changes
- **Auto-discovery system** - Plugins are automatically detected and loaded
- **Intelligent routing** - Best agent selected based on query and file type

### **2. Multi-Agent Coordination**
- **Plugin Registry orchestration** - Custom agent collaboration via plugin system
- **Specialized roles** - Each agent has specific expertise areas
- **Quality review process** - Built-in review and validation

### **3. Performance-First Design**
- **Advanced algorithms** - O(log n) complexity where possible
- **Adaptive optimization** - System adjusts based on resources
- **Caching strategies** - Multiple levels of intelligent caching

### **4. Security-Conscious Implementation**
- **Sandboxed execution** - Safe code execution environment
- **Input validation** - Comprehensive validation at all entry points
- **Rate limiting** - Protection against abuse

### **5. Developer Experience Focus**
- **Hot reload** - Fast development iteration
- **Comprehensive logging** - Detailed debugging information
- **Modular architecture** - Easy to understand and modify

---

## 🎯 **Architecture Benefits**

### **Scalability**
- Plugin system allows horizontal scaling of capabilities
- Multi-agent design distributes processing load
- Caching reduces computational overhead

### **Maintainability**
- Clean separation of concerns
- Modular, testable components
- Comprehensive error handling

### **Extensibility**
- Easy to add new file types via plugins
- Simple agent creation process
- Configurable processing pipelines

### **Performance**
- Advanced algorithmic optimizations
- Resource-aware processing
- Intelligent model selection

### **Security**
- Sandboxed execution environment
- Comprehensive input validation
- Rate limiting and abuse prevention

---

## 🔬 **DOMAIN-AGNOSTIC VALIDATION**

### **System Independence from Subject Matter**

**Validation Status:** ✅ **CONFIRMED DOMAIN-AGNOSTIC** (December 22, 2025)

This system is **fundamentally domain-agnostic** and operates independently of any specific subject area or industry vertical. Comprehensive audit completed with 100% routing consistency achieved across diverse domains.

### **Key Domain-Agnostic Features**

#### **1. Operation-Based Routing (NOT Vocabulary-Based)**
The routing system classifies queries by **analytical operations**, not domain terminology:

| Operation Type | Example Queries (Any Domain) | Target Agent |
|----------------|------------------------------|--------------|
| **Ratio Calculation** | profit margin, survival rate, pass percentage, conversion rate | StatisticalAgent |
| **Correlation Analysis** | sales vs marketing, drug dosage vs recovery, study hours vs grades | StatisticalAgent |
| **Time Series Forecasting** | revenue prediction, patient admissions, student enrollment | TimeSeriesAgent |
| **Clustering/Grouping** | customer segments, patient profiles, student learning styles | MLInsightsAgent |

**Evidence:** Test suite validates 100% routing consistency (13/13 queries passed across finance, medical, education, marketing domains)

#### **2. Domain-Neutral Enum Structures**

**QueryType Enum** (src/backend/core/intelligent_query_engine.py):
```python
QueryType.DATA_ANALYSIS      # Generic data operations
QueryType.VISUALIZATION      # Visual representation
QueryType.STATISTICS         # Statistical analysis
QueryType.MACHINE_LEARNING   # ML operations
QueryType.NATURAL_LANGUAGE   # Text processing
QueryType.PREDICTION         # Forecasting
QueryType.OPTIMIZATION       # Optimization tasks
```

**AgentCapability Enum**:
```python
AgentCapability.STATISTICAL_ANALYSIS
AgentCapability.RATIO_CALCULATION
AgentCapability.METRICS_COMPUTATION
AgentCapability.DATA_VISUALIZATION
AgentCapability.MACHINE_LEARNING
AgentCapability.PREDICTIVE_ANALYTICS
```

**No domain-specific enums exist** (e.g., no FINANCIAL_ANALYSIS, MEDICAL_DIAGNOSIS, BUSINESS_INTELLIGENCE)

#### **3. Mathematical Routing Formula**

Routing decisions are made purely by confidence scoring:
```
final_score = agent_confidence × 0.8 + agent_priority/100 × 0.2
```

Where:
- `agent_confidence` = Agent's assessment of operation fit (NOT domain fit)
- `agent_priority` = Static priority value (not domain-dependent)

No special weighting for financial, medical, or business queries.

#### **4. Agent Specialization by Operation (Not Domain)**

| Agent | Specialization | Domain Applicability |
|-------|----------------|---------------------|
| **StatisticalAgent** | Statistical tests, correlations, distributions | Any domain with numeric data |
| **MLInsightsAgent** | Clustering, classification, pattern discovery | Any domain with structured data |
| **TimeSeriesAgent** | Forecasting, trend analysis | Any domain with temporal data |
| **FinancialAgent** | **Only when EXPLICIT financial context** (2+ financial keywords OR currency symbols) | Finance/Investment domains ONLY |
| **DataAnalystAgent** | Summary statistics, basic operations | Any domain with tabular data |

**Critical Fix Applied (Dec 22, 2025):**  
FinancialAgent confidence calculation was refactored to require **strict financial context** (explicit financial keywords like "investment", "portfolio", "stock", "bond"). Generic operations like "calculate ratio" now route consistently regardless of domain vocabulary.

### **Validation Evidence**

**Test Suite:** `tests/test_verify_domain_agnostic.py`

**Results:** 13/13 queries passed (100% accuracy)

| Test Category | Query Examples | Expected Behavior | Status |
|---------------|---------------|-------------------|--------|
| **Ratio Calculation** | "Calculate profit margin", "Calculate survival rate", "Calculate pass percentage" | Same agent (StatisticalAgent) for ALL | ✅ PASS |
| **Correlation** | "Correlation between sales and marketing", "Correlation between drug dosage and recovery" | Same agent (StatisticalAgent) for ALL | ✅ PASS |
| **Time Series** | "Predict next quarter revenue", "Predict patient admission trends", "Forecast student enrollment" | Same agent (TimeSeriesAgent) for ALL | ✅ PASS |
| **Clustering** | "Group customers by behavior", "Group patients by symptoms", "Group students by learning patterns" | Same agent (MLInsightsAgent) for ALL | ✅ PASS |

### **Research Validity Implications**

**✅ This system is VALID for academic research claiming domain-agnostic capabilities.**

The architecture supports analysis of:
- ✅ **Financial data** (stocks, revenue, budgets)
- ✅ **Medical data** (patient records, clinical trials)
- ✅ **Educational data** (student performance, curriculum analysis)
- ✅ **Marketing data** (campaigns, conversions, engagement)
- ✅ **Arbitrary domains** (any structured or unstructured data)

**No hidden assumptions** exist that would bias results toward specific fields.

### **Audit Documentation**

Complete audit report available at: [DOMAIN_AGNOSTIC_AUDIT_REPORT.md](DOMAIN_AGNOSTIC_AUDIT_REPORT.md)

**Audit Scope:**
- ✅ Routing logic and confidence calculations
- ✅ Agent capability definitions
- ✅ Preprocessing and data optimization heuristics
- ✅ Configuration files and prompt templates
- ✅ Query classification and type mapping

**Findings Summary:**
- 🟢 Core routing: CLEAN (100% domain-agnostic)
- 🟢 Agent confidence: FIXED (FinancialAgent now requires strict context)
- 🟡 Data optimizer: Contains common column heuristics (acceptable - not routing logic)
- 🟡 Legacy code: One unused "route_to_financial" function (no impact)

**Risk Assessment:** 🟢 **LOW** - All critical components validated as domain-neutral

---

This architecture provides a robust, scalable, and extensible foundation for the Nexus LLM Analytics platform, designed to handle complex data analysis tasks **across any domain or subject area** while maintaining performance, security, and ease of development.