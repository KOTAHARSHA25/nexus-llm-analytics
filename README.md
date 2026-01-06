# 🚀 Nexus LLM Analytics v2

**Advanced AI-Powered Data Analytics Platform with Multi-Agent Architecture**

> A sophisticated, privacy-first data analytics platform that combines the power of Large Language Models with specialized AI agents to provide comprehensive data analysis, visualization, and insights.

---

## 🎯 **Project Overview**

Nexus LLM Analytics is a next-generation data analytics platform that leverages **multi-agent AI architecture** to transform natural language queries into actionable data insights. Built with privacy-first principles, all processing happens locally using **Ollama** for LLM inference.

### **Key Differentiators**
- 🤖 **Multi-Agent System**: 5 core AI agents working in coordination
- 🔌 **Plugin Architecture**: Extensible system with domain specialist agents (Financial, Statistical, ML, Time-Series, SQL)
- 🏠 **Local-First**: Complete privacy - no data leaves your machine
- 🎨 **Modern UI**: React/Next.js interface with real-time updates
- 📊 **Comprehensive Analytics**: Operation-based routing supporting statistical, financial, ML, and time-series analysis
- 🔍 **File Preview**: Advanced preview system for multiple file formats
- ⚡ **Performance Optimized**: Data preprocessing optimized for business analytics workloads

---

## ✨ **Core Features**

### **🧠 Multi-Agent Intelligence**
- **Data Analyst Agent**: Statistical analysis and data manipulation
- **RAG Specialist Agent**: Document analysis and information retrieval
- **Reviewer Agent**: Quality assurance and validation
- **Visualizer Agent**: Interactive chart and graph generation
- **Reporter Agent**: Professional report compilation

### **🔌 Specialized Plugin Agents**
1. **📊 Statistical Agent**: Advanced statistical analysis, hypothesis testing, t-tests, ANOVA, correlation
2. **📈 Time Series Agent**: ARIMA forecasting, trend analysis, seasonality detection, exponential smoothing
3. **💰 Financial Agent**: 
    - **Liquidity & Efficiency**: Current/Quick ratios, Asset/Inventory turnover
    - **Profitability**: ROI, ROE, ROA calculations
    - **Forecasting**: Revenue/Profit projections
    - **Customer Analysis**: CLV proxy, segmentation, churn risk
4. **🤖 ML Insights Agent**: 
    - **Classification**: Decision Trees, Random Forest (Auto-detected targets)
    - **Regression**: Predictive modeling for continuous variables
    - **Pattern Recognition**: K-means clustering, Association rules, Feature importance ranking
5. **🗄️ SQL Agent ("Chat with your Data")**: 
    - **Natural Language to SQL**: Convert questions ("Show top 5 sales") into safe SQL queries
    - **Format Support**: Upload CSV/Excel and query them immediately via in-memory SQL engine
    - **External DBs**: Connect to PostgreSQL/MySQL (Safe Mode: Blocks destructive DROP/DELETE ops)

### **🤖 Machine Learning & Statistical Capabilities** (NEW!)
- **Clustering**: K-means, DBSCAN, Hierarchical clustering
- **Classification**: Random Forest, Logistic Regression, Decision Trees, SVM
- **Regression**: Linear, Ridge, Lasso, Polynomial regression
- **Dimensionality Reduction**: PCA, Truncated SVD, Feature importance
- **Statistical Tests**: T-tests, ANOVA, Chi-square, Pearson/Spearman correlation
- **Time Series**: ARIMA, Exponential smoothing, Seasonal decomposition
- **Anomaly Detection**: Z-score method, Isolation Forest
- **Model Evaluation**: Accuracy, Precision, Recall, F1, ROC curves

### **📁 File Support**
- **Structured Data**: CSV, JSON, XLSX, XLS
- **Documents**: PDF, DOCX, PPTX, RTF, TXT
- **Raw Text Input**: Direct text analysis capability
- **Database Files**: SQL, SQLite, DB files

### **🎨 Advanced UI Features**
- **File Preview**: Popup previews for all supported file types
- **Tabbed Results**: Analysis, Review Insights, Charts, Technical Details
- **Collapsible Sections**: Clean, organized result presentation
- **Download Reports**: Export analysis results in multiple formats
- **Real-time Updates**: Live progress tracking and status updates

---

## 🏗️ **Architecture**

### **Technology Stack**
- **Backend**: FastAPI, Python 3.11+
- **Frontend**: React 18, Next.js 14, TypeScript
- **AI Framework**: Custom Plugin System with self-correction
- **LLM**: Ollama (local inference)
- **Database**: ChromaDB for vector storage
- **UI**: Tailwind CSS, Radix UI components
- **Visualization**: Plotly, Recharts

### **System Architecture**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React UI      │    │   FastAPI        │    │   AI Agents     │
│   - File Upload │◄──►│   - REST API     │◄──►│   - Data Analyst│
│   - Results     │    │   - WebSocket    │    │   - RAG Spec.   │
│   - Preview     │    │   - File Mgmt    │    │   - Reviewer    │
└─────────────────┘    └──────────────────┘    │   - Visualizer  │
                                                │   - Reporter    │
┌─────────────────┐    ┌──────────────────┐    └─────────────────┘
│   Plugin System │    │   Ollama LLM     │    ┌─────────────────┐
│   - Statistical │◄──►│   - Local Inf.   │    │   Data Storage  │
│   - Time Series │    │   - Model Mgmt   │    │   - ChromaDB    │
│   - Financial   │    │   - Privacy      │    │   - File Cache  │
│   - ML Insights │    └──────────────────┘    │   - Logs        │
│   - SQL Agent   │                            └─────────────────┘
└─────────────────┘
```

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.11 or higher
- Node.js 18 or higher
- Ollama installed and running
- 8GB+ RAM recommended

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/KOTAHARSHA25/nexus-llm-analytics.git
   cd nexus-llm-analytics
   ```

2. **Set up Python environment**
   ```bash
   python -m venv env
   # Windows
   env\Scripts\activate
   # Linux/Mac
   source env/bin/activate
   
   pip install -r requirements.txt
   ```

3. **Install and configure Ollama**
   ```bash
   # Install Ollama (visit https://ollama.ai for installation)
   ollama pull llama3.1:8b
   ollama pull phi3:latest
   ```

4. **Set up frontend**
   ```bash
   cd src/frontend
   npm install
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your preferences
   ```

### **Running the Application**

1. **Start the backend**
   ```bash
   cd src/backend
   python -m uvicorn main:app --reload --port 8000
   ```

2. **Start the frontend**
   ```bash
   cd src/frontend
   npm run dev
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - API Documentation: http://localhost:8000/docs

---

## 💡 **Usage Examples**

### **Basic Statistical Analysis**
```
Query: "Analyze the correlation between sales and marketing spend"
→ Statistical Agent performs Pearson/Spearman correlation analysis
→ Results include correlation coefficients, p-values, confidence intervals
```

### **Machine Learning - Clustering**
```
Query: "Perform K-means clustering with 3 clusters based on sales and revenue"
→ ML Insights Agent applies K-means algorithm
→ Results: Optimal clusters, silhouette scores, cluster centers, visualization
```

### **Machine Learning - Classification**
```
Query: "Build a random forest classifier to predict if revenue > 5000"
→ ML Insights Agent trains Random Forest model
→ Results: Accuracy, precision, recall, feature importance, confusion matrix
```

### **Machine Learning - Dimensionality Reduction**
```
Query: "Apply PCA to reduce dimensions of sales, revenue, price to 2 components"
→ ML Insights Agent performs Principal Component Analysis
→ Results: Explained variance ratios, component loadings, transformed data
```

### **Statistical Tests**
```
Query: "Run a t-test to compare sales between Region A and Region B"
→ Statistical Agent performs independent t-test
→ Results: T-statistic, p-value, means, confidence intervals, significance
```

### **Time Series Forecasting**
```
Query: "Build ARIMA model to forecast next 12 months of sales"
→ Time Series Agent applies ARIMA modeling with auto-selection
→ Results: Forecasts with confidence intervals, trend decomposition, accuracy metrics
```

### **Financial Analysis**
```
Query: "Calculate our company's financial health metrics"
→ Financial Agent computes profitability ratios, growth metrics, ROI
→ Generates comprehensive financial dashboard
```

### **Anomaly Detection**
```
Query: "Identify anomalies in revenue using z-score method (threshold 3)"
→ ML Insights Agent applies z-score anomaly detection
→ Results: Anomalous data points, z-scores, visualization of outliers
```

### **Regression Analysis**
```
Query: "Create a linear regression model to predict revenue from sales and price"
→ ML Insights Agent trains linear regression model
→ Results: Coefficients, R², p-values, residual plots, prediction intervals
```

### **SQL Analysis (Chat with Data)**
```
Query: "Show top 5 customers by total revenue from the uploaded data"
→ SQL Agent loads CSV/Excel into secure in-memory database
→ Generates safe SQL: `SELECT customer_id, SUM(revenue) FROM analyzed_data GROUP BY customer_id ORDER BY 2 DESC LIMIT 5`
→ Returns exact results directly from your file
```

---

## 🔧 **Advanced Features**

### **Retry Logic with Fallback**
- **2 automatic retries** for failed analyses
- **Model switching**: Falls back to review model if primary fails
- **Graceful degradation**: Continues with available agents

### **Review Model Integration**
- **Dual-model validation**: Primary model + review model
- **Quality assessment**: Automated result validation
- **Feedback loop**: Review insights improve future analyses

### **Plugin Extensibility**
- **Hot-reloadable plugins**: Add new agents without restart
- **Configuration-driven**: JSON-based plugin configuration
- **Capability routing**: Intelligent agent selection based on query

### **Enhanced Error Handling**
- **User-friendly messages**: Clear, actionable error descriptions
- **Detailed logging**: Comprehensive system monitoring
- **Recovery mechanisms**: Multiple fallback strategies

---

## 📊 **Plugin System Deep Dive**

### **How Plugins Work**
1. **Auto-Discovery**: System scans `/plugins` directory for new agents
2. **Capability Matching**: Each plugin declares what it can handle
3. **Confidence Scoring**: Plugins rate their ability to handle queries (0.0-1.0)
4. **Intelligent Routing**: Best plugin is selected automatically
5. **Fallback Chain**: If plugin fails, system uses built-in agents

### **Plugin Benefits**
- **Specialized Expertise**: Domain-specific algorithms and calculations
- **Better Accuracy**: Precise computations vs. LLM approximations
- **Faster Performance**: Optimized algorithms for specific tasks
- **Extensibility**: Easy to add new analytical capabilities

### **Current Plugin Capabilities**
- **Statistical**: 15+ statistical tests and methods
- **Time Series**: ARIMA, seasonal decomposition, stationarity tests
- **Financial**: 20+ business metrics and financial ratios
- **ML Insights**: Clustering, PCA, anomaly detection, pattern recognition
- **SQL**: Multi-database queries, schema analysis, optimization

---

## 🔐 **Privacy & Security**

### **Local-First Architecture**
- ✅ All LLM inference runs locally via Ollama
- ✅ No data transmitted to external APIs
- ✅ Complete control over your data
- ✅ Offline capability for sensitive environments

### **Security Features**
- 🔒 Sandboxed code execution
- 🔒 Input validation and sanitization
- 🔒 File upload restrictions
- 🔒 Secure file handling

---

## 📈 **Performance**

### **System Requirements**
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, 8-core CPU
- **Storage**: 10GB+ for models and data

### **Performance Optimizations**
- **Lazy Loading**: Components load on-demand
- **Caching**: Intelligent result caching
- **Streaming**: Real-time result updates
- **Memory Management**: Optimized for long-running analyses

---

## 🛠️ **Development**

### **Project Structure**
```
nexus-llm-analytics/
├── src/
│   ├── backend/
│   │   ├── api/            # REST API endpoints
│   │   ├── agents/         # Core AI agents
│   │   ├── core/           # System utilities
│   │   └── main.py         # FastAPI application
│   └── frontend/
│       ├── app/            # Next.js pages
│       ├── components/     # React components
│       └── hooks/          # Custom hooks
├── plugins/                # Specialized AI agents
├── docs/                   # Documentation
├── tests/                  # Test suites
└── data/                   # Data storage
```

### **Adding New Plugins**
1. Create new Python file in `/plugins`
2. Inherit from `BasePluginAgent`
3. Implement required methods: `get_metadata()`, `can_handle()`, `execute()`
4. System automatically discovers and loads the plugin

### **Testing**
```bash
pytest tests/                    # Run all tests
pytest tests/test_plugins.py     # Test plugin system
pytest tests/test_api.py         # Test API endpoints
```

---

## 🤝 **Contributing**

### **Development Setup**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make changes and add tests
5. Run tests: `pytest`
6. Submit pull request

### **Plugin Development**
- Follow the `BasePluginAgent` interface
- Add comprehensive error handling
- Include metadata and capability declarations
- Write tests for your plugin

---

## 📋 **Roadmap**

### **Upcoming Features**
- [ ] **Real-time Collaboration**: Multi-user analysis sessions
- [ ] **Advanced Visualizations**: Interactive dashboards
- [ ] **Export Formats**: PDF, Excel, PowerPoint reports
- [ ] **API Integration**: External data source connectors
- [ ] **Model Marketplace**: Community plugin sharing

### **Long-term Vision**
- Enterprise deployment options
- Advanced security features
- Distributed computing support
- Mobile application

---

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **CrewAI** - Multi-agent framework
- **Ollama** - Local LLM inference
- **FastAPI** - High-performance API framework
- **React/Next.js** - Modern frontend framework

---

## 📞 **Support**

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/KOTAHARSHA25/nexus-llm-analytics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/KOTAHARSHA25/nexus-llm-analytics/discussions)

---

<div align="center">

**Made with ❤️ for data scientists, analysts, and AI enthusiasts**

⭐ **Star this repository if you find it useful!** ⭐

</div>