# 🚀 Nexus LLM Analytics - Distribution Package

## 📦 What's Included

This is a **clean distribution** containing only the essential source code files.

### ✅ Included:
- **src/** - Complete source code (backend + frontend)
- **docs/** - Documentation
- **config/** - Configuration files
- **scripts/** - Launch scripts
- **tests/** - Test files
- **data/samples/** - Sample data files
- **requirements.txt** - Python dependencies
- **package.json** - Node.js dependencies
- **README.md** - Main documentation
- **LICENSE** - Project license

### ❌ Excluded (Development/Build Files):
- node_modules/ - Install with `npm install`
- env/ - Create virtual environment with `python -m venv env`
- .next/ - Built automatically with `npm run build`
- logs/ - Generated at runtime
- __pycache__/ - Python cache
- .git/ - Git repository
- data/uploads/ - User uploaded files
- Reference folders (lida-main, src2, nexus-llm-analytics-dist)

## 🚀 Quick Start

### 1. **Setup Python Environment**
```bash
python -m venv env
env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. **Setup Frontend**
```bash
cd src/frontend
npm install
```

### 3. **Install Ollama & Models**
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.1:8b
ollama pull phi3:latest
```

### 4. **Run the Application**
```bash
# Option 1: Use launch script (recommended)
python scripts/launch.py

# Option 2: Manual start
# Terminal 1 - Backend
cd src/backend
python -m uvicorn main:app --reload

# Terminal 2 - Frontend
cd src/frontend
npm run dev
```

### 5. **Access**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📝 System Requirements

- Python 3.11+
- Node.js 18+
- 8GB+ RAM
- Windows/Linux/Mac

## 📚 Documentation

See `docs/` folder for:
- QUICK_START.md - Detailed setup guide
- TECHNICAL_ARCHITECTURE_OVERVIEW.md - System architecture
- SMART_MODEL_SELECTION.md - Model configuration
- PROJECT_STRUCTURE.md - Code organization

## 🎯 Features

✨ Multi-Agent AI System  
✨ Local-First (Privacy-Focused)  
✨ Plugin Architecture  
✨ Smart Model Selection  
✨ Advanced Analytics (Statistical, ML, Financial, Time Series)  
✨ RAG Document Analysis  
✨ Modern React UI  

## 📧 Support

For issues or questions, check the documentation in the `docs/` folder.

---

**Made with ❤️ for Data Scientists & Analysts**

Generated: 2025-10-15 18:39:21
