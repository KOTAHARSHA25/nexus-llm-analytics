# Quick Start Guide - Nexus LLM Analytics

## 🚀 Get Started in 3 Minutes

### Step 1: Prerequisites Check
Ensure you have:
- ✅ **Python 3.8+** (`python --version`)
- ✅ **Node.js 18+** (`node --version`)
- ✅ **Ollama installed** (https://ollama.ai)

### Step 2: Download Required Models
```bash
# Essential models for full functionality
ollama pull llama3.1:8b      # Primary analysis (4.7GB)
ollama pull phi3:mini        # Review agent (2.3GB)
ollama pull nomic-embed-text # Embeddings (274MB)
```

### Step 3: Launch the Platform
```bash
# Backend (Terminal 1)
cd src/backend
python -m uvicorn main:app --reload

# Frontend (Terminal 2)
cd src/frontend
npm run dev
```

That's it! Your platform will be available at:
- 🌐 **Frontend**: http://localhost:3000
- 📊 **Backend API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs

---

## 🎯 First Analysis in 2 Minutes

### 1. Upload Your Data
- Drag & drop files (CSV, JSON, PDF, TXT)
- Or use the sample data included

### 2. Ask Natural Language Questions
Try these examples:
- *"Analyze the trends in this data"*
- *"Create visualizations showing key patterns"*
- *"Generate a comprehensive report"*
- *"What are the main insights from this dataset?"*

### 3. Get AI-Powered Results
- Interactive charts and visualizations
- Detailed analysis and insights
- Professional PDF/Excel reports
- All processed locally on your machine

---



## 🆘 Quick Troubleshooting

### Ollama Not Working?
```bash
# Start Ollama service
ollama serve

# Check if models are available
ollama list
```

### Port Already in Use?
```bash
# Check what's using the ports
netstat -ano | findstr :8000
netstat -ano | findstr :3000

# Kill processes if needed
taskkill /PID <process_id> /F
```

### Permission Errors?
```bash
# Run as administrator or fix permissions
pip install --user -r requirements.txt
```

---

## ✨ What Makes This Special?

🤖 **True AI Agents**: CrewAI-powered collaborative AI  
🔒 **Enterprise Security**: Local processing + sandboxed execution  
📊 **Advanced Viz**: Interactive Plotly charts  
🧠 **Natural Language**: Ask questions in plain English  
📄 **Professional Reports**: PDF & Excel generation  
🏠 **Privacy First**: Your data never leaves your machine  

---

**Ready to revolutionize your data analysis? Launch the platform now!** 🚀