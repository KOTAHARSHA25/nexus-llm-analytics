# 🚀 One-Click Startup Scripts

This folder contains convenient scripts to launch the entire Nexus LLM Analytics platform with **ONE CLICK** or **ONE COMMAND**.

---

## 📋 Available Scripts

### Windows Batch Files (.bat)
| Script | Description |
|--------|-------------|
| **START_ALL.bat** | Start all 3 services (Ollama + Backend + Frontend) in separate windows |
| **STOP_ALL.bat** | Stop all running services gracefully |
| restart_all.bat | Legacy script (only starts Backend + Frontend) |
| start_backend.bat | Start only the backend server |

### PowerShell Scripts (.ps1) — RECOMMENDED
| Script | Description |
|--------|-------------|
| **START_ALL.ps1** | 🌟 **BEST OPTION** - Colorful, modern, starts all 3 services |
| **STOP_ALL.ps1** | Stop all running services with better process detection |

---

## 🎯 Quick Start Guide

### Method 1: Double-Click (Easiest)
1. **Locate** `START_ALL.bat` in your project folder
2. **Double-click** it
3. **Wait** for 3 terminal windows to open (Ollama, Backend, Frontend)
4. **Open browser** to http://localhost:3000

### Method 2: PowerShell (Recommended for better visuals)
```powershell
# Navigate to project folder
cd "C:\Users\mitta\OneDrive\Documents\Major Project\Phase_2\nexus-llm-analytics-dist(Main)\nexus-llm-analytics-dist"

# Run startup script
.\START_ALL.ps1
```

### Method 3: Command Line
```cmd
cd "C:\Users\mitta\OneDrive\Documents\Major Project\Phase_2\nexus-llm-analytics-dist(Main)\nexus-llm-analytics-dist"
START_ALL.bat
```

---

## 🛑 Stopping All Services

### Option 1: Double-Click
Double-click `STOP_ALL.bat` or `STOP_ALL.ps1`

### Option 2: PowerShell
```powershell
.\STOP_ALL.ps1
```

### Option 3: Manual
Press `Ctrl+C` in each of the 3 terminal windows

---

## 🔍 What Each Script Does

### START_ALL Scripts (.bat or .ps1)

**Step-by-step execution:**
1. ✅ **Check Ollama** - Verifies Ollama is installed
2. 🚀 **Start Ollama** - Launches Ollama LLM server (if not running)
3. 📦 **Check Models** - Verifies llama3.1:8b is pulled (downloads if missing)
4. 🧹 **Cleanup** - Kills any old processes on ports 8000/3000
5. ⚙️ **Start Backend** - Launches FastAPI server (port 8000)
6. 🌐 **Start Frontend** - Launches Next.js UI (port 3000)
7. ✅ **Show URLs** - Displays all access URLs

**Opens 3 separate terminal windows:**
- **Window 1:** Ollama LLM Server (http://localhost:11434)
- **Window 2:** Backend API (http://localhost:8000)
- **Window 3:** Frontend UI (http://localhost:3000)

### STOP_ALL Scripts (.bat or .ps1)

**Step-by-step execution:**
1. 🛑 Stop Frontend (kills process on port 3000)
2. 🛑 Stop Backend (kills process on port 8000)
3. 🛑 Stop Ollama (kills ollama.exe process)
4. 🧹 Close terminal windows
5. ✅ Confirmation message

---

## 📊 Service Details

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| Ollama | 11434 | http://localhost:11434 | Local LLM inference engine |
| Backend | 8000 | http://localhost:8000 | FastAPI REST API + SSE streaming |
| Frontend | 3000 | http://localhost:3000 | Next.js web interface |
| API Docs | 8000 | http://localhost:8000/docs | Interactive Swagger documentation |

---

## ⚙️ First-Time Setup

### Prerequisites
1. **Ollama** must be installed
   - Download: https://ollama.com/download
   - After install, Ollama should be in PATH

2. **Python 3.11+** with dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. **Node.js 18+** with npm
   ```bash
   cd src/frontend
   npm install
   ```

### First Run
The first time you run `START_ALL`, it will:
- ⏳ **Pull llama3.1:8b** (~4.7GB download) - takes 5-15 minutes
- 📦 Install any missing Python/Node packages
- 🔧 Initialize ChromaDB vector store

**Subsequent runs** take only ~10-15 seconds to start all services!

---

## 🎨 PowerShell vs Batch Scripts

### Use PowerShell (.ps1) if:
✅ You want **colored output** and better UI  
✅ You want **better error messages**  
✅ You need **accurate process detection**  
✅ You're comfortable with modern Windows tools

### Use Batch (.bat) if:
✅ You want **maximum compatibility** (works everywhere)  
✅ You prefer **simple double-click** (no execution policy issues)  
✅ You're on an older Windows version

---

## 🐛 Troubleshooting

### PowerShell Execution Policy Error
If you get "cannot be loaded because running scripts is disabled":
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then run the script again
.\START_ALL.ps1
```

### Ollama Not Found
```
❌ ERROR: Ollama not found in PATH
```
**Fix:**
1. Install Ollama: https://ollama.com/download
2. Restart your terminal/computer
3. Verify: `ollama --version`

### Port Already in Use
```
Error: Port 8000 already in use
```
**Fix:**
1. Run `STOP_ALL.bat` or `STOP_ALL.ps1`
2. Wait 5 seconds
3. Run `START_ALL` again

### Model Download Hanging
If `ollama pull llama3.1:8b` gets stuck:
1. Press `Ctrl+C` in the Ollama window
2. Manually run: `ollama pull llama3.1:8b`
3. Wait for completion
4. Run `START_ALL` again

---

## 🔄 Development Workflow

### Typical Daily Workflow
```bash
# Morning - Start everything
.\START_ALL.ps1

# Work all day - services auto-reload on code changes
# Backend: uvicorn --reload automatically restarts
# Frontend: Next.js hot-reloads automatically

# Evening - Stop everything
.\STOP_ALL.ps1
```

### Auto-Reload Features
✅ **Backend:** `--reload` flag automatically restarts on `.py` file changes  
✅ **Frontend:** Next.js hot module replacement reloads on save  
❌ **Ollama:** Must restart manually if models change

---

## 🌐 Network Access

After starting, you'll see output like:
```
⚡ Your local network IP addresses:
   Access from network: http://192.168.1.105:3000
```

This means:
- **Your laptop:** http://localhost:3000
- **Other devices on WiFi:** http://192.168.1.105:3000 (replace with your IP)
- **Phone/Tablet:** Same URL - access from anywhere on your network

---

## 📁 Script Locations

```
nexus-llm-analytics-dist/
├── START_ALL.bat          ← Main startup (Windows batch)
├── START_ALL.ps1          ← Main startup (PowerShell - RECOMMENDED)
├── STOP_ALL.bat           ← Shutdown script (Windows batch)
├── STOP_ALL.ps1           ← Shutdown script (PowerShell - RECOMMENDED)
├── restart_all.bat        ← Legacy (doesn't start Ollama)
└── start_backend.bat      ← Backend only
```

---

## 💡 Tips & Tricks

### 1. Create Desktop Shortcuts
**Right-click** `START_ALL.bat` → **Send to** → **Desktop (create shortcut)**

Now you can start everything from your desktop with one click!

### 2. Pin to Taskbar
1. Right-click `START_ALL.bat`
2. Pin to Taskbar
3. Click taskbar icon to launch

### 3. Custom Startup Order
Edit `START_ALL.bat` and adjust `timeout` values:
```batch
timeout /t 5 /nobreak   REM Wait 5 seconds for Ollama
timeout /t 8 /nobreak   REM Wait 8 seconds for Backend
```

### 4. Silent Startup (No Windows)
Replace `start "Title" cmd /k` with:
```batch
start "Title" /min cmd /k   REM Minimized windows
```

### 5. Check What's Running
```powershell
# PowerShell - check services
Get-NetTCPConnection -LocalPort 3000,8000,11434 | Select LocalPort, State, OwningProcess
```

---

## ⚡ Performance Tips

1. **SSD Recommended:** Ollama model loading is I/O intensive
2. **8GB+ RAM:** llama3.1:8b needs ~6GB RAM
3. **Keep Services Running:** Startup time is ~10-15s, shutdown/restart wastes time
4. **Use PowerShell Scripts:** Better process management = faster startups

---

## 📝 Customization

### Change Ports
Edit the scripts to use different ports:

**Backend (default 8000):**
```batch
# In START_ALL.bat, change:
python -m uvicorn src.backend.main:app --reload --host 0.0.0.0 --port 8080
```

**Frontend (default 3000):**
```bash
# Edit src/frontend/.env.local
NEXT_PUBLIC_BACKEND_URL=http://localhost:8080
```

### Add More Services
To add another service (e.g., database):
1. Copy the backend startup section
2. Modify command and port
3. Add to cleanup section in STOP_ALL

---

## 🎓 Learning Resources

- **Ollama Docs:** https://github.com/ollama/ollama
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **Next.js Docs:** https://nextjs.org/docs

---

## ✅ Summary

**Startup:** Just run `START_ALL.ps1` or `START_ALL.bat`  
**Shutdown:** Run `STOP_ALL.ps1` or `STOP_ALL.bat`  
**Access:** http://localhost:3000

**That's it! No more juggling 3 terminals manually.** 🎉
