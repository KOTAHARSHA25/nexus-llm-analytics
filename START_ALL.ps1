# ============================================================================
# Nexus LLM Analytics - ONE-CLICK STARTUP (PowerShell Version)
# ============================================================================
# Starts all three services in separate terminal windows:
#   1. Ollama (LLM Server)
#   2. Backend (FastAPI on port 8000)
#   3. Frontend (Next.js on port 3000)
# ============================================================================

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         NEXUS LLM ANALYTICS - ONE-CLICK STARTUP                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ============================================================================
# STEP 1: Check if Ollama is installed
# ============================================================================
Write-Host "[1/6] Checking Ollama installation..." -ForegroundColor Yellow
$OllamaPath = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $OllamaPath) {
    Write-Host "❌ ERROR: Ollama not found in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Ollama from: https://ollama.com/download" -ForegroundColor Yellow
    Write-Host "After installation, restart this script."
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "✅ Ollama found at: $($OllamaPath.Source)" -ForegroundColor Green

# ============================================================================
# STEP 2: Start Ollama Service
# ============================================================================
Write-Host ""
Write-Host "[2/6] Starting Ollama service..." -ForegroundColor Yellow

$OllamaRunning = Get-Process -Name ollama -ErrorAction SilentlyContinue
if ($OllamaRunning) {
    Write-Host "ℹ️  Ollama already running (PID: $($OllamaRunning.Id))" -ForegroundColor Cyan
} else {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Starting Ollama LLM Server...' -ForegroundColor Green; ollama serve" -WindowStyle Normal
    Write-Host "✅ Ollama service started in new window" -ForegroundColor Green
    Start-Sleep -Seconds 3
}

# ============================================================================
# STEP 3: Pull required LLM models (if not already present)
# ============================================================================
Write-Host ""
Write-Host "[3/6] Checking LLM models..." -ForegroundColor Yellow

# Check for llama3.1:8b
$LlamaModel = ollama list 2>$null | Select-String "llama3.1:8b"
if (-not $LlamaModel) {
    Write-Host "⚠️  llama3.1:8b not found - pulling model (this may take a few minutes)..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Pulling llama3.1:8b model...' -ForegroundColor Yellow; ollama pull llama3.1:8b; Write-Host 'Model pulled successfully!' -ForegroundColor Green; Start-Sleep -Seconds 3" -WindowStyle Normal
} else {
    Write-Host "✅ llama3.1:8b ready" -ForegroundColor Green
}

# Check for phi3:mini
$PhiModel = ollama list 2>$null | Select-String "phi3:mini"
if (-not $PhiModel) {
    Write-Host "ℹ️  phi3:mini not found (optional) - you can pull it later with: ollama pull phi3:mini" -ForegroundColor Cyan
} else {
    Write-Host "✅ phi3:mini ready" -ForegroundColor Green
}

# ============================================================================
# STEP 4: Kill any existing backend/frontend processes
# ============================================================================
Write-Host ""
Write-Host "[4/6] Cleaning up old processes..." -ForegroundColor Yellow

# Kill backend on port 8000
$BackendPID = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($BackendPID) {
    Write-Host "  Stopping old backend (PID $BackendPID)..." -ForegroundColor Gray
    Stop-Process -Id $BackendPID -Force -ErrorAction SilentlyContinue
}

# Kill frontend on port 3000
$FrontendPID = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($FrontendPID) {
    Write-Host "  Stopping old frontend (PID $FrontendPID)..." -ForegroundColor Gray
    Stop-Process -Id $FrontendPID -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2
Write-Host "✅ Cleanup complete" -ForegroundColor Green

# ============================================================================
# STEP 5: Start Backend Server
# ============================================================================
Write-Host ""
Write-Host "[5/6] Starting Backend (FastAPI)..." -ForegroundColor Yellow

$BackendCmd = "Set-Location '$ScriptDir'; Write-Host 'Starting Nexus Backend API...' -ForegroundColor Green; python -m uvicorn src.backend.main:app --reload --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCmd -WindowStyle Normal

Write-Host "✅ Backend starting in new window..." -ForegroundColor Green
Start-Sleep -Seconds 8

# ============================================================================
# STEP 6: Start Frontend Server
# ============================================================================
Write-Host ""
Write-Host "[6/6] Starting Frontend (Next.js)..." -ForegroundColor Yellow

$FrontendCmd = "Set-Location '$ScriptDir\src\frontend'; Write-Host 'Starting Nexus Frontend UI...' -ForegroundColor Green; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $FrontendCmd -WindowStyle Normal

Write-Host "✅ Frontend starting in new window..." -ForegroundColor Green
Start-Sleep -Seconds 3

# ============================================================================
# SUCCESS MESSAGE
# ============================================================================
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                  ✅ ALL SERVICES STARTED!                       ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "📡 Ollama LLM Server:   " -NoNewline; Write-Host "http://localhost:11434" -ForegroundColor Cyan
Write-Host "🔧 Backend API:         " -NoNewline; Write-Host "http://localhost:8000" -ForegroundColor Cyan
Write-Host "🌐 Frontend UI:         " -NoNewline; Write-Host "http://localhost:3000" -ForegroundColor Cyan
Write-Host "📚 API Docs:            " -NoNewline; Write-Host "http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""

# Get local IP addresses
Write-Host "⚡ Your local network IP addresses:" -ForegroundColor Yellow
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -eq "Dhcp" } | ForEach-Object {
    Write-Host "   Access from network: " -NoNewline
    Write-Host "http://$($_.IPAddress):3000" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "ℹ️  TIP: Keep the terminal windows open - services run there" -ForegroundColor DarkGray
Write-Host "ℹ️  TIP: Press Ctrl+C in each terminal to stop individual services" -ForegroundColor DarkGray
Write-Host "ℹ️  TIP: Run STOP_ALL.ps1 to shutdown all services at once" -ForegroundColor DarkGray
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     Press any key to close this launcher (services continue)  ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

Read-Host ""
