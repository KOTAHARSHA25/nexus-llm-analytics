# ============================================================================
# Nexus LLM Analytics - ONE-CLICK SHUTDOWN (PowerShell Version)
# ============================================================================
# Gracefully stops all three services:
#   1. Frontend (Next.js on port 3000)
#   2. Backend (FastAPI on port 8000)
#   3. Ollama (LLM Server)
# ============================================================================

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Red
Write-Host "║         NEXUS LLM ANALYTICS - SHUTDOWN ALL SERVICES            ║" -ForegroundColor Red
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Red
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ============================================================================
# STEP 1: Stop Frontend (port 3000)
# ============================================================================
Write-Host "[1/3] Stopping Frontend (Next.js)..." -ForegroundColor Yellow
$FrontendPID = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($FrontendPID) {
    Write-Host "  Killing frontend process (PID $FrontendPID)..." -ForegroundColor Gray
    Stop-Process -Id $FrontendPID -Force -ErrorAction SilentlyContinue
    Write-Host "✅ Frontend stopped" -ForegroundColor Green
} else {
    Write-Host "ℹ️  Frontend not running" -ForegroundColor Cyan
}

# ============================================================================
# STEP 2: Stop Backend (port 8000)
# ============================================================================
Write-Host ""
Write-Host "[2/3] Stopping Backend (FastAPI)..." -ForegroundColor Yellow
$BackendPID = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($BackendPID) {
    Write-Host "  Killing backend process (PID $BackendPID)..." -ForegroundColor Gray
    Stop-Process -Id $BackendPID -Force -ErrorAction SilentlyContinue
    Write-Host "✅ Backend stopped" -ForegroundColor Green
} else {
    Write-Host "ℹ️  Backend not running" -ForegroundColor Cyan
}

# ============================================================================
# STEP 3: Stop Ollama Service
# ============================================================================
Write-Host ""
Write-Host "[3/3] Stopping Ollama LLM Server..." -ForegroundColor Yellow
$OllamaProcess = Get-Process -Name ollama -ErrorAction SilentlyContinue
if ($OllamaProcess) {
    Write-Host "  Stopping Ollama service (PID $($OllamaProcess.Id))..." -ForegroundColor Gray
    Stop-Process -Name ollama -Force -ErrorAction SilentlyContinue
    Write-Host "✅ Ollama stopped" -ForegroundColor Green
} else {
    Write-Host "ℹ️  Ollama not running" -ForegroundColor Cyan
}

# ============================================================================
# Cleanup: Close any PowerShell windows running Nexus services
# ============================================================================
Write-Host ""
Write-Host "Cleaning up terminal windows..." -ForegroundColor Yellow
Start-Sleep -Seconds 1

# ============================================================================
# SUCCESS MESSAGE
# ============================================================================
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              ✅ ALL SERVICES STOPPED SUCCESSFULLY               ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "🛑 Frontend:  Stopped" -ForegroundColor Gray
Write-Host "🛑 Backend:   Stopped" -ForegroundColor Gray
Write-Host "🛑 Ollama:    Stopped" -ForegroundColor Gray
Write-Host ""
Write-Host "You can restart all services by running START_ALL.ps1" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit"
