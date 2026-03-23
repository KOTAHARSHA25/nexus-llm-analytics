@echo off
REM ============================================================================
REM Nexus LLM Analytics - ONE-CLICK STARTUP
REM ============================================================================
REM Starts all three services in separate terminal windows:
REM   1. Ollama (LLM Server)  — OFFLINE mode only
REM   2. Backend (FastAPI on port 8000)
REM   3. Frontend (Next.js on port 3000)
REM ============================================================================

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         NEXUS LLM ANALYTICS - ONE-CLICK STARTUP                ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

REM ============================================================================
REM Read NEXUS_MODE from .env  (defaults to offline)
REM ============================================================================
set NEXUS_MODE=offline
if exist ".env" (
    for /f "usebackq tokens=1,2 delims==" %%A in (".env") do (
        if /i "%%A"=="NEXUS_MODE" set NEXUS_MODE=%%B
    )
)
REM Trim any trailing whitespace/carriage-return from the value
for /f "tokens=1" %%X in ("%NEXUS_MODE%") do set NEXUS_MODE=%%X

if /i "%NEXUS_MODE%"=="online" (
    echo ℹ️  NEXUS_MODE=online — Ollama will NOT be started (using cloud APIs)
    echo ℹ️  Skipping Steps 1-3 ^(Ollama + model checks^)
    goto :skip_ollama
) else (
    echo ℹ️  NEXUS_MODE=offline — will start Ollama local models
)

REM ============================================================================
REM STEP 1: Check if Ollama is installed
REM ============================================================================
echo [1/6] Checking Ollama installation...

REM Check if ollama is in PATH
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Ollama found in PATH
    set OLLAMA_CMD=ollama
    goto :ollama_found
)

REM Check common installation locations
set OLLAMA_CMD=
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set OLLAMA_CMD=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
    echo ✅ Ollama found at %LOCALAPPDATA%\Programs\Ollama\
    goto :ollama_found
)
if exist "%PROGRAMFILES%\Ollama\ollama.exe" (
    set OLLAMA_CMD=%PROGRAMFILES%\Ollama\ollama.exe
    echo ✅ Ollama found at %PROGRAMFILES%\Ollama\
    goto :ollama_found
)
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe" (
    set OLLAMA_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe
    echo ✅ Ollama found at C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\
    goto :ollama_found
)

REM Ollama not found
echo ❌ ERROR: Ollama not found
echo.
echo Please either:
echo   1. Install Ollama from: https://ollama.com/download
echo   2. Or add Ollama to your PATH by restarting your computer after installation
echo.
pause
exit /b 1

:ollama_found

REM ============================================================================
REM STEP 2: Start Ollama Service
REM ============================================================================
echo.
echo [2/6] Starting Ollama service...

REM Check if Ollama is already running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo ℹ️  Ollama already running
) else (
    start "Nexus - Ollama LLM Server" cmd /k "echo Starting Ollama... && "%OLLAMA_CMD%" serve"
    echo ✅ Ollama service started
    timeout /t 3 /nobreak >nul
)

REM ============================================================================
REM STEP 3: Pull required LLM models (if not already present)
REM ============================================================================
echo.
echo [3/6] Checking LLM models...

REM Check for llama3.1:8b
"%OLLAMA_CMD%" list | find "llama3.1:8b" >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  llama3.1:8b not found - pulling model (this may take a few minutes)...
    start "Nexus - Pulling llama3.1" cmd /k ""%OLLAMA_CMD%" pull llama3.1:8b && echo Model pulled successfully && timeout /t 3"
) else (
    echo ✅ llama3.1:8b ready
)

REM Check for phi3:mini
"%OLLAMA_CMD%" list | find "phi3:mini" >nul 2>&1
if %errorlevel% neq 0 (
    echo ℹ️  phi3:mini not found (optional) - you can pull it later with the model path
) else (
    echo ✅ phi3:mini ready
)

REM ============================================================================
REM STEP 4: Kill any existing backend/frontend processes
REM ============================================================================
:skip_ollama
echo.
echo [4/6] Cleaning up old processes...

REM Kill backend on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo   Stopping old backend (PID %%a)...
    taskkill /F /PID %%a 2>nul
)

REM Kill frontend on port 3000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do (
    echo   Stopping old frontend (PID %%a)...
    taskkill /F /PID %%a 2>nul
)

timeout /t 2 /nobreak >nul
echo ✅ Cleanup complete

REM ============================================================================
REM STEP 5: Start Backend Server
REM ============================================================================
echo.
echo [5/6] Starting Backend (FastAPI)...
start "Nexus - Backend API" cmd /k "cd /d "%~dp0" && echo Starting Nexus Backend... && python -m uvicorn src.backend.main:app --reload --host 0.0.0.0 --port 8000"
echo ✅ Backend starting...
timeout /t 8 /nobreak >nul

REM ============================================================================
REM STEP 6: Start Frontend Server
REM ============================================================================
echo.
echo [6/6] Starting Frontend (Next.js)...
cd src\frontend
start "Nexus - Frontend UI" cmd /k "echo Starting Nexus Frontend... && npm run dev"
cd ..\..
echo ✅ Frontend starting...
timeout /t 3 /nobreak >nul

REM ============================================================================
REM SUCCESS MESSAGE
REM ============================================================================
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                  ✅ ALL SERVICES STARTED!                       ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
if /i "%NEXUS_MODE%"=="online" (
    echo 🌐 Mode:            ONLINE  ^(Cloud APIs — Groq / OpenRouter^)
    echo ℹ️  Ollama:          NOT started  ^(not needed in online mode^)
) else (
    echo 🖥️  Mode:            OFFLINE ^(Local Ollama models^)
    echo 📡 Ollama LLM:      http://localhost:11434
)
echo 🔧 Backend API:         http://localhost:8000
echo 🌐 Frontend UI:         http://localhost:3000
echo 📚 API Docs:            http://localhost:8000/docs
echo.
echo ⚡ Your local network IP addresses:
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    echo    Access from network: http://%%a:3000
)
echo.
echo ℹ️  TIP: Keep this window open to see status messages
echo ℹ️  TIP: Press Ctrl+C in each terminal to stop individual services
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║  Press any key to close this launcher (services keep running) ║
echo ╚════════════════════════════════════════════════════════════════╝
pause >nul
