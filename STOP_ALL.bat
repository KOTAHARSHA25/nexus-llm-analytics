@echo off
REM ============================================================================
REM Nexus LLM Analytics - ONE-CLICK SHUTDOWN
REM ============================================================================
REM Gracefully stops all three services:
REM   1. Frontend (Next.js on port 3000)
REM   2. Backend (FastAPI on port 8000)
REM   3. Ollama (LLM Server)
REM ============================================================================

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         NEXUS LLM ANALYTICS - SHUTDOWN ALL SERVICES            ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

REM ============================================================================
REM STEP 1: Stop Frontend (port 3000)
REM ============================================================================
echo [1/3] Stopping Frontend (Next.js)...
set FOUND_FRONTEND=0
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do (
    echo   Killing frontend process (PID %%a)...
    taskkill /F /PID %%a 2>nul
    set FOUND_FRONTEND=1
)
if %FOUND_FRONTEND%==0 (
    echo ℹ️  Frontend not running
) else (
    echo ✅ Frontend stopped
)

REM ============================================================================
REM STEP 2: Stop Backend (port 8000)
REM ============================================================================
echo.
echo [2/3] Stopping Backend (FastAPI)...
set FOUND_BACKEND=0
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo   Killing backend process (PID %%a)...
    taskkill /F /PID %%a 2>nul
    set FOUND_BACKEND=1
)
if %FOUND_BACKEND%==0 (
    echo ℹ️  Backend not running
) else (
    echo ✅ Backend stopped
)

REM ============================================================================
REM STEP 3: Stop Ollama Service
REM ============================================================================
echo.
echo [3/3] Stopping Ollama LLM Server...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo   Stopping Ollama service...
    taskkill /F /IM ollama.exe 2>nul
    echo ✅ Ollama stopped
) else (
    echo ℹ️  Ollama not running
)

REM Also close any open terminal windows with "Nexus" in title
echo.
echo Closing terminal windows...
taskkill /FI "WINDOWTITLE eq Nexus*" /F 2>nul

timeout /t 2 /nobreak >nul

REM ============================================================================
REM SUCCESS MESSAGE
REM ============================================================================
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║              ✅ ALL SERVICES STOPPED SUCCESSFULLY               ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo 🛑 Frontend:  Stopped
echo 🛑 Backend:   Stopped
echo 🛑 Ollama:    Stopped
echo.
echo You can restart all services by running START_ALL.bat
echo.
pause
