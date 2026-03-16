@echo off
echo ========================================
echo Nexus LLM Analytics - Complete Restart
echo ========================================
echo.

REM Kill only the Nexus backend process (uvicorn on port 8000), not all Python processes
echo Stopping any running backend on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Killing PID %%a on port 8000...
    taskkill /F /PID %%a 2>nul
)
timeout /t 2 /nobreak >nul

REM Navigate to project directory
cd /d "%~dp0"

REM Start backend in new window
echo.
echo Starting Backend Server...
start "Nexus Backend" cmd /k "python -m uvicorn src.backend.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait for backend to initialize
echo Waiting for backend to start (10 seconds)...
timeout /t 10 /nobreak

REM Start frontend in new window
echo.
echo Starting Frontend Server...
cd src\frontend
start "Nexus Frontend" cmd /k "npm run dev"

echo.
echo ========================================
echo Both servers starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo ========================================
echo.
echo Press any key to close this window...
pause >nul
