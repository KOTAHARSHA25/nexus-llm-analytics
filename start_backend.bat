@echo off
cd /d "%~dp0"
echo Starting Nexus LLM Analytics Backend...
echo.
echo ============================================
echo  Backend will be accessible on the network
echo  Local: http://localhost:8000
echo  Network: http://YOUR_IP:8000
echo ============================================
echo.
echo Finding your IP address...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    echo   Possible: http://%%a:8000
)
echo.
python -m uvicorn src.backend.main:app --reload --host 0.0.0.0 --port 8000
pause
