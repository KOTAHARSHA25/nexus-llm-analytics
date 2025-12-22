@echo off
cd /d "%~dp0"
echo Starting Nexus LLM Analytics Backend...
echo.
python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
pause
