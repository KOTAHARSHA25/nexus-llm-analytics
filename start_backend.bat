@echo off
cd /d "%~dp0"
echo Starting Nexus LLM Analytics Backend...
echo.
python -m uvicorn src.backend.main:app --reload
pause
