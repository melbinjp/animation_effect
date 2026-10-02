@echo off
setlocal
cd /d "%~dp0native"

echo [Linearty] Checking Python environment...
python -c "import fastapi, uvicorn" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [Linearty] Installing required dependencies...
    python -m pip install -r requirements.txt -r requirements-webui.txt
)

echo [Linearty] Starting Linearty Studio Server...
python webui.py --host 127.0.0.1 --port 8765
if %ERRORLEVEL% neq 0 (
    pause
)
