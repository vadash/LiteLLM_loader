@echo off
echo ============================================
echo  LiteLLM Proxy - Install
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo Download Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo Found: %%v
echo.

REM Install litellm
echo Installing litellm...
pip install litellm
if errorlevel 1 (
    echo [ERROR] Failed to install litellm.
    pause
    exit /b 1
)
echo.

REM Check .env in src directory
if not exist "%~dp0src\.env" (
    echo [WARNING] .env file not found. Creating template...
    (
        echo NVIDIA_API_BASE=https://your-api-base-url/
        echo NVIDIA_API_KEY=your-api-key-here
    ) > "%~dp0src\.env"
    echo.
    echo Created src\.env file. Edit it with your API credentials before starting.
)

echo.
echo ============================================
echo  Done! Usage:
echo    start.cmd    - Start proxy (no window)
echo    stop.cmd     - Stop proxy
echo    status.cmd   - Check status
echo    restart.cmd  - Restart proxy
echo ============================================
pause