@echo off
echo ============================================
echo  LiteLLM Proxy - Install (uv)
echo ============================================
echo.

REM Check uv
where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv is not installed or not in PATH.
    echo.
    echo Install uv with:
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo Or visit: https://docs.astral.sh/uv/getting-started/installation/
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('uv --version 2^>^&1') do echo Found: %%v
echo.

REM Sync dependencies
echo Installing dependencies...
uv sync
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
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
