@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0litellm_ctl.py" restart %*
timeout 5
