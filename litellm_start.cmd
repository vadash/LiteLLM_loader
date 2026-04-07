@echo off
"%~dp0.venv\Scripts\pythonw.exe" "%~dp0litellm_ctl.py" start %*
timeout 5
