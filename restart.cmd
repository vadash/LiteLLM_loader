@echo off
python "%~dp0litellm_ctl.py" restart %*
timeout 5
