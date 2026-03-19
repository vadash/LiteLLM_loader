@echo off
python "%~dp0litellm_ctl.py" stop %*
timeout 5
