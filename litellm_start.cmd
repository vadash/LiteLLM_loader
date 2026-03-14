@echo off
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set "%%a=%%b"
litellm --config config.yaml
