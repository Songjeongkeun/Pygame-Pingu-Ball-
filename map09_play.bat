@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\verify_map09.py" --play
if errorlevel 1 pause
