@echo off
cd /d "%~dp0"
if exist "%~dp0output\executable\PinguBall.exe" (
    "%~dp0output\executable\PinguBall.exe" --lab
) else (
    "%~dp0.venv\Scripts\python.exe" "%~dp0main.py" --lab
)
if errorlevel 1 pause
