@echo off
setlocal
cd /d "%~dp0"
if not exist .venv-windows\Scripts\python.exe (
  py -3.13 -m venv .venv-windows
  if errorlevel 1 exit /b 1
)
"%~dp0.venv-windows\Scripts\python.exe" -m pip install -e .
if errorlevel 1 exit /b 1
"%~dp0.venv-windows\Scripts\python.exe" -m framepilot %*
