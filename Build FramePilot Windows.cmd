@echo off
setlocal
cd /d "%~dp0"
if not exist .venv-windows\Scripts\python.exe (
  py -3.13 -m venv .venv-windows
  if errorlevel 1 goto failed
)
set "FP_PYTHON=%~dp0.venv-windows\Scripts\python.exe"
"%FP_PYTHON%" -c "import sys,sysconfig;sys.exit(0 if sysconfig.get_platform()=='win-amd64' else 1)"
if errorlevel 1 goto failed
"%FP_PYTHON%" -m pip install -e ".[build,dev]"
if errorlevel 1 goto failed
"%FP_PYTHON%" tools\windows_release.py
if errorlevel 1 goto failed
echo Built: dist\FramePilot\FramePilot.exe and versioned Windows ZIP in dist
exit /b 0
:failed
echo Build or self-test failed. A verified Windows release has not been created.
exit /b 1
