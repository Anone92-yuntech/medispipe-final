@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "PIP_CACHE_DIR=%~dp0.venv\pip-cache"
set "UV_CACHE_DIR=%~dp0.venv\uv-cache"
if exist ".venv\Scripts\python.exe" goto install
where uv >nul 2>nul
if errorlevel 1 goto python_venv
uv venv --python 3.12 .venv
if errorlevel 1 goto failed
goto install
:python_venv
python -m venv .venv
if errorlevel 1 goto failed
:install
where uv >nul 2>nul
if errorlevel 1 goto pip_install
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
if errorlevel 1 goto failed
goto model
:pip_install
".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
:model
".venv\Scripts\python.exe" -B environment_check.py --prepare
if errorlevel 1 goto failed
exit /b 0
:failed
echo Setup failed. Check the error above and your internet connection.
pause
exit /b 1
