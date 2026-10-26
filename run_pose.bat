@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist ".venv\Scripts\python.exe" goto setup
".venv\Scripts\python.exe" -B environment_check.py >nul 2>nul
if not errorlevel 1 goto run
:setup
call "%~dp0setup_env.bat"
if errorlevel 1 exit /b 1
:run
".venv\Scripts\python.exe" -B "%~dp0experiment.py" %*
set "experiment_exit=%errorlevel%"
echo.
if not "%experiment_exit%"=="0" echo Experiment reported an error. See above.
pause
exit /b %experiment_exit%
