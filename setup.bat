@echo off
REM =========================================================
REM  setup.bat — One-shot setup for ColdBridge on Windows
REM  Run once after cloning the repo.
REM =========================================================

echo === ColdBridge Setup ===
echo.

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found. Install Python 3.11+ from https://python.org
    exit /b 1
)
echo [OK] Python found
python --version

REM Check Docker
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Docker not running. Start Docker Desktop and try again.
    exit /b 1
)
echo [OK] Docker running

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing Python dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo [OK] Dependencies installed

REM Create output directories
if not exist results mkdir results
if not exist logs   mkdir logs
echo [OK] Output directories ready

REM Build Docker images
echo.
echo Building Docker worker images (this may take a few minutes)...
call scripts\build_images.bat
if %ERRORLEVEL% neq 0 ( echo ERROR: Image build failed & exit /b 1 )

echo.
echo =========================================================
echo  Setup complete!
echo.
echo  Activate the environment:
echo    .venv\Scripts\activate
echo.
echo  Run baseline experiment:
echo    python -m experiments.run_experiment run --mode baseline --invocations 30
echo.
echo  Run with Module A:
echo    python -m experiments.run_experiment run --mode module_a --invocations 30
echo.
echo  Compare all runs:
echo    python -m experiments.run_experiment compare
echo =========================================================
