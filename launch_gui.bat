@echo off
title OUROBOROS GUI Launcher
echo.
echo  ==========================================
echo   OUROBOROS -- self-learning chess bot GUI
echo  ==========================================
echo.

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found.
    echo.
    echo  Install Python 3.10 or later from https://python.org
    echo  Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

:: Install requests if missing (silent if already installed)
echo  Checking dependencies...
python -c "import requests" >nul 2>&1
if errorlevel 1 (
    echo  Installing requests...
    python -m pip install requests --quiet
    if errorlevel 1 (
        echo.
        echo  WARNING: Could not install requests automatically.
        echo  Run manually:  pip install requests
        echo.
    )
)

echo  Starting GUI...
echo.
python ouroboros_gui.py

if errorlevel 1 (
    echo.
    echo  The GUI exited with an error. See above for details.
    pause
)
