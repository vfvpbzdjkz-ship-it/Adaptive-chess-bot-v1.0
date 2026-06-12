@echo off
setlocal enabledelayedexpansion

set VENV_DIR=.venv
set REQUIREMENTS=requirements.txt
set MIN_MAJOR=3
set MIN_MINOR=11

echo === OUROBOROS Bootstrap ===

:: Find a suitable Python
set PYTHON=
for %%P in (python3.11 python3.12 python3.13 python3 python) do (
    where %%P >nul 2>&1 && (
        for /f "tokens=2 delims= " %%V in ('%%P --version 2^>^&1') do (
            for /f "tokens=1,2 delims=." %%A in ("%%V") do (
                if %%A GEQ %MIN_MAJOR% if %%B GEQ %MIN_MINOR% (
                    set PYTHON=%%P
                    goto :found_python
                )
            )
        )
    )
)

echo ERROR: Python %MIN_MAJOR%.%MIN_MINOR%+ not found.
echo Please install Python 3.11+ from https://python.org
exit /b 1

:found_python
echo Using Python: %PYTHON%

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON% -m venv %VENV_DIR%
    echo Installing dependencies...
    %VENV_DIR%\Scripts\pip install --upgrade pip -q
    %VENV_DIR%\Scripts\pip install -r %REQUIREMENTS% -q
    echo Dependencies installed.
) else (
    %VENV_DIR%\Scripts\python -c "import torch" >nul 2>&1 || (
        echo Re-installing dependencies...
        %VENV_DIR%\Scripts\pip install -r %REQUIREMENTS% -q
    )
)

%VENV_DIR%\Scripts\python main.py %*
