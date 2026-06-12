#!/usr/bin/env bash
set -euo pipefail

REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=11
VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"

echo "=== OUROBOROS Bootstrap ==="

# Find python3.11+
PYTHON=""
for candidate in python3.11 python3.12 python3.13 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge "$REQUIRED_PYTHON_MAJOR" ] && [ "$minor" -ge "$REQUIRED_PYTHON_MINOR" ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python $REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR+ not found."
    echo "Please install Python 3.11 or newer from https://python.org"
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
    echo "Installing dependencies (this may take a few minutes)..."
    "$VENV_DIR/bin/pip" install --upgrade pip -q
    "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS" -q
    echo "Dependencies installed."
else
    # Quick check: if torch missing, reinstall
    if ! "$VENV_DIR/bin/python" -c "import torch" &>/dev/null; then
        echo "Re-installing dependencies..."
        "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS" -q
    fi
fi

exec "$VENV_DIR/bin/python" main.py "$@"
