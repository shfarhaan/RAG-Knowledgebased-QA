@echo off
REM Setup script for Agentic RAG System (Windows)

echo ============================================================
echo Agentic RAG System - Quick Setup with Virtual Environment
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!
python --version
echo.

REM Check if virtual environment already exists
if exist "venv\" (
    echo Virtual environment already exists.
    echo.
) else (
    echo Step 1: Creating virtual environment...
    echo.
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
    echo.
)

echo Step 2: Activating virtual environment...
echo.
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Step 3: Upgrading pip...
echo.
python -m pip install --upgrade pip

echo.
echo Step 4: Installing dependencies...
echo This may take a few minutes...
echo.
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Installation complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Create .env file with: GEMINI_API_KEY=your-key-here
echo 2. Get API key from: https://makersuite.google.com/app/apikey
echo 3. Run: streamlit run app.py
echo.
echo Or run: python setup.py to verify installation
echo.
pause
