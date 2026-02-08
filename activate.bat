@echo off
REM Quick activation script for virtual environment

if not exist "venv\" (
    echo ERROR: Virtual environment not found!
    echo Please run: install.bat
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Virtual environment activated!
echo You should see (venv) in your prompt now.
echo ============================================================
echo.
echo What would you like to do?
echo   1. Run Streamlit app:  streamlit run app.py
echo   2. Run CLI:            python cli.py chat
echo   3. Verify setup:       python setup.py
echo   4. Process docs:       python cli.py process
echo.
echo Or just type your command directly.
echo.
