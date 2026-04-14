@echo off
title AI Chatbot Loader
color 0A

echo ==========================================
echo   Starting Terminal AI Chatbot
echo ==========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python found.
echo.

:: Check if requirements need to be installed
if exist "requirements.txt" (
    echo Checking dependencies...
    pip install -r requirements.txt --quiet
    if %errorlevel% neq 0 (
        echo [WARNING] Some dependencies might have failed to install.
    ) else (
        echo [OK] Dependencies verified.
    )
)
echo.

:: Create data directory if it doesn't exist
if not exist "data\training_data.json" (
    echo [INFO] No training data found. Generating demo data...
    mkdir data 2>nul
)

echo ==========================================
echo   Launching Chatbot...
echo ==========================================
echo.
echo * To train on custom data, edit 'data/training_data.json'
echo * Type 'quit' in the chat to exit.
echo.

:: Run the chatbot with demo mode and light training for quick start
python run.py --demo --epochs 3 --samples 20

pause
