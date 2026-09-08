@echo off
REM ============================================================================
REM S.A.R.A. Universal Launcher (Windows)
REM ============================================================================
title S.A.R.A. -- Sentiment Analysis & Response AI

echo ============================================================
echo   S.A.R.A. -- Sentiment Analysis ^& Response AI
echo ============================================================
echo.
echo Select launch mode:
echo   [1] Web Portal (Streamlit Interface) [Recommended]
echo   [2] Desktop Live HUD (Webcam + Spacebar Microphone)
echo   [3] Auth Server Only (Port 5001)
echo   [4] Docker Compose (Full Stack)
echo.

set /p choice="Enter choice [1-4] (default: 1): "
if "%choice%"=="" set choice=1

if "%choice%"=="1" (
    echo Starting S.A.R.A. Web Portal on http://localhost:8501 ...
    streamlit run app.py
) else if "%choice%"=="2" (
    echo Launching Auth Server in new window...
    start "SARA Auth Server" python auth_server.py
    timeout /t 2 /nobreak >nul
    echo Launching S.A.R.A. Desktop Live HUD...
    python main.py
) else if "%choice%"=="3" (
    echo Starting S.A.R.A. Authentication Server on http://localhost:5001 ...
    python auth_server.py
) else if "%choice%"=="4" (
    echo Building and starting Docker containers...
    docker compose up --build
) else (
    echo Invalid choice. Exiting.
)
pause
