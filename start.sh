#!/usr/bin/env bash
# ==============================================================================
# S.A.R.A. Universal Launcher (Linux / macOS)
# ==============================================================================
set -e

echo "============================================================"
echo "  S.A.R.A. -- Sentiment Analysis & Response AI"
echo "============================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH."
    exit 1
fi

echo "Select launch mode:"
echo "  1) Web Portal (Streamlit Cloud/Local Interface) [Recommended]"
echo "  2) Desktop Live HUD (Webcam + Microphone Spacebar)"
echo "  3) Auth Server Only (Port 5001)"
echo "  4) Docker Compose (Full Stack)"
read -p "Enter choice [1-4] (default: 1): " choice
choice=${choice:-1}

case "$choice" in
    1)
        echo "Starting S.A.R.A. Streamlit Web Portal on http://localhost:8501 ..."
        streamlit run app.py
        ;;
    2)
        echo "Starting Auth Server in background..."
        python3 auth_server.py &
        AUTH_PID=$!
        sleep 2
        echo "Starting S.A.R.A. Desktop Live HUD..."
        python3 main.py || true
        kill $AUTH_PID 2>/dev/null || true
        ;;
    3)
        echo "Starting S.A.R.A. Authentication Server on http://localhost:5001 ..."
        python3 auth_server.py
        ;;
    4)
        echo "Building and launching Docker containers..."
        docker compose up --build
        ;;
    *)
        echo "Invalid selection. Exiting."
        exit 1
        ;;
esac
