#!/bin/bash

echo "========================================"
echo "  Smart Research Assistant v2.0"
echo "  Powered by TinyLlama + Agentic AI"
echo "  (All-in-One Streamlit App)"
echo "========================================"
echo

echo "Starting Smart Research Assistant..."
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv venv"
    echo "Then activate it and install requirements"
    exit 1
fi

# Activate virtual environment
echo "1. Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"
echo

# Check if dependencies are installed
if ! python -c "import transformers, langchain, streamlit" 2>/dev/null; then
    echo "❌ Dependencies not installed!"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

echo "2. Starting Streamlit frontend..."
streamlit run streamlit_app.py &
FRONTEND_PID=$!
echo "   Frontend started (PID: $FRONTEND_PID)"
echo

echo "========================================"
echo "   Application Started Successfully!"
echo "========================================"
echo
echo "Frontend: http://localhost:8501"
echo
echo "Press Ctrl+C to stop the service..."
echo

# Function to cleanup on exit
cleanup() {
    echo
    echo "🛑 Stopping service..."
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Service stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait
