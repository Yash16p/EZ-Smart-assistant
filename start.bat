@echo off
echo ========================================
echo   Smart Research Assistant v2.0
echo   Powered by TinyLlama + Agentic AI
echo ========================================
echo.

echo Starting Smart Research Assistant...
echo.

echo 1. Starting FastAPI backend...
start "Backend Server" cmd /k "python main.py"
echo    Backend started in new window
echo.

echo 2. Waiting for backend to initialize...
timeout /t 5 /nobreak >nul
echo.

echo 3. Starting Streamlit frontend...
start "Frontend" cmd /k "streamlit run streamlit_app.py"
echo    Frontend started in new window
echo.

echo ========================================
echo   Application Started Successfully!
echo ========================================
echo.
echo Frontend: http://localhost:8501
echo Backend:  http://localhost:8000
echo.
echo Press any key to close this window...
pause >nul
