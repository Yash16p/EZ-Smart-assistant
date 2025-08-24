@echo off
echo ========================================
echo   Smart Research Assistant v2.0
echo   Powered by TinyLlama + Agentic AI
echo   (All-in-One Streamlit App)
echo ========================================
echo.

echo Starting Smart Research Assistant...
echo.

echo 1. Starting Streamlit frontend...
start "Smart Research Assistant" cmd /k "streamlit run streamlit_app.py"
echo    Frontend started in new window
echo.

echo ========================================
echo   Application Started Successfully!
echo ========================================
echo.
echo Frontend: http://localhost:8501
echo.
echo Press any key to close this window...
pause >nul
