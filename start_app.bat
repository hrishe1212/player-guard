@echo off
echo ========================================================
echo   Launching Northbridge Meridian FC - PlayerGuard AI
echo ========================================================
start http://localhost:8502
python -m streamlit run app.py --server.port 8502
pause
