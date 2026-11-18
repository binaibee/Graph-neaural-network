@echo off
rem Starts the demonstration. Double-click this file, or run it from a terminal.
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
if not exist "results\model.pt" (
    echo No trained model found. Training all models now ^(this takes several minutes^)...
    python -W ignore -m scam_gnn.train || goto :error
)
echo.
echo Starting the demo. Your browser will open at http://localhost:8501
echo Keep this window open during the presentation. Press Ctrl+C here to stop.
echo.
rem Open the browser a few seconds after the server starts (ping is used as a delay that works in every kind of terminal).
start "" /b cmd /c "ping -n 6 127.0.0.1 >nul & start http://localhost:8501"
python -W ignore -m streamlit run app\app.py --server.headless true --server.address localhost --server.port 8501 --browser.gatherUsageStats false --theme.base light
goto :eof

:error
echo.
echo The demo could not start. See the message above.
pause
