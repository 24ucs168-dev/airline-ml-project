@echo off
echo ========================================================
echo   SkyPulse - Airline Customer Satisfaction ML App
echo ========================================================
echo Starting Streamlit Web Application...

IF EXIST "C:\ProgramData\anaconda3\python.exe" (
    "C:\ProgramData\anaconda3\python.exe" -m streamlit run app.py
) ELSE (
    python -m streamlit run app.py
)

pause
