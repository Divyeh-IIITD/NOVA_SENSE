@echo off
setlocal

cd /d "%~dp0"
python -m streamlit run streamlit_app.py

endlocal