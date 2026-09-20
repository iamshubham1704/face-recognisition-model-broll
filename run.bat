@echo off
REM Always runs uvicorn using the project's own .venv, so it never
REM accidentally picks up the global Python (which lacks insightface).
"%~dp0.venv\Scripts\python.exe" -m uvicorn main:app --reload
