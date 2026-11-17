@echo off
rem Serve the MuJoCo offline folder locally (needs Python), then open the English or Chinese docs from the page.
rem Use this if double-clicking index.html shows a blank page or search does not work.
if "%NOSTART%"=="" start "" http://localhost:8765/
python -m http.server 8765 -d "%~dp0."
