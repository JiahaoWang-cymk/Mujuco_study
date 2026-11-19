@echo off
rem Serve the MuJoCo offline folder locally (needs Python), then open the home page.
rem Use this if double-clicking index.html shows a blank page or search does not work.
rem It also lets the "lab" panels in the G1 lessons run the lesson scripts (needs mujoco in this Python).
if "%NOSTART%"=="" start "" http://localhost:8765/
python "%~dp0serve.py" 8765
