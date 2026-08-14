@echo off
rem Run vibevoice.py with the project's Python, no activation needed.
rem
rem   vv --file speech.txt --voice Carter
rem   vv --cpu --num-threads 4 --file speech.txt
rem
rem This deliberately does not activate anything: a batch file runs in a child
rem cmd.exe, so any environment it sets is gone when it exits. To activate the
rem venv in your own shell, use  .\activate.ps1  instead.
setlocal
set "VENV=%~dp0.venv"
if exist "%~dp0.venvpath" set /p VENV=<"%~dp0.venvpath"
if exist "%VENV%\Scripts\python.exe" (
  "%VENV%\Scripts\python.exe" "%~dp0vibevoice.py" %*
) else (
  echo [vv] no venv at "%VENV%", falling back to python on PATH 1>&2
  python "%~dp0vibevoice.py" %*
)
