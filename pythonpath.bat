@ECHO OFF
setlocal
set PYTHONPATH="%PYTHONPATH%";%~dp0
for /f "tokens=1,* delims= " %%a in ("%*") do set exp_args=%%b
python %1 %exp_args%
endlocal