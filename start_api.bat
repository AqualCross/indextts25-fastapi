@echo off
rem =====================================================================
rem  indextts25-fastapi - one-click start (Windows)
rem  Expected layout:   <index-tts>\indextts25-fastapi\start_api.bat
rem  The scripts cd's to the parent directory (= index-tts root) because
rem  the upstream code resolves checkpoints/, characters/, examples/
rem  relative to the working directory.
rem  Edit GPU / PORT below, extra arguments are forwarded to run_api.py.
rem =====================================================================
setlocal
set GPU=1
set PORT=8000

set WRAPPER=%~dp0
for %%I in ("%~dp0..") do set INDEX_ROOT=%%~fI

cd /d "%INDEX_ROOT%"
echo [start_api] index-tts root: %INDEX_ROOT%
echo [start_api] GPU=%GPU% PORT=%PORT% (BF16)
uv run "%WRAPPER%run_api.py" --gpu %GPU% --bf16 --host 0.0.0.0 --port %PORT% %*
endlocal
