@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  gpu_env.bat  —  Activate the Python 3.12 + CUDA venv
REM  Usage:
REM    gpu_env.bat                      (opens an interactive shell in the venv)
REM    gpu_env.bat python run_training.py
REM    gpu_env.bat python scale_6b.py
REM    gpu_env.bat python run_inference.py
REM ─────────────────────────────────────────────────────────────────────────
call "%~dp0.venv\Scripts\activate.bat"
if "%~1"=="" (
    echo.
    echo  Transformer GPU environment active ^(Python 3.12 + CUDA^)
    echo  Commands:
    echo    python run_training.py    -- train EN-^>FR on RTX 4050
    echo    python scale_6b.py        -- 6B analysis + GPU benchmark
    echo    python run_inference.py   -- interactive translation REPL
    echo    python visualize.py       -- attention + PE visualizations
    echo.
    cmd /k
) else (
    %*
)
