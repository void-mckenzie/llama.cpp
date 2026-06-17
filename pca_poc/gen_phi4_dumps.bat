@echo off
setlocal

set MODEL=C:\Users\MukkeshGanesh\Downloads\Phi-4-mini-reasoning-BF16.gguf
set PCA=F:\Repos\llama.cpp\kl_dumps\variants\phi4mini_mean.bin
set CLI=F:\Repos\llama.cpp\build-cuda\bin\Debug\llama-cli.exe
set DUMP=F:\Repos\llama.cpp\kv_dumps_phi4
set PROMPT=What is 1+5?

if not exist "%DUMP%" mkdir "%DUMP%"
set LLAMA_KV_DUMP_DIR=%DUMP%

set COMMON=-m "%MODEL%" -p "%PROMPT%" --single-turn --reasoning off --temp 0 --top-k 1 --top-p 1.0 --min-p 0.0 --repeat-penalty 1.0 --seed 0 -c 2048 -b 256 -ub 256 -t 4 -ngl 999

echo === (1/3) F16 baseline (no PCA, no quant) ===
set LLAMA_KV_PCA_FILE=
set LLAMA_KV_MEAN_ONLY=
"%CLI%" %COMMON%
if errorlevel 1 echo   FAILED config 1

echo.
echo === (2/3) Q4_0 Hadamard (plain rotation) ===
set LLAMA_KV_PCA_FILE=
set LLAMA_KV_MEAN_ONLY=
"%CLI%" %COMMON% -ctk q4_0 -ctv q4_0
if errorlevel 1 echo   FAILED config 2

echo.
echo === (3/3) Q4_0 mean-only (Hadamard + per-head mean) ===
set LLAMA_KV_PCA_FILE=%PCA%
set LLAMA_KV_MEAN_ONLY=1
"%CLI%" %COMMON% -ctk q4_0 -ctv q4_0
if errorlevel 1 echo   FAILED config 3

echo.
echo === DONE - dump subdirs in %DUMP% ===
dir /b /ad "%DUMP%"
