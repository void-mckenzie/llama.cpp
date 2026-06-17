@echo off
setlocal

set MODEL=C:\Users\MukkeshGanesh\Downloads\Phi-4-mini-reasoning-BF16.gguf
set CLI=F:\Repos\llama.cpp\build-cuda\bin\Debug\llama-cli.exe
set DUMP=F:\Repos\llama.cpp\kv_dumps_phi4_diag
set PROMPT=What is 1+5?

if not exist "%DUMP%" mkdir "%DUMP%"
set LLAMA_KV_DUMP_DIR=%DUMP%

echo === Test A: plain Hadamard, no rot disable (should crash) ===
set LLAMA_KV_PCA_FILE=
set LLAMA_KV_MEAN_ONLY=
set LLAMA_ATTN_ROT_DISABLE=
"%CLI%" -m "%MODEL%" -p "%PROMPT%" --single-turn --reasoning off --temp 0 --top-k 1 -c 2048 -b 256 -ub 256 -t 4 -ngl 999 -ctk q4_0 -ctv q4_0 -n 5
if errorlevel 1 echo   A: CRASH (as expected)

echo.
echo === Test B: Hadamard DISABLED via env var ===
set LLAMA_KV_PCA_FILE=
set LLAMA_KV_MEAN_ONLY=
set LLAMA_ATTN_ROT_DISABLE=1
"%CLI%" -m "%MODEL%" -p "%PROMPT%" --single-turn --reasoning off --temp 0 --top-k 1 -c 2048 -b 256 -ub 256 -t 4 -ngl 999 -ctk q4_0 -ctv q4_0 -n 5
if errorlevel 1 echo   B: CRASH (Hadamard not the culprit)
echo   If no crash above: Hadamard IS the culprit

echo.
echo === Test C: asymmetric quant (q4_0 K, f16 V) ===
set LLAMA_KV_PCA_FILE=
set LLAMA_KV_MEAN_ONLY=
set LLAMA_ATTN_ROT_DISABLE=
"%CLI%" -m "%MODEL%" -p "%PROMPT%" --single-turn --reasoning off --temp 0 --top-k 1 -c 2048 -b 256 -ub 256 -t 4 -ngl 999 -ctk q4_0 -ctv f16 -n 5
if errorlevel 1 echo   C: CRASH (K-side Hadamard broken)
echo   If no crash: V-side Hadamard was the culprit

echo.
echo DONE
