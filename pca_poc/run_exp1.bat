@echo off
REM Experiment 1: chunk size variation, WHOLE dataset
REM Axis: chunk_size_tokens in {2048, 4096, 8192}, max_tokens = all
REM Output: 3 .bin files + 3 KL logs

setlocal

set PY=C:\Users\MukkeshGanesh\miniconda3\envs\kvpress_env\python.exe
set SCRIPT=F:\Repos\llama.cpp\pca_poc\pca_pool_calibrate.py
set VARS=F:\Repos\llama.cpp\kl_dumps\variants
set LOGS=F:\Repos\llama.cpp\kl_dumps\logs

set PERPLEXITY=F:\Repos\llama.cpp\build-cuda\bin\Debug\llama-perplexity.exe
set MODEL=C:\Users\MukkeshGanesh\Downloads\Qwen3.5-2B-BF16.gguf
set WIKI=F:\Repos\llama.cpp\scripts\wikitext-2-raw\wiki.test.raw
set BASE=F:\Repos\llama.cpp\kl_dumps\baseline_f16.kld
set KL_COMMON=-m "%MODEL%" -f "%WIKI%" -ngl 999 -c 2048 -b 512 --chunks 10 -ctk q4_0 -ctv q4_0 --kl-divergence --kl-divergence-base "%BASE%"
set LLAMA_KV_MEAN_ONLY=1

if not exist "%BASE%" (
    echo ERROR: baseline_f16.kld not found at %BASE%
    echo Run regen_baseline_bf16.bat first.
    goto :eof
)

echo.
echo ===== Exp 1: chunk=2048, max=all =====
"%PY%" "%SCRIPT%" --chunk_size_tokens 2048 --max_tokens all --tag exp1_2k_all
if errorlevel 1 (
    echo FAILED: python script for exp1_2k_all
    goto try_4k
)
if not exist "%VARS%\exp1_2k_all.bin" (
    echo FAILED: .bin missing for exp1_2k_all
    goto try_4k
)
set LLAMA_KV_PCA_FILE=%VARS%\exp1_2k_all.bin
"%PERPLEXITY%" %KL_COMMON% > "%LOGS%\exp1_2k_all.log" 2>&1
if errorlevel 1 echo FAILED: KL test for exp1_2k_all

:try_4k
echo.
echo ===== Exp 1: chunk=4096, max=all =====
"%PY%" "%SCRIPT%" --chunk_size_tokens 4096 --max_tokens all --tag exp1_4k_all
if errorlevel 1 (
    echo FAILED: python script for exp1_4k_all
    goto try_8k
)
if not exist "%VARS%\exp1_4k_all.bin" (
    echo FAILED: .bin missing for exp1_4k_all
    goto try_8k
)
set LLAMA_KV_PCA_FILE=%VARS%\exp1_4k_all.bin
"%PERPLEXITY%" %KL_COMMON% > "%LOGS%\exp1_4k_all.log" 2>&1
if errorlevel 1 echo FAILED: KL test for exp1_4k_all

:try_8k
echo.
echo ===== Exp 1: chunk=8192, max=all =====
"%PY%" "%SCRIPT%" --chunk_size_tokens 8192 --max_tokens all --tag exp1_8k_all
if errorlevel 1 (
    echo FAILED: python script for exp1_8k_all
    goto done
)
if not exist "%VARS%\exp1_8k_all.bin" (
    echo FAILED: .bin missing for exp1_8k_all
    goto done
)
set LLAMA_KV_PCA_FILE=%VARS%\exp1_8k_all.bin
"%PERPLEXITY%" %KL_COMMON% > "%LOGS%\exp1_8k_all.log" 2>&1
if errorlevel 1 echo FAILED: KL test for exp1_8k_all

:done
echo.
echo === Exp 1 done. Logs in %LOGS% ===
endlocal
