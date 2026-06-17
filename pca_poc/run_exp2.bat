@echo off
REM Experiment 2: token count variation, chunk size fixed at 2K
REM Axis: max_tokens in {100000, 200000, 300000}, chunk_size_tokens = 2048
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
echo ===== Exp 2: chunk=2048, max=100000 =====
"%PY%" "%SCRIPT%" --chunk_size_tokens 2048 --max_tokens 100000 --tag exp2_2k_100k
if errorlevel 1 (
    echo FAILED: python script for exp2_2k_100k
    goto try_200k
)
if not exist "%VARS%\exp2_2k_100k.bin" (
    echo FAILED: .bin missing for exp2_2k_100k
    goto try_200k
)
set LLAMA_KV_PCA_FILE=%VARS%\exp2_2k_100k.bin
"%PERPLEXITY%" %KL_COMMON% > "%LOGS%\exp2_2k_100k.log" 2>&1
if errorlevel 1 echo FAILED: KL test for exp2_2k_100k

:try_200k
echo.
echo ===== Exp 2: chunk=2048, max=200000 =====
"%PY%" "%SCRIPT%" --chunk_size_tokens 2048 --max_tokens 200000 --tag exp2_2k_200k
if errorlevel 1 (
    echo FAILED: python script for exp2_2k_200k
    goto try_300k
)
if not exist "%VARS%\exp2_2k_200k.bin" (
    echo FAILED: .bin missing for exp2_2k_200k
    goto try_300k
)
set LLAMA_KV_PCA_FILE=%VARS%\exp2_2k_200k.bin
"%PERPLEXITY%" %KL_COMMON% > "%LOGS%\exp2_2k_200k.log" 2>&1
if errorlevel 1 echo FAILED: KL test for exp2_2k_200k

:try_300k
echo.
echo ===== Exp 2: chunk=2048, max=300000 =====
"%PY%" "%SCRIPT%" --chunk_size_tokens 2048 --max_tokens 300000 --tag exp2_2k_300k
if errorlevel 1 (
    echo FAILED: python script for exp2_2k_300k
    goto done
)
if not exist "%VARS%\exp2_2k_300k.bin" (
    echo FAILED: .bin missing for exp2_2k_300k
    goto done
)
set LLAMA_KV_PCA_FILE=%VARS%\exp2_2k_300k.bin
"%PERPLEXITY%" %KL_COMMON% > "%LOGS%\exp2_2k_300k.log" 2>&1
if errorlevel 1 echo FAILED: KL test for exp2_2k_300k

:done
echo.
echo === Exp 2 done. Logs in %LOGS% ===
endlocal
