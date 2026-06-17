@echo off
REM Experiment A: rotation strategy x KV cache quant x model weight quant
REM
REM Two tables per (model, KV quant, strategy) cell:
REM   Table 1: KL vs universal BF16+F16 baseline
REM   Table 2: KL vs same-model F16 baseline (isolates KV-quant effect)
REM
REM For BF16 model, the two tables are identical, so we only run Table 1.
REM
REM Matrix size: 22 KL tests + 2 new baselines = ~12 min total
REM   BF16 model:  4 strategy runs (Table 1 only, since Table 2 == Table 1)
REM   Q8_KXL model: 1 model-only + 4 strategy x 2 = 9 runs
REM   Q4_KXL model: 1 model-only + 4 strategy x 2 = 9 runs

setlocal enabledelayedexpansion

set PERPLEXITY=F:\Repos\llama.cpp\build-cuda\bin\Debug\llama-perplexity.exe
set MODEL_BF16=C:\Users\MukkeshGanesh\Downloads\Qwen3.5-2B-BF16.gguf
set MODEL_Q8XL=C:\Users\MukkeshGanesh\Downloads\Qwen3.5-2B-UD-Q8_K_XL.gguf
set MODEL_Q4XL=C:\Users\MukkeshGanesh\Downloads\Qwen3.5-2B-UD-Q4_K_XL.gguf
set WIKI=F:\Repos\llama.cpp\scripts\wikitext-2-raw\wiki.test.raw
set DUMPS=F:\Repos\llama.cpp\kl_dumps
set LOGS=%DUMPS%\logs
set MEAN=%DUMPS%\variants\exp2_2k_200k.bin

set BASE_BF16=%DUMPS%\baseline_f16.kld
set BASE_Q8XL=%DUMPS%\baseline_f16_q8xl.kld
set BASE_Q4XL=%DUMPS%\baseline_f16_q4xl.kld

set COMMON=-f "%WIKI%" -ngl 999 -c 2048 -b 512 --chunks 10 --kl-divergence

REM === Verify paths ===
for %%P in ("%PERPLEXITY%" "%MODEL_BF16%" "%MODEL_Q8XL%" "%MODEL_Q4XL%" "%WIKI%" "%MEAN%") do (
    if not exist %%P (
        echo ERROR: missing %%~P
        goto :eof
    )
)

REM === Generate Q8_KXL and Q4_KXL baselines (F16 cache, no rotation, no PCA) ===
set LLAMA_KV_PCA_FILE=
set LLAMA_KV_MEAN_ONLY=

echo.
echo ===== Baseline Q8_K_XL =====
"%PERPLEXITY%" -m "%MODEL_Q8XL%" -f "%WIKI%" -ngl 999 -c 2048 -b 512 --chunks 10 --kl-divergence-base "%BASE_Q8XL%"
if errorlevel 1 echo FAILED: Q8_K_XL baseline

echo.
echo ===== Baseline Q4_K_XL =====
"%PERPLEXITY%" -m "%MODEL_Q4XL%" -f "%WIKI%" -ngl 999 -c 2048 -b 512 --chunks 10 --kl-divergence-base "%BASE_Q4XL%"
if errorlevel 1 echo FAILED: Q4_K_XL baseline

REM ============================================================
REM BF16 MODEL: 4 strategy runs vs BF16 baseline (Table 1 only)
REM ============================================================

echo.
echo ===== BF16 + Q4_0 + plain =====
set LLAMA_KV_PCA_FILE=
set LLAMA_KV_MEAN_ONLY=
"%PERPLEXITY%" -m "%MODEL_BF16%" %COMMON% --kl-divergence-base "%BASE_BF16%" -ctk q4_0 -ctv q4_0 > "%LOGS%\expA_bf16_q4_0_plain_vs_bf16.log" 2>&1

echo ===== BF16 + Q4_0 + mean =====
set LLAMA_KV_MEAN_ONLY=1
set LLAMA_KV_PCA_FILE=%MEAN%
"%PERPLEXITY%" -m "%MODEL_BF16%" %COMMON% --kl-divergence-base "%BASE_BF16%" -ctk q4_0 -ctv q4_0 > "%LOGS%\expA_bf16_q4_0_mean_vs_bf16.log" 2>&1

echo ===== BF16 + Q8_0 + plain =====
set LLAMA_KV_PCA_FILE=
set LLAMA_KV_MEAN_ONLY=
"%PERPLEXITY%" -m "%MODEL_BF16%" %COMMON% --kl-divergence-base "%BASE_BF16%" -ctk q8_0 -ctv q8_0 > "%LOGS%\expA_bf16_q8_0_plain_vs_bf16.log" 2>&1

echo ===== BF16 + Q8_0 + mean =====
set LLAMA_KV_MEAN_ONLY=1
set LLAMA_KV_PCA_FILE=%MEAN%
"%PERPLEXITY%" -m "%MODEL_BF16%" %COMMON% --kl-divergence-base "%BASE_BF16%" -ctk q8_0 -ctv q8_0 > "%LOGS%\expA_bf16_q8_0_mean_vs_bf16.log" 2>&1

REM ============================================================
REM Q8_K_XL MODEL: 1 model-only + 4 strategy x 2 references = 9 runs
REM ============================================================

echo.
echo ===== Q8_K_XL + F16 (model-only, vs BF16) =====
set LLAMA_KV_PCA_FILE=
set LLAMA_KV_MEAN_ONLY=
"%PERPLEXITY%" -m "%MODEL_Q8XL%" %COMMON% --kl-divergence-base "%BASE_BF16%" > "%LOGS%\expA_q8xl_f16_none_vs_bf16.log" 2>&1

echo ===== Q8_K_XL + Q4_0 + plain (vs BF16) =====
"%PERPLEXITY%" -m "%MODEL_Q8XL%" %COMMON% --kl-divergence-base "%BASE_BF16%" -ctk q4_0 -ctv q4_0 > "%LOGS%\expA_q8xl_q4_0_plain_vs_bf16.log" 2>&1

echo ===== Q8_K_XL + Q4_0 + plain (vs SELF) =====
"%PERPLEXITY%" -m "%MODEL_Q8XL%" %COMMON% --kl-divergence-base "%BASE_Q8XL%" -ctk q4_0 -ctv q4_0 > "%LOGS%\expA_q8xl_q4_0_plain_vs_self.log" 2>&1

echo ===== Q8_K_XL + Q4_0 + mean (vs BF16) =====
set LLAMA_KV_MEAN_ONLY=1
set LLAMA_KV_PCA_FILE=%MEAN%
"%PERPLEXITY%" -m "%MODEL_Q8XL%" %COMMON% --kl-divergence-base "%BASE_BF16%" -ctk q4_0 -ctv q4_0 > "%LOGS%\expA_q8xl_q4_0_mean_vs_bf16.log" 2>&1

echo ===== Q8_K_XL + Q4_0 + mean (vs SELF) =====
"%PERPLEXITY%" -m "%MODEL_Q8XL%" %COMMON% --kl-divergence-base "%BASE_Q8XL%" -ctk q4_0 -ctv q4_0 > "%LOGS%\expA_q8xl_q4_0_mean_vs_self.log" 2>&1

echo ===== Q8_K_XL + Q8_0 + plain (vs BF16) =====
set LLAMA_KV_PCA_FILE=
set LLAMA_KV_MEAN_ONLY=
"%PERPLEXITY%" -m "%MODEL_Q8XL%" %COMMON% --kl-divergence-base "%BASE_BF16%" -ctk q8_0 -ctv q8_0 > "%LOGS%\expA_q8xl_q8_0_plain_vs_bf16.log" 2>&1

echo ===== Q8_K_XL + Q8_0 + plain (vs SELF) =====
"%PERPLEXITY%" -m "%MODEL_Q8XL%" %COMMON% --kl-divergence-base "%BASE_Q8XL%" -ctk q8_0 -ctv q8_0 > "%LOGS%\expA_q8xl_q8_0_plain_vs_self.log" 2>&1

echo ===== Q8_K_XL + Q8_0 + mean (vs BF16) =====
set LLAMA_KV_MEAN_ONLY=1
set LLAMA_KV_PCA_FILE=%MEAN%
"%PERPLEXITY%" -m "%MODEL_Q8XL%" %COMMON% --kl-divergence-base "%BASE_BF16%" -ctk q8_0 -ctv q8_0 > "%LOGS%\expA_q8xl_q8_0_mean_vs_bf16.log" 2>&1

echo ===== Q8_K_XL + Q8_0 + mean (vs SELF) =====
"%PERPLEXITY%" -m "%MODEL_Q8XL%" %COMMON% --kl-divergence-base "%BASE_Q8XL%" -ctk q8_0 -ctv q8_0 > "%LOGS%\expA_q8xl_q8_0_mean_vs_self.log" 2>&1

REM ============================================================
REM Q4_K_XL MODEL: 1 model-only + 4 strategy x 2 references = 9 runs
REM ============================================================

echo.
echo ===== Q4_K_XL + F16 (model-only, vs BF16) =====
set LLAMA_KV_PCA_FILE=
set LLAMA_KV_MEAN_ONLY=
"%PERPLEXITY%" -m "%MODEL_Q4XL%" %COMMON% --kl-divergence-base "%BASE_BF16%" > "%LOGS%\expA_q4xl_f16_none_vs_bf16.log" 2>&1

echo ===== Q4_K_XL + Q4_0 + plain (vs BF16) =====
"%PERPLEXITY%" -m "%MODEL_Q4XL%" %COMMON% --kl-divergence-base "%BASE_BF16%" -ctk q4_0 -ctv q4_0 > "%LOGS%\expA_q4xl_q4_0_plain_vs_bf16.log" 2>&1

echo ===== Q4_K_XL + Q4_0 + plain (vs SELF) =====
"%PERPLEXITY%" -m "%MODEL_Q4XL%" %COMMON% --kl-divergence-base "%BASE_Q4XL%" -ctk q4_0 -ctv q4_0 > "%LOGS%\expA_q4xl_q4_0_plain_vs_self.log" 2>&1

echo ===== Q4_K_XL + Q4_0 + mean (vs BF16) =====
set LLAMA_KV_MEAN_ONLY=1
set LLAMA_KV_PCA_FILE=%MEAN%
"%PERPLEXITY%" -m "%MODEL_Q4XL%" %COMMON% --kl-divergence-base "%BASE_BF16%" -ctk q4_0 -ctv q4_0 > "%LOGS%\expA_q4xl_q4_0_mean_vs_bf16.log" 2>&1

echo ===== Q4_K_XL + Q4_0 + mean (vs SELF) =====
"%PERPLEXITY%" -m "%MODEL_Q4XL%" %COMMON% --kl-divergence-base "%BASE_Q4XL%" -ctk q4_0 -ctv q4_0 > "%LOGS%\expA_q4xl_q4_0_mean_vs_self.log" 2>&1

echo ===== Q4_K_XL + Q8_0 + plain (vs BF16) =====
set LLAMA_KV_PCA_FILE=
set LLAMA_KV_MEAN_ONLY=
"%PERPLEXITY%" -m "%MODEL_Q4XL%" %COMMON% --kl-divergence-base "%BASE_BF16%" -ctk q8_0 -ctv q8_0 > "%LOGS%\expA_q4xl_q8_0_plain_vs_bf16.log" 2>&1

echo ===== Q4_K_XL + Q8_0 + plain (vs SELF) =====
"%PERPLEXITY%" -m "%MODEL_Q4XL%" %COMMON% --kl-divergence-base "%BASE_Q4XL%" -ctk q8_0 -ctv q8_0 > "%LOGS%\expA_q4xl_q8_0_plain_vs_self.log" 2>&1

echo ===== Q4_K_XL + Q8_0 + mean (vs BF16) =====
set LLAMA_KV_MEAN_ONLY=1
set LLAMA_KV_PCA_FILE=%MEAN%
"%PERPLEXITY%" -m "%MODEL_Q4XL%" %COMMON% --kl-divergence-base "%BASE_BF16%" -ctk q8_0 -ctv q8_0 > "%LOGS%\expA_q4xl_q8_0_mean_vs_bf16.log" 2>&1

echo ===== Q4_K_XL + Q8_0 + mean (vs SELF) =====
"%PERPLEXITY%" -m "%MODEL_Q4XL%" %COMMON% --kl-divergence-base "%BASE_Q4XL%" -ctk q8_0 -ctv q8_0 > "%LOGS%\expA_q4xl_q8_0_mean_vs_self.log" 2>&1

echo.
echo === Exp A done. Logs in %LOGS% ===
endlocal
