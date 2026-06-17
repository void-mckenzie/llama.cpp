@echo off
setlocal

set MODEL=C:\Users\MukkeshGanesh\Downloads\Qwen3.5-2B-UD-Q8_K_XL.gguf
set CLI=F:\Repos\llama.cpp\build-cuda\bin\Debug\llama-perplexity.exe
set WIKI=F:\Repos\llama.cpp\scripts\wikitext-2-raw\wiki.test.raw
set BASE=F:\Repos\llama.cpp\kl_dumps\baseline_f16.kld
set VARS=F:\Repos\llama.cpp\kl_dumps\variants
set LOGS=F:\Repos\llama.cpp\kl_dumps\logs
set LLAMA_KV_MEAN_ONLY=1

set COMMON=-m "%MODEL%" -f "%WIKI%" -ngl 999 -c 2048 -b 512 --chunks 10 -ctk q4_0 -ctv q4_0 --kl-divergence --kl-divergence-base "%BASE%"

for %%N in (32 128 512 2048 4096) do (
    echo === Running prefix N=%%N ===
    set LLAMA_KV_PCA_FILE=%VARS%\pca_prefix_N%%N.bin
    "%CLI%" %COMMON% > "%LOGS%\prefix_N%%N.log" 2>&1
)

echo.
echo === All prefix runs done. Logs in %LOGS% ===
