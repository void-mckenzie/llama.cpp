@echo off
REM Regenerate baseline_f16.kld using BF16 GGUF as the model
REM - Moves existing baseline to .bak before regen
REM - Restores from .bak if perplexity fails
REM - Existing .bak is NOT overwritten (one-time migration only)

setlocal

set PERPLEXITY=F:\Repos\llama.cpp\build-cuda\bin\Debug\llama-perplexity.exe
set MODEL=C:\Users\MukkeshGanesh\Downloads\Qwen3.5-2B-BF16.gguf
set WIKI=F:\Repos\llama.cpp\scripts\wikitext-2-raw\wiki.test.raw
set BASE=F:\Repos\llama.cpp\kl_dumps\baseline_f16.kld
set BAK=F:\Repos\llama.cpp\kl_dumps\baseline_f16_q8xl.kld.bak

if not exist "%MODEL%" (
    echo ERROR: BF16 GGUF not found at %MODEL%
    goto :eof
)
if not exist "%PERPLEXITY%" (
    echo ERROR: llama-perplexity.exe not found at %PERPLEXITY%
    goto :eof
)
if not exist "%WIKI%" (
    echo ERROR: wikitext not found at %WIKI%
    goto :eof
)

REM Backup current baseline (skip if .bak already exists — preserves original Q8 baseline)
if exist "%BASE%" (
    if not exist "%BAK%" (
        move "%BASE%" "%BAK%"
        echo Backed up existing baseline to %BAK%
    ) else (
        echo WARNING: %BAK% already exists, deleting current baseline without backup
        del "%BASE%"
    )
)

set LLAMA_KV_PCA_FILE=
set LLAMA_KV_MEAN_ONLY=

echo.
echo === Regenerating baseline_f16.kld with BF16 GGUF ===
"%PERPLEXITY%" -m "%MODEL%" -f "%WIKI%" -ngl 999 -c 2048 -b 512 --chunks 10 --kl-divergence-base "%BASE%"
if errorlevel 1 (
    echo.
    echo FAILED: perplexity returned errorlevel 1
    if exist "%BAK%" (
        echo Restoring baseline from backup...
        if exist "%BASE%" del "%BASE%"
        move "%BAK%" "%BASE%"
    )
    goto :eof
)

if not exist "%BASE%" (
    echo FAILED: baseline file was not created
    if exist "%BAK%" move "%BAK%" "%BASE%"
    goto :eof
)

echo.
echo === Success ===
echo New baseline: %BASE%
if exist "%BAK%" echo Old baseline preserved at: %BAK%
endlocal
