---
name: KV cache dumper + dashboard exist for PCA POC
description: PCA-vs-Hadamard validation toolchain is built and verified — progress.md has the full state
type: project
originSessionId: 0bb1b394-b1bf-4d19-a7a4-c47efd868285
---
The KV cache dumper + Plotly Dash dashboard for the PCA POC are implemented and verified end-to-end. Full state is in `F:\Repos\llama.cpp\pca_poc\progress.md` — read it before touching anything PCA-related.

**Why:** This was a multi-session build. The dumper triggers on `/exit` via env var `LLAMA_KV_DUMP_DIR`, writes per-layer `.npy` + `meta.json` per run. The dashboard (`pca_poc/kv_dashboard.py`) loads a dump dir, back-rotates each run to original K/V space, and shows heatmaps, metrics tables, cross-layer bars, distributions, and a text-divergence tab. Pure-numpy Q4_0 simulation matches llama-cli output to 4 decimals — proves the rotation alone explains the quantization difference.

**How to apply:** Before re-explaining "what PCA does" or "why PCA beats Hadamard", read `progress.md`. The verified numbers (PCA −12% RMS / +1.15 dB SNR vs Hadamard at Q4_0 average across 6 layers) and the W-direction sign convention (`K @ W.T` forward, `K_pca @ W + mean` inverse, NOT the other way around) are documented there with reproduction scripts.
