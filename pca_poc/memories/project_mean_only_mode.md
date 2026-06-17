---
name: mean-only PCA mode + CUDA build + KL divergence validation
description: LLAMA_KV_MEAN_ONLY=1 (Hadamard+mean) beats full PCA on logit KL divergence on Wikitext-2; build-cuda/ has CUDA binaries
type: project
originSessionId: 0bb1b394-b1bf-4d19-a7a4-c47efd868285
---
Three pieces of state from the latest work:

**1. Mean-only mode (Hadamard + per-head mean subtract).** Triggered by `LLAMA_KV_PCA_FILE=...` + `LLAMA_KV_MEAN_ONLY=1`. Loads the PCA file but only uses the per-head means — Hadamard rotation stays enabled (NOT suppressed like full-PCA mode). `build_attn` subtracts mean before Hadamard forward, adds mean_v back after V back-rotation (K mean is dropped, softmax absorbs). Ablation showed this captures ~all of PCA's Q4_0 win without the calibration pipeline. Code changes in `src/llama-kv-cache.{h,cpp}` (ctor + accessors) and `src/llama-graph.cpp` (new `pca_apply_mean_offset` helper, modified `build_attn`). CPU-validated on math prompt: cache values match `quantize(H(K - mean_k))` to 0.001, generated text matches F16 baseline at word 178 vs 52 for plain Had/PCA.

**2. CUDA build at `F:\Repos\llama.cpp\build-cuda\`.** Configured `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=89` (RTX 3500 Ada), shared libs. MSVC multi-config still puts binaries in `bin/Debug/` subdir despite Release CMake config. Hardware: RTX 3500 Ada 12GB, CUDA 12.8. **GPU-validated for all three Q4_0 modes**: plain Hadamard (85.9 t/s), mean-only (78.6 t/s), full PCA (67.0 t/s). Mean-only is faster than full PCA on GPU (WH transform beats dense matmul). GPU cache values match pure-numpy sim to ~1%. `llama-perplexity` is also built in this dir.

**3. KL divergence validation (decisive).** Ran `llama-perplexity --kl-divergence` on Wikitext-2 test (10 chunks ≈ 5120 tokens) with F16 cache baseline. **Mean-only beats full PCA on every metric**: Mean KLD 0.00358 vs 0.00429 (17% better), RMS Δp 1.68% vs 1.87%, same-top-p 96.70% vs 96.21%, ΔPPL +0.094% vs +0.156%. Full PCA has catastrophic Max KLD outliers (1.567 vs mean-only's 0.171). Logs at `F:\Repos\llama.cpp\kl_dumps\logs\0[1-4]_*.log`. This answers the "Q2 generation quality" question — mean-only is the upstreamable choice.

**4. Mean direction matters + tiny calibration wins (variant + prefix-N experiments).** Two follow-up experiments closed the remaining questions:
   - **Variant .bins** (`kl_dumps/variants/pca_{zero,rand_dir,rand_gauss,shuffle,per_layer_avg}.bin`): replacing the PCA mean with random direction / random Gaussian / per-layer avg all *fail* (KL worse than zero mean in the random cases). The mean must be a real per-head activation mean — direction matters.
   - **Prefix-N** (`kl_dumps/variants/pca_prefix_N{32,128,512,2048,4096}.bin`): computing means from N-token prefixes of `calibration_datav5.txt` shows convergence to PCA-mean (0.00358) is fast — N=32 lands at 0.00433 (21% above), N=512 at 0.00405 (13% above), N=2048 at 0.00392 (9% above). Beyond ~2K tokens, returns flatten.

**Conclusion: pre-computed mean file from ~200K tokens of diverse calibration corpus is the right upstreaming recipe.** Pool calibration experiments (Exp 1: chunk size 2K/4K/8K × whole dataset; Exp 2: first 100K/200K/300K × 2K chunks) showed:
- First-N from start of corpus beats sampling across the whole file (8-13% better Mean KLD, 4-7× better Max KLD)
- Sweet spot is ~200K tokens (Mean KLD 0.003184 against BF16 baseline on Wikitext-2); plateaus past 200K
- Chunk size doesn't matter (2K/4K/8K within 5% at fixed data range)
- Diversity is load-bearing — single-paper mean (0.00390 from prefix-N=4096 biomedical-only) is worse than diverse 100K mean (0.00321)
- Recipe: take first 200K tokens of calibration_datav5.txt, slice into 2K-token chunks, compute per-head means, ship as ~6 KB `.bin`. No PCA pipeline needed.

**Exp A (cross-quant matrix: 3 model weight quants × 2 KV cache quants × 2 strategies) refined the target deployment:**
- Mean-only's relative Q4_0 KL win shrinks as weight quant worsens: BF16 48%, Q8_K_XL 42%, Q4_K_XL 12%
- Q4_K_XL weight floor (0.01707 KLD) dominates total — model-only noise swamps KV improvement
- At Q8_0 cache, mean vs plain statistically identical — Q8_0 has enough resolution
- **Mean-only is most valuable for: BF16/Q8 weights + Q4_0 cache.** For Q4_K_XL or Q8_0 cache, marginal benefit
- Capacity angle: Q4_0 cache saves 50% VRAM vs Q8_0 — KL alone doesn't capture this; long-context workloads benefit regardless of model weight quant
- Recommended UX: ship as opt-in flag, default-on for `-ctk q4_0 -ctv q4_0`

**Cross-architecture compatibility check:** reviewed `src/models/*.cpp`. Allowlist defined for upstreaming; non-allowlisted archs fall back to plain Hadamard.
- **Works (allowlist):** qwen2/3/3moe/35/35moe (Qwen3.5 is hybrid — verified working), llama/llama4, mistral/mixtral, gemma2/3 (with QK-norm placement verification), glm4/glm4_moe, phi2/3, yi, internlm2/3, granite, stablelm, command-r
- **Same mechanism, unverified:** other hybrids (jamba, falcon_h1, lfm2, granite_hybrid, nemotron_h, kimi_linear, qwen3next, plamo2/3) — they share the `llama-memory-recurrent.h` + `is_recr_impl[]` infrastructure with Qwen3.5, so should work identically, just need a smoke test
- **Doesn't apply:** deepseek2/32, glm_dsa (MLA — KV is compressed latent, not per-head)
- **Format breaks:** gemma3n, gemma4 (per-layer variable head_dim, KV layer reuse)
- Qwen3.5 (user's model) is the hybrid variant: 6 attention layers out of 24 (interval=4), head_dim=256, n_kv_head=2. Verified via `gguf_dump.py`.

**Why:** Mean-only mode is the right upstreaming target — proven better than full PCA on logit KL at scale, faster on GPU, equivalent or better cache fidelity, drops the entire PCA calibration pipeline (no `.pt`, no `.bin`, no covariance/eigendecomposition).

**How to apply:** Read `pca_poc/progress.md` (now ~610 lines, has full state including CPU + GPU + KL validation, ablation results, all dump dirs, the V Hadamard bug history, and open work). The V Hadamard rotation uses blocked H_64 (not H_256) — fixed in `verify_simulation.py` and `ablation_rotations.py` but still broken in `kv_dashboard.py`, `form_factor.py`, `variance_per_dim.py`. The CPU-buffer PCA tensor risk on GPU did NOT materialize — ggml scheduler handles it cleanly. Open work after KL validation: Q8/Q3/Q2 KL sweep, long-context test, dashboard V Hadamard fix, upstreaming prep.
