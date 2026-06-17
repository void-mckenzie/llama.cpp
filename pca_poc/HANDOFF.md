# PCA KV Cache Rotation — POC Handoff

## TL;DR

Implementing **PCA-based KV cache rotation** in llama.cpp, inspired by the Python research repo at `C:\Users\MukkeshGanesh\Downloads\kv_pca_standalone`. The math works end-to-end (verified on Qwen3.5-2B with F16 cache). The Q4_0/Q8_0 quantized cache path crashes in the ggml scheduler due to a tensor buffer-assignment problem that hasn't been solved yet. **This is the current blocker.**

---

## 1. Background: the Python repo

Location: `C:\Users\MukkeshGanesh\Downloads\kv_pca_standalone`

Python research repo exploring PCA-based KV cache compression as an alternative to scalar quantization. Tested on Qwen3.5-2B.

**Key files:**
- `compress.py` — compressors. The relevant class is `StaticPCAQuantCompressor` (lines 481–554): loads pre-calibrated per-layer PCA matrices from `pca_scales.pt`, projects K/V into PCA space, quantizes, reconstructs.
- `calibrate_pca.py` — produces `pca_scales.pt` by streaming calibration data through the model, accumulating per-head K/V covariance, eigendecomposing.
- `pca_scales.pt` — torch file with per-layer data.

**Python benchmark results** (from `sweep_stats.json`), 4-bit quantization, Qwen3.5-2B:
| Method | KL | Token agreement |
|---|---|---|
| `kvarn_poc` (Hadamard + Sinkhorn + quant) | 11.28 | 31.25% |
| `static_pca_quant` (calibrated PCA + quant) | **7.42** | **56.25%** |
| `mixed_static_pca` (per-dim bit schedule) | 11.40 | 29.69% |
| `pca_quant` (on-the-fly per-batch PCA) | 10.82 | 34.38% |

**Headline:** `static_pca_quant` is 1.8× better than Hadamard at 4 bits. Mixed precision actually hurt. Pre-calibration is essential (on-the-fly is worse).

---

## 2. The `.pt` file structure

Verified by loading with torch:
```
top-level: dict keyed by layer_idx
layer indices: [3, 7, 11, 15, 19, 23]   # exactly 6 attention layers in Qwen3.5-2B's hybrid arch
per layer: {
    W_k:   shape [2, 256, 256]  float32  # [n_kv_head, head_dim, head_dim]
    W_v:   shape [2, 256, 256]  float32
    mean_k: shape [2, 256]      float32
    mean_v: shape [2, 256]      float32
}
```

Orthogonality verified: `max ‖W·Wᵀ − I‖_F = 3.49e-06` across all matrices. So `Wᵀ = W⁻¹` — back-rotation with the transpose is correct.

---

## 3. The target model

`C:\Users\MukkeshGanesh\Downloads\Qwen3.5-2B-UD-Q8_K_XL.gguf` (Unsloth quant, ~2.83 GB)

Key metadata:
- `general.architecture = qwen35` (hybrid: linear-attention/SSM + full attention)
- `qwen35.block_count = 24`, `qwen35.full_attention_interval = 4`
- Attention layers: `(i+1) % 4 == 0` → **i = 3, 7, 11, 15, 19, 23** (matches `.pt` keys)
- Other 18 layers are recurrent/SSM (no KV cache)
- `qwen35.attention.head_count = 8` (Q heads)
- `qwen35.attention.head_count_kv = 2` (GQA 4:1)
- `qwen35.attention.key_length = 256`
- `qwen35.attention.value_length = 256`
- Qwen3.5 attention build uses joint Q+gate projection, per-head Q/K RMSNorm, MRoPE

`qwen35` is supported in llama.cpp: `LLM_ARCH_QWEN35` in `src/llama-arch.h`, `src/models/qwen35.cpp`.

---

## 4. The math

**Cache stores** (per KV head h):
```
K_c = W_k[h] · (K − μ_k[h])
V_c = W_v[h] · (V − μ_v[h])
```

**Read path:**
- Rotate Q with `W_k[h]` (no centering): `Q_rot = W_k[h] · Q`
- Attention logit: `Q_rot · K_c = (W_k Q) · (W_k (K − μ_k)) = Q · (K − μ_k)` (since W_k orthogonal) `= Q·K − Q·μ_k`
- The term `Q·μ_k` is per-query constant → softmax absorbs it (softmax is shift-invariant along the key axis)
- Attention output is in V-rotated space: `softmax · V_c = softmax · W_v · (V − μ_v) = (softmax · V − μ_v) · W_v` (since softmax sums to 1 and μ_v is constant over tokens)
- **Back-rotation:** `out · W_vᵀ + μ_v` recovers the original attention output

**mul_mat direction in ggml** (carefully derived, see §6):
- `mul_mat(W_stored_as_is, x)` = `W @ x` (FORWARD)
- `mul_mat(W_t_stored, x)` = `Wᵀ @ x` (BACKWARD) — where W_t is the per-head transpose of W

**Per-head slicing for GQA:** group_size = `n_head_q / n_kv_head`. For Qwen3.5-2B: group_size = 4. Each KV head's W applies to 4 contiguous Q heads.

---

## 5. What was implemented

### 5.1 Python `.pt` → `.bin` converter

File: `F:\Repos\llama.cpp\pca_poc\pca_dump.py`

Run with kvpress_env python: `C:/Users/MukkeshGanesh/miniconda3/envs/kvpress_env/python.exe`

Output: `C:\Users\MukkeshGanesh\Downloads\kv_pca_standalone\pca_scales.bin` (6.02 MiB)

Binary format (little-endian):
```
magic[8]      = "PCAKV001"
version       = uint32(1)
n_layers      = uint32
n_kv_head     = uint32
head_dim      = uint32
for each layer:
    layer_idx = uint32
    W_k       = float32[n_kv_head * head_dim * head_dim]  # [n_kv_head, D_in, D_out] row-major
    W_v       = float32[...]
    mean_k    = float32[n_kv_head * head_dim]
    mean_v    = float32[...]
```

Script also verifies orthogonality and round-trips.

### 5.2 C++ changes

**Files modified:**
- `src/llama-kv-cache.h`
- `src/llama-kv-cache.cpp`
- `src/llama-graph.cpp`

**Public accessors added to `llama_kv_cache`:**
```cpp
struct pca_layer_data {
    ggml_tensor * W_k   = nullptr;  // F32 [head_dim, head_dim, n_kv_head]
    ggml_tensor * W_v   = nullptr;
    ggml_tensor * W_v_t = nullptr;  // precomputed per-head transpose of W_v
    ggml_tensor * mean_k = nullptr; // F32 [head_dim, n_kv_head]
    ggml_tensor * mean_v = nullptr;
};

bool pca_has(int32_t il) const;
const pca_layer_data * pca_get(int32_t il) const;
```

Same accessors on `llama_kv_cache_context` (delegates to `llama_kv_cache`).

**Loader (`llama_kv_cache::load_pca`)** triggered by env var `LLAMA_KV_PCA_FILE`. Currently emits loud `fprintf(stderr, ...)` banners. Three-phase:
1. Read all raw data into host `std::vector<float>` per layer
2. Create tensor metadata in `pca_ctx`; allocate CPU backend buffer via `ggml_backend_alloc_ctx_tensors_from_buft(pca_ctx.get(), ggml_backend_cpu_buffer_type())` (stores in `pca_buf`)
3. Memcpy data into allocated tensors; also compute `W_v_t` per head in this phase

**Per-head rotation helper `pca_apply_rotation`** in `src/llama-graph.cpp` (lines ~98–178). Signature:
```cpp
static ggml_tensor * pca_apply_rotation(
    ggml_context * ctx, ggml_tensor * cur, ggml_tensor * W, ggml_tensor * mean,
    bool subtract_mean, bool add_mean,
    int head_dim, int n_head, int n_kv_head, int n_tokens);
```

Implementation: loops over `n_kv_head`, slices `cur` via `ggml_view_3d` (group_size heads per KV head), `ggml_cont` + `ggml_reshape_2d`, optional mean subtract (broadcast via `ggml_repeat`+`ggml_sub`), `ggml_mul_mat` with per-head `ggml_view_2d` of W, optional mean add, reshape to 3D, `ggml_concat` along dim 1.

**`build_attn` modified** (`src/llama-graph.cpp:2394+`). For each layer with PCA:
- Q rotation: `mul_mat(W_k, Q)` no centering
- K rotation: `sub(mean_k)` then `mul_mat(W_k, K)`
- V rotation: `sub(mean_v)` then `mul_mat(W_v, V)`
- After `build_attn_mha`: reshape output to 3D, `mul_mat(W_v_t, out)` then `add(mean_v)`, reshape back to 2D

Non-PCA path (Hadamard) untouched — falls through when `pca_get(il) == nullptr`.

---

## 6. mul_mat direction derivation (IMPORTANT — easy to get backwards)

Worked example, D=2, head h=0, math W = `[[1,2],[3,4]]` (W[e,d] = entry at row e, col d).

`.pt` row-major flat: `[1, 2, 3, 4]` (W[0,0]=1, W[0,1]=2, W[1,0]=3, W[1,1]=4).

Create ggml tensor ne=[2,2,1] from this data. ggml element (i0,i1) at flat `i0 + 2·i1`:
- (0,0)=1, (1,0)=2, (0,1)=3, (1,1)=4

`ggml_mul_mat(A, B)` computes `out[i,j] = Σ_k A.ggml(k,i) · B[k,j]`.

For our A:
- `out[0,j] = 1·B[0,j] + 2·B[1,j]`
- `out[1,j] = 3·B[0,j] + 4·B[1,j]`

This matches `W @ B[:,j]` in math. **So `mul_mat(W_stored_as_is, x) = W @ x` (FORWARD).**

For back-rotation `Wᵀ @ x`: we precompute `W_v_t` (numpy per-head transpose, `W_v_t[h,e,d] = W_v[h,d,e]`) and store it the same way. `mul_mat(W_t_stored, x) = Wᵀ @ x` (BACKWARD). Verified.

---

## 7. Test results so far

All runs on Qwen3.5-2B, prompt `"The capital of France is"`, `--temp 0 --top-k 1`, CPU-only build.

| # | Cache | Rotation | Result |
|---|---|---|---|
| 1 | F16 | none | ✓ "Paris" |
| 2 | F16 | PCA | ✓ "Paris" (small drift, expected) — **math validated** |
| 3 | Q4_0 | Hadamard | ✓ "Paris" |
| 4 | Q4_0 | PCA | ✗ **crash** — `ggml-alloc.c:623: GGML_ASSERT(buffer_id >= 0)` |
| 5 | Q8_0 | Hadamard | ✓ "Paris" |
| 6 | Q8_0 | PCA | ✗ **crash** — same as #4 |

**The F16 sanity test (#2) passes** — math is correct, integration works.

**Quantized + PCA crashes (#4, #6).** F16 works because cache writes are pure memcpy. Quantized writes use `ggml_set_rows` which triggers the scheduler's full buffer-assignment path.

---

## 8. The blocker: quantized cache + PCA crash

### Symptoms

With **any** quantized cache type (`q4_0`, `q8_0`, etc.) + PCA enabled, llama-cli crashes during context creation (after `load_pca` succeeds, before inference):
```
F:\Repos\llama.cpp\ggml\src\ggml-alloc.c:623: GGML_ASSERT(buffer_id >= 0) failed
```

**ALL quantized cache types break with PCA.** F16 (unquantized) works fine. Quantized cache WITHOUT PCA (Hadamard) works fine. Only the combination fails.

**Why this matters:** the whole point of the POC is to compare PCA vs Hadamard at Q4_0/Q8_0 and reproduce the Python benchmark's 56% vs 31% token-agreement result. Until this crash is fixed, we can't run that comparison.

### What we've tried

1. **Initial loader** stored PCA tensors in raw `std::vector<uint8_t>` with manual `t->data` override → crashed because `t->buffer` was null.
2. **Current loader** allocates PCA tensors via `ggml_backend_alloc_ctx_tensors_from_buft(pca_ctx.get(), ggml_backend_cpu_buffer_type())` → tensors have proper `t->buffer`. F16 path now works. **Quantized path still crashes with the same `buffer_id >= 0` error.**
3. **Tried replacing `ggml_concat` with `ggml_cpy` into views of a pre-allocated output tensor** → broke even F16. Reverted.

### Root cause analysis (incomplete)

The crash is at `ggml-alloc.c:623` inside `ggml_gallocr_allocate_node(galloc, node, buffer_id)` which asserts `buffer_id >= 0`. The `buffer_id` is passed in from `leaf_buffer_ids[i]` for each leaf in the graph.

`leaf_buffer_ids` is computed by the scheduler in `ggml-backend.cpp`. For each leaf:
```cpp
int * leaf_backend_id = &tensor_backend_id(leaf);
if (*leaf_backend_id == -1) {
    *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
}
```

`ggml_backend_sched_backend_id_from_cur` (line 878) calls `ggml_backend_sched_backend_from_buffer` (line 845) which:
- Gets the tensor's buffer
- Iterates over registered backends, returns the first one that supports the buffer type AND the op

For our PCA tensors, the buffer is CPU type and should be supported. But somehow the result is still -1 for some leaf in the graph when the cache is quantized.

**Key observation:** F16 path works. So the PCA tensors themselves aren't the problem (they're the same in both cases). The issue is that the QUANTIZED graph has some additional leaf or interaction that the scheduler can't place.

The crash only happens with the specific combination of PCA rotation + quantized cache write. Either alone works.

### Suspects

1. **`ggml_concat` output combined with `set_rows(quantized)` for cache write.** The concat output is a fresh intermediate tensor. Maybe its interaction with the quantize-on-write path creates a state the scheduler can't resolve. (Hadamard's `ggml_mul_mat_aux` produces a different intermediate shape — `reshape_4d` of `mul_mat` output, not concat.)

2. **The meta backend (`ggml-backend-meta.cpp`) analysis.** We previously saw a different error from meta analysis: `ggml-backend.cpp:194: GGML_ASSERT(buffer) failed` — meaning some tensor had null buffer during meta analysis. The error changed after the loader fix, so meta might still be involved. The meta analyzer runs specifically when `set_rows` needs to determine split state for the quantize-on-write op.

3. **PCA tensors' buffer-usage flag.** `ggml_backend_alloc_ctx_tensors_from_buft` may set `buffer->usage = COMPUTE` by default. The meta analyzer at `ggml-backend-meta.cpp:789` checks `usage != COMPUTE && view_src == nullptr` — if our PCA buffer has COMPUTE usage, it might take a different code path that doesn't handle the situation correctly. Model weights use `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` — different code path.

### Things to try next (in rough order of priority) — ALL aimed at making quantized cache work

1. **Set the PCA buffer usage to WEIGHTS.** After `ggml_backend_alloc_ctx_tensors_from_buft`, call `ggml_backend_buffer_set_usage(pca_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS)`. This matches how model weights are marked and may satisfy the scheduler/meta analyzer. **Smallest possible change, highest probability of fixing.**

2. **Add diagnostics to pinpoint the failing leaf.** Patch `ggml_gallocr_allocate_node` (ggml-alloc.c:622) or `ggml_backend_sched_backend_id_from_cur` (ggml-backend.cpp:878) to print `tensor->name` and `tensor->op` when `buffer_id == -1`. Will immediately tell us which tensor is the problem. **Highest diagnostic value.**

3. **Switch to the graph-input pattern for PCA tensors.** This is how Hadamard works (it's the canonical pattern). Create per-layer PCA input tensors in `ctx0` (graph context), mark via `ggml_set_input`, fill via a `set_input_*_rot`-style callback. Implementation requires extending `llm_graph_input_attn_kv` to hold per-layer PCA tensors (currently it has shared `self_k_rot`/`self_v_rot`). More work but guarantees scheduler compatibility because it mirrors the working Hadamard path exactly.

4. **Replace `ggml_concat` with a single tiled `ggml_mul_mat`.** Build a block-diagonal W matrix at load time: `[n_kv_head·D, n_kv_head·D]` with per-head W[h] on the diagonal blocks. For Q (group_size > 1), tile each W[h] group_size times along the diagonal. One mul_mat per layer instead of n_kv_head matmuls + concat. Avoids whatever concat-related issue might exist.

5. **Move PCA tensors into the kv cache's existing CPU context** (`ctxs_bufs[0].first`) instead of a separate `pca_ctx`. They'd share the kv cache's buffer and might be treated consistently with the K/V cache tensors themselves.

**Do not pursue:** anything that avoids quantized cache (e.g., restricting to F16, or using unquantized intermediate buffers). The POC's value depends on validating PCA's benefit under quantization. If approach #1 or #2 doesn't reveal a quick fix, #3 (graph-input pattern) is the safest path because it mirrors the working Hadamard code.

---

## 9. How to run what's there

Build (CPU only, Debug):
```
cd F:\Repos\llama.cpp
cmake -B build -DGGML_NATIVE=ON -DLLAMA_NATIVE=ON
cmake --build build --target llama-cli -j
```

Python converter (one-time):
```
"C:/Users/MukkeshGanesh/miniconda3/envs/kvpress_env/python.exe" F:/Repos/llama.cpp/pca_poc/pca_dump.py
```

Baseline (F16, no PCA):
```
set LLAMA_KV_PCA_FILE=
build\bin\Debug\llama-cli.exe -m "C:\Users\MukkeshGanesh\Downloads\Qwen3.5-2B-UD-Q8_K_XL.gguf" -p "The capital of France is" --temp 0 --top-k 1
```

PCA enabled:
```
set LLAMA_KV_PCA_FILE=C:\Users\MukkeshGanesh\Downloads\kv_pca_standalone\pca_scales.bin
build\bin\Debug\llama-cli.exe -m "C:\Users\MukkeshGanesh\Downloads\Qwen3.5-2B-UD-Q8_K_XL.gguf" -p "The capital of France is" --temp 0 --top-k 1
```

Quantized cache (e.g. q4_0): append `-ctk q4_0 -ctv q4_0` (lowercase required).

Three stderr lines print on startup when PCA loads:
```
llama_kv_cache::llama_kv_cache: === kv cache constructed (PCA env check) LLAMA_KV_PCA_FILE=... ===
llama_kv_cache::load_pca: === PCA KV cache rotation POC ===
llama_kv_cache::load_pca: LLAMA_KV_PCA_FILE = ...
llama_kv_cache::load_pca: === PCA rotation ACTIVE: 6 layers, n_kv_head=2, head_dim=256 ===
```

(The kv cache ctor runs twice during context creation — once for the main cache, once for warmup/MTP. Log appears twice. Not a bug.)

---

## 10. Key file locations in the codebase

| What | Where |
|---|---|
| kv cache class | `src/llama-kv-cache.h`, `src/llama-kv-cache.cpp` |
| Hadamard rotation existing impl | `src/llama-kv-cache.cpp:333-373` (precompute), `:1778-1794` (set_input) |
| Graph input for KV | `src/llama-graph.h:305+` (`llm_graph_input_attn_kv`) |
| `ggml_mul_mat_aux` (Hadamard helper) | `src/llama-graph.cpp:64-82` |
| `pca_apply_rotation` (new) | `src/llama-graph.cpp:98-178` |
| `build_attn` (modified) | `src/llama-graph.cpp:2394+` |
| `build_attn_inp_kv_impl` | `src/llama-graph.cpp:2361+` |
| `cpy_k` / cache write | `src/llama-kv-cache.cpp:1463+` |
| Qwen3.5 attention build | `src/models/qwen35.cpp:255-334` |
| Qwen3.5 hybrid layer decision | `src/models/qwen35.cpp:21-27` |
| Scheduler buffer assignment | `ggml/src/ggml-backend.cpp:845-933` (backend_id_from_cur, backend_from_buffer) |
| Graph allocator leaf loop | `ggml/src/ggml-alloc.c:717-810` |
| Meta backend split state | `ggml/src/ggml-backend-meta.cpp:785-1051` |

---

## 11. Task list state

- [x] Python `.pt` → `.bin` converter
- [x] PCA struct + loader in `llama_kv_cache`
- [x] Per-head rotation helper
- [x] `build_attn` PCA path
- [x] F16 sanity test (PCA + F16 ≈ baseline F16)
- [ ] **Q4_0 vs Hadamard comparison** — blocked by scheduler crash
- [ ] (deferred) `--kv-pca` CLI flag in `common/arg.cpp` (currently env-var only)
- [ ] (deferred) Remove `fprintf(stderr)` diagnostics once stable
- [ ] (deferred) GPU buffer placement (currently CPU-only)
- [ ] (deferred) Calibration tool `llama-pca-cal` (currently using Python)

---

## 12. Conversation context to keep

- The user prefers discussing design before coding ("let's think", "be thorough"). Don't jump straight to implementation without alignment.
- User's environment: Windows 11, cmd shell, Python at `C:/Users/MukkeshGanesh/miniconda3/envs/kvpress_env/python.exe` (kvpress_env conda env).
- llama.cpp was built CPU-only (no CUDA) for the POC. CUDA rebuild was deferred because PCA tensors live in CPU host memory and would need GPU placement work.
- User rejected `--no-cnv` (doesn't exist) and prefers minimal CLI flags. Default sampling args + `--temp 0 --top-k 1` for deterministic output.
- The thinking traces differ slightly between F16 baseline and F16 PCA runs (e.g. typo "Paris.cw" appears in both sometimes). This is expected F16 drift under greedy sampling, not a bug.
- The repeated "kv cache constructed" log is because the kv cache is built twice during context creation (main + warmup/MTP layer). Cosmetic, not a bug.


---

# Update — 2026-06-15: quantized cache crash FIXED, POC validated end-to-end

This addendum covers everything done in the session that followed the original handoff. The blocker described in §7–§8 is **resolved**. PCA now works for F16, Q4_0, and Q8_0 caches. Read this addendum first; the original handoff (§1–§12) is background.

## §13. Root cause of the quantized crash

**TL;DR:** Hadamard rotation in llama.cpp is enabled *only* for quantized caches. When PCA was also enabled, both ran together — `build_attn_inp_kv_impl` created the Hadamard matrix leaf (`self_k_rot`), but the PCA branch in `build_attn` skipped `mul_mat_aux`, leaving `self_k_rot` as a **dead leaf** with no consumer. The scheduler therefore didn't allocate a buffer for it. Later, `set_input_k_rot` (called from `llm_graph_input_mem_hybrid::set_input`) dereferenced the null buffer and crashed.

### Why F16+PCA worked but Q4_0/Q8_0+PCA crashed

In `llama-kv-cache.cpp:334-349`, the flags are:
```cpp
attn_rot_k = ... && ggml_is_quantized(type_k) && hparams.n_embd_head_k() % 64 == 0;
attn_rot_v = ... && ggml_is_quantized(type_v) && hparams.n_embd_head_v() % 64 == 0;
```
So Hadamard is OFF for F16 (no conflict with PCA — F16+PCA worked). For any quantized cache, Hadamard is ON, which collided with PCA.

### How the root cause was found (the diagnostic path that worked)

1. **Tried fix #1 from §8** — set PCA buffer usage to `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`. The crash changed from `ggml-alloc.c:623 buffer_id >= 0` to `ggml-backend.cpp:194 GGML_ASSERT(buffer)`. Progress: meant a tensor had a null buffer somewhere.
2. **Diagnostic at `ggml-backend-meta.cpp:789`** to print when `tensor->buffer == nullptr` in `calculate_split_state`. No output → wrong call site.
3. **Diagnostic at meta.cpp lines 1830/1946** for null `view_src->buffer`. No output → wrong call site.
4. **Instrumented all 5 helpers** in `ggml-backend.cpp` that internally call `ggml_backend_buffer_get_type` (`name`, `get_alignment`, `get_max_size`, `get_alloc_size`, `is_host`). Each logged which helper was called with null buffer. Identified **`ggml_backend_buffer_is_host`** as the failing helper.
5. **Added backtrace via `CaptureStackBackTrace`** (Windows) — got 16 raw addresses but no symbols.
6. **Per-frame module info via `GetModuleHandleExA` + `GetModuleFileNameA`** — gave `module+0xNNN` for each frame. Frame [0] was in `llama.dll+0x1d9b8c` (not `ggml-base.dll` as expected).
7. **Disassembled `llama.dll`** with `dumpbin /disasm:nobytes` (path: `C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64/dumpbin.exe`).
8. **Resolved the offsets** using `C:/Users/MukkeshGanesh/miniconda3/envs/kvpress_env/python.exe` (native Windows Python — must use `C:\` paths, not `/c/`). Map symbols to addresses by parsing disasm for `<symbol>:\n  <addr>:` patterns. ImageBase = `0x180000000`.
9. Frame [0] resolved to **`llama_kv_cache_context::set_input_k_rot`** at `llama-kv-cache.cpp:1955`. Confirmed by reading the code: line 1955 asserts `is_host(dst->buffer)` unconditionally.

### The fix

Single-line change at end of `llama_kv_cache` constructor (after `load_pca`), `llama-kv-cache.cpp` around line 387:
```cpp
if (pca_enabled) {
    if (attn_rot_k || attn_rot_v) {
        fprintf(stderr, "%s: PCA enabled — suppressing Hadamard attention rotation (attn_rot_k=%d attn_rot_v=%d -> 0)\n",
                __func__, attn_rot_k, attn_rot_v);
        fflush(stderr);
    }
    attn_rot_k = false;
    attn_rot_v = false;
}
```

### WEIGHTS flag turned out to be unnecessary

The `ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS)` line added during step-1 debugging was a red herring — it only masked the dead-leaf issue by changing the failure mode. Once the dead-leaf root cause was fixed, removing the WEIGHTS line still works. **WEIGHTS flag has been removed**. The PCA buffer is now allocated with default (`ANY`) usage.

## §14. Diagnostic instrumentation (all reverted)

The following temporary diagnostics were added during root-cause hunting and **all reverted** before validating the fix:
- 5 helper-level null-buffer diagnostics in `ggml-backend.cpp` (`is_host`, `name`, `get_alignment`, `get_max_size`, `get_alloc_size`)
- Null-buffer diagnostic at `ggml-backend-meta.cpp:789` in `calculate_split_state`
- Null-buffer diagnostics at meta.cpp lines 1830 and 1946 (`view_src->buffer == nullptr`)

Current `ggml-backend.cpp` and `ggml-backend-meta.cpp` should be **identical to upstream master**. Verify with `git diff ggml/` before any PR.

## §15. Updated test results (2026-06-15)

All runs on Qwen3.5-2B, CPU-only Debug build, default sampling (`--temp 0 --top-k 1`).

### Test A: "The capital of France is" (6-token prompt)

| # | Cache | Rotation | Result |
|---|---|---|---|
| 1 | F16 | none | OK "Paris" |
| 2 | F16 | PCA | OK "Paris" |
| 3 | Q4_0 | Hadamard (default) | OK "Paris" |
| 4 | Q4_0 | PCA | OK "Paris" |
| 5 | Q8_0 | Hadamard (default) | OK "Paris" |
| 6 | Q8_0 | PCA | OK "Paris" |

### Test B: GSM8K "Janet's ducks" math problem

Command template:
```
-sys "Give the final answer as #### followed by the numerical answer." -p "Janet's ducks..." --temp 0 --top-k 1 --reasoning off [-ctk q4_0 -ctv q4_0]
```

| # | Cache | Rotation | Output | Notes |
|---|---|---|---|---|
| 1 | F16 | none | `#### 18` | ground truth |
| 2 | Q4_0 | Hadamard | `#### 18` | cosmetic wording diff vs run 1 |
| 3 | F16 | PCA | `#### 18` | **byte-identical to run 1** |
| 4 | Q8_0 | PCA | `#### 18` | one cosmetic line fusion |
| 5 | Q4_0 | PCA | `#### 18` | **byte-identical to run 2** |

**Key takeaway:** F16 with/without PCA produces byte-identical output, confirming the PCA math is sound. On these prompts, Q4_0 Hadamard and Q4_0 PCA are also indistinguishable — but the prompts are too easy to discriminate them. The Python paper's claim of 56% vs 31% token agreement at 4 bits has **not yet been reproduced in llama.cpp**.

### Commands used (CLI flags)

- Disable Qwen3 thinking: `--reasoning off`
- System prompt (short form): `-sys "<text>"` (NOT `--sys-prompt`)
- Cache types (lowercase): `-ctk q4_0 -ctv q4_0`
- PCA env var: `set LLAMA_KV_PCA_FILE=C:\Users\MukkeshGanesh\Downloads\kv_pca_standalone\pca_scales.bin` (or `set LLAMA_KV_PCA_FILE=` to clear)

## §16. Open task — KV cache state dumper (NOT YET STARTED, task #10)

User wants: on `/exit` in llama-cli, dump the live KV cache state to disk so Python can analyze it (verify what's stored, compute quantization error, compare PCA vs Hadamard vs no-rotation at the byte level).

### Design decided in conversation

- **Trigger:** env var `LLAMA_KV_DUMP_DIR` (when set, dumping happens on exit)
- **Layout:** `<dir>/pca_{true|false}_ctk_{type}_ctv_{type}_<timestamp>/` per run
- **Inside the run dir:**
  - `K_layer_<i>.npy` per kv-cache layer (numpy `.npy` format, F32, little-endian)
  - `V_layer_<i>.npy` per kv-cache layer
  - `meta.json` with everything else
- **Data representation: (b) dequantized F32 in PCA-rotated space** — i.e. read the cache through ggml's dequant path, write as F32 in the natural cache layout (no back-rotation to original space; Python can do that using the W matrices from `pca_scales.pt`)
- **Meta.json contents:** prompt, system prompt, model name/path, cache types (k/v), generation params (temp, top_k, etc.), PCA-on flag, PCA layer indices, n_head/n_kv_head/head_dim, n_tokens_prefill, n_tokens_generated, **full raw output text** of the conversation, timestamp, llama.cpp build hash
- **Scope:** all kv-cache layers (for Qwen3.5: the 6 full-attention layers; the SSM layers don't have KV cache)

### Not yet decided

- How to access the kv cache from llama-cli's `/exit` handler — need to find where the handler is (`tools/main/`) and what's in scope there (llama_context, common params, conversation history).
- Whether to expose the dump method as a `llama_kv_cache::dump_state(dir, meta_json)` method called explicitly from llama-cli, or to have the kv cache destructor read metadata from env vars / globals.
- Subdir timestamp format (ISO-8601 local? `%Y%m%d_%H%M%S`?).
- Whether to dump one file per layer or one combined `.npz`.

### Suggested next steps for this task

1. Use Explore agent (read-only) to map the `/exit` handler in `tools/main/`, what's in scope, and how to get `llama_kv_cache *` from `llama_context *`.
2. Implement `llama_kv_cache::dump_to_dir(dir)` that writes `.npy` files (simple F32 .npy writer; format is magic + version + header_len + header_dict + raw data).
3. Hook from llama-cli exit handler: build subdir name from PCA flag + cache types + timestamp, build meta.json, call dump_to_dir.
4. Test: run a few prompts, load the .npy in Python (kvpress_env), verify shapes look right.

## §17. Updated task list state

| # | Status | Task |
|---|---|---|
| #2 | pending | Add `--kv-pca` CLI flag in `common/arg.cpp` (currently env-var only) |
| #3 | done | Add PCA struct + loader to `llama_kv_cache` |
| #4 | done | Quantized-cache comparison (smoke level — both Q4_0 and Q8_0+PCA produce correct answers) |
| #5 | done | F16 sanity test (PCA + F16 byte-identical to no-rotation F16) |
| #6 | done | Modify `build_attn` for per-head PCA rotation |
| #7 | done | Verify WEIGHTS buffer usage flag is still needed → **not needed, removed** |
| #8 | done | Decide Hadamard suppression scope → **global, matches existing llama.cpp pattern** (Hadamard is already a global flag, not per-layer; `self_k_rot` is one shared tensor) |
| #9 | pending | Gate PCA stderr diagnostics behind `LLAMA_KV_CACHE_DEBUG` |
| #10 | **open** | KV cache state dumper (see §16) |

## §18. Memories saved for future agents

The following feedback memories are persisted in `C:\Users\MukkeshGanesh\.claude\projects\F--Repos-llama-cpp\memory\` and indexed in `MEMORY.md`. Read these before starting work — they represent explicit user corrections:

- **`feedback_no_run_llama_cli.md`** — Don't invoke `llama-cli` (or similar end-to-end inference CLIs) via the Bash tool. Hand the user the command to run; they run it and paste output back.
- **`feedback_native_windows_paths.md`** — kvpress_env Python (and other native Windows binaries) need `C:\` paths, not Git-Bash mounts like `/c/` or `/tmp/`. Python from conda is native Windows, not Cygwin.
- **`feedback_llama_cli_sys_flag.md`** — System prompt flag in llama-cli is `-sys`, not `--sys-prompt`.
- **`feedback_qwen_disable_thinking.md`** — Disable Qwen3 thinking with `--reasoning off`, not `--chat-template-kwargs "{...}"`.

User preferences also covered in §12 still apply (discuss before coding, Windows + cmd shell, CPU-only build for POC, minimal CLI flags, `--temp 0 --top-k 1` for determinism, "kv cache constructed" log appears twice because main cache + warmup/MTP).

## §19. Reference dumpbin / Python recipe (for future debugging)

In case the next agent needs to resolve runtime addresses again:

```bash
DUMPBIN='C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64/dumpbin.exe'
PY='C:/Users/MukkeshGanesh/miniconda3/envs/kvpress_env/python.exe'

# 1. Disassemble the DLL (use /c/temp style for bash redirection, C:\temp\ for native Python):
"$DUMPBIN" //disasm:nobytes 'F:\Repos\llama.cpp\build\bin\Debug\llama.dll' > /c/temp/llama_disasm.txt

# 2. Resolve offsets (ImageBase for llama.dll = 0x180000000):
"$PY" - <<'PYEOF'
import re
with open(r'C:\temp\llama_disasm.txt') as f:
    text = f.read()
sym_pattern = re.compile(r'^([^\s].+?):\n\s+([0-9A-F]+):', re.MULTILINE)
symbols = sorted([(int(m.group(2),16), m.group(1)) for m in sym_pattern.finditer(text)])
IMAGE_BASE = 0x180000000
for f_off in [0x1d9b8c, 0x19b910]:  # your offsets here
    target = IMAGE_BASE + f_off
    best = max((a for a, _ in symbols if a <= target), default=None)
    if best:
        name = next(n for a, n in symbols if a == best)
        print(f'llama.dll+0x{f_off:x} -> delta=+0x{target-best:x} in {name[:150]}')
PYEOF
```

For frame backtrace with module info, the `CaptureStackBackTrace(2, N, frames, NULL)` + `GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)frames[i], &hmod)` + `GetModuleFileNameA(hmod, path, MAX_PATH)` pattern worked well.
