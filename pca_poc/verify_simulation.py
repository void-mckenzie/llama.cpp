"""
Verify the Q4_0 simulation against real llama-cli output for a given dump dir.

Loads F16 baseline + Q4_0 Hadamard + Q4_0 PCA runs from a dump dir, finds the
token where runs diverge (per-token L2 spike), truncates to the pre-divergence
prefix, then compares pure-numpy Q4_0 simulation to real cached values for
both Hadamard and PCA.

Usage:
    C:/Users/MukkeshGanesh/miniconda3/envs/kvpress_env/python.exe F:/Repos/llama.cpp/pca_poc/verify_simulation.py [dump_dir] [layer] [mode]

    dump_dir  default F:\\Repos\\llama.cpp\\kv_dumps_math
    layer     default 3
    mode      default K  (K or V)

Example:
    python verify_simulation.py F:\\Repos\\llama.cpp\\kv_dumps 7 V
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch


PCA_PT = Path(r"C:\Users\MukkeshGanesh\Downloads\kv_pca_standalone\pca_scales.pt")
HEAD_DIM = 256
N_KV_HEAD = 2
QK = 32  # Q4_0 block size

# Module-level Hadamard matrices (built lazily to avoid import-time cost if unused)
_H256 = None
_H64 = None

def _H_matrices():
    global _H256, _H64
    if _H256 is None:
        _H256 = hadamard_matrix(HEAD_DIM)
        _H64  = hadamard_matrix(64)
    return _H256, _H64


# ---- Q4_0 simulation (matches ggml's quantize_row_q4_0) ----

def quantize_dequantize_q4_0(M: np.ndarray) -> np.ndarray:
    """M shape (n_rows, multiple_of_32). Per-row block Q4_0 quant+dequant, returns F32."""
    assert M.shape[1] % QK == 0, f"M.shape[1]={M.shape[1]} not divisible by {QK}"
    out = np.zeros_like(M, dtype=np.float32)
    for b in range(M.shape[1] // QK):
        block = M[:, b * QK:(b + 1) * QK]
        amax_idx = np.argmax(np.abs(block), axis=1)
        signed_max = block[np.arange(block.shape[0]), amax_idx]
        d = signed_max / -8.0
        d_safe = np.where(d == 0, 1.0, d)
        scaled = block * (1.0 / d_safe)[:, None]
        q = np.clip(np.floor(scaled + 8.5).astype(np.int32), 0, 15)
        out[:, b * QK:(b + 1) * QK] = d[:, None] * (q - 8).astype(np.float32)
    return out


# ---- rotations ----

def hadamard_matrix(n: int) -> np.ndarray:
    H = np.zeros((n, n), dtype=np.float32)
    H[0, 0] = 1.0 / np.sqrt(n)
    s = 1
    while s < n:
        for i in range(s):
            for j in range(s):
                v = H[i, j]
                H[i + s, j] = v
                H[i, j + s] = v
                H[i + s, j + s] = -v
        s *= 2
    return H


def apply_hadamard(M: np.ndarray, mode: str) -> np.ndarray:
    """Apply the correct Hadamard rotation for K or V cache.

    K uses nrot = head_dim (H_256 for Qwen3.5) — dense rotation.
    V uses nrot = 64 fixed (see src/llama-kv-cache.cpp:1786 build_input_v_rot) —
    block-diagonal H_64 applied to 4 sub-blocks of 64 dims each.

    Symmetric + orthonormal in both cases, so forward = inverse.

    M shape: (n_rows, head_dim=256) per head.
    """
    if mode == "K":
        return M @ _H256
    # V: blocked H_64
    n = M.shape[0]
    return (M.reshape(n, 4, 64) @ _H64).reshape(n, HEAD_DIM)


def load_pca_scales(layer: int):
    raw = torch.load(str(PCA_PT), map_location="cpu", weights_only=False)
    s = raw[layer]
    return {
        "W_k":    s["W_k"].   float().numpy(),
        "W_v":    s["W_v"].   float().numpy(),
        "mean_k": s["mean_k"].float().numpy(),
        "mean_v": s["mean_v"].float().numpy(),
    }


# ---- dump loaders ----

def find_runs(dump_dir: Path):
    """Return dict: (pca_enabled: bool, ctk: str) -> subdir Path."""
    runs = {}
    for d in sorted(dump_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        kv = meta["kv_cache"]
        runs[(kv["pca_enabled"], kv["type_k"])] = d
    return runs


def load_k_or_v(subdir: Path, layer: int, mode: str) -> np.ndarray:
    return np.load(subdir / f"{mode}_layer_{layer}.npy").astype(np.float32)


# ---- back-rotation ----

def back_rotate(M: np.ndarray, layer: int, mode: str, pca_enabled: bool, ctk: str,
                pca_scales: dict, H256: np.ndarray) -> np.ndarray:
    """Take stored values and back-rotate to original K/V space.

    M shape: (n_tokens, n_kv_head * head_dim). Returns same shape.

    NOTE: For Hadamard, K uses H_256 (full head_dim) but V uses H_64 applied in
    4 sub-blocks (matches src/llama-kv-cache.cpp:1786 build_input_v_rot which
    fixes nrot=64 for V). The `H256` argument is now ignored for V — we use the
    module-level _H64 instead.
    """
    n_tokens = M.shape[0]
    Mr = M.reshape(n_tokens, N_KV_HEAD, HEAD_DIM)
    out = np.zeros_like(Mr)
    if pca_enabled and layer in {int(k) for k in __import__("torch").load(str(PCA_PT), map_location="cpu", weights_only=False).keys()}:
        W    = pca_scales["W_k"] if mode == "K" else pca_scales["W_v"]
        mean = pca_scales["mean_k"] if mode == "K" else pca_scales["mean_v"]
        for h in range(N_KV_HEAD):
            # forward was (K-mean) @ W.T  ->  inverse is K_pca @ W + mean
            out[:, h, :] = Mr[:, h, :] @ W[h] + mean[h]
    elif ctk != "f16":
        # Hadamard: K uses H_256 dense, V uses H_64 block-diagonal
        for h in range(N_KV_HEAD):
            out[:, h, :] = apply_hadamard(Mr[:, h, :], mode)
    else:
        out = Mr  # F16 baseline already in original space
    return out.reshape(n_tokens, -1)


# ---- divergence detection ----

def find_divergence(K_base: np.ndarray, K_other: np.ndarray, noise_floor_mult: float = 2.5) -> int:
    """Find first token where per-token L2 diff exceeds noise_floor_mult * median.

    Both inputs shape (n_tokens, n_dims). Returns the index of first divergence,
    or min(n_tokens) if no divergence detected.
    """
    n = min(K_base.shape[0], K_other.shape[0])
    per_tok = np.linalg.norm(K_base[:n] - K_other[:n], axis=1)
    # noise floor = typical Q4_0 quantization noise per token, ~1.9 for head_dim=256
    median = np.median(per_tok[:min(n, 64)])  # use first 64 tokens as reference (should be prefill)
    threshold = median * noise_floor_mult
    for i in range(n):
        if per_tok[i] > threshold:
            return i
    return n


# ---- main ----

def run(dump_dir: Path, layer: int, mode: str):
    runs = find_runs(dump_dir)
    if (False, "f16") not in runs:
        print(f"ERROR: no F16 baseline run found in {dump_dir}")
        sys.exit(1)
    if (False, "q4_0") not in runs or (True, "q4_0") not in runs:
        print(f"ERROR: need both Q4_0 Hadamard (False, q4_0) and Q4_0 PCA (True, q4_0) runs")
        print(f"       found: {list(runs.keys())}")
        sys.exit(1)

    base_dir   = runs[(False, "f16")]
    q4had_dir  = runs[(False, "q4_0")]
    q4pca_dir  = runs[(True, "q4_0")]

    print(f"dump_dir:   {dump_dir}")
    print(f"layer:      {layer}")
    print(f"mode:       {mode}")
    print(f"baseline:   {base_dir.name}")
    print(f"q4 Hadamard: {q4had_dir.name}")
    print(f"q4 PCA:      {q4pca_dir.name}")
    print()

    f16   = load_k_or_v(base_dir,  layer, mode)
    q4had = load_k_or_v(q4had_dir, layer, mode)
    q4pca = load_k_or_v(q4pca_dir, layer, mode)
    print(f"n_tokens: F16={f16.shape[0]}  Q4Had={q4had.shape[0]}  Q4PCA={q4pca.shape[0]}")

    H256 = hadamard_matrix(HEAD_DIM)
    _H_matrices()  # populate module-level _H256 / _H64 for apply_hadamard
    pca_scales = load_pca_scales(layer)

    # back-rotate Q4 Hadamard to original space for divergence detection
    q4had_orig = back_rotate(q4had, layer, mode, pca_enabled=False, ctk="q4_0",
                              pca_scales=pca_scales, H256=H256)
    div_idx = find_divergence(f16, q4had_orig)
    # round down to a multiple of 32 for clean Q4_0 block alignment
    N = (div_idx // 32) * 32
    if N < 32:
        N = 32
    print(f"first divergence at token {div_idx} (L2 spike); truncating to N={N} tokens")
    print()

    # per-head sim vs real
    print(f"  {'method':<20} {'head':>4} {'sim':>10} {'real':>10} {'ratio':>8}")
    print("  " + "-" * 56)
    rows = []
    for h in range(N_KV_HEAD):
        K = f16[:N].reshape(N, N_KV_HEAD, HEAD_DIM)[:, h, :]

        # Hadamard sim (K uses H_256, V uses H_64 sub-blocks)
        K_had_recon = apply_hadamard(quantize_dequantize_q4_0(apply_hadamard(K, mode)), mode)
        sim_had = float(np.sqrt(np.mean((K_had_recon - K) ** 2)))
        # Hadamard real
        K_had_real = back_rotate(q4had, layer, mode, pca_enabled=False, ctk="q4_0",
                                  pca_scales=pca_scales, H256=H256)[:N].reshape(N, N_KV_HEAD, HEAD_DIM)[:, h, :]
        real_had = float(np.sqrt(np.mean((K_had_real - K) ** 2)))
        print(f"  {'Q4_0 Had':<20} {h:>4} {sim_had:>10.4f} {real_had:>10.4f} {sim_had/max(real_had,1e-9):>8.3f}")

        # PCA sim (always full W matrix per head)
        W_h = pca_scales["W_k"][h] if mode == "K" else pca_scales["W_v"][h]
        mean_h = pca_scales["mean_k"][h] if mode == "K" else pca_scales["mean_v"][h]
        K_pca = (K - mean_h) @ W_h.T
        K_pca_recon = quantize_dequantize_q4_0(K_pca) @ W_h + mean_h
        sim_pca = float(np.sqrt(np.mean((K_pca_recon - K) ** 2)))
        # PCA real
        K_pca_real = back_rotate(q4pca, layer, mode, pca_enabled=True, ctk="q4_0",
                                  pca_scales=pca_scales, H256=H256)[:N].reshape(N, N_KV_HEAD, HEAD_DIM)[:, h, :]
        real_pca = float(np.sqrt(np.mean((K_pca_real - K) ** 2)))
        print(f"  {'Q4_0 PCA':<20} {h:>4} {sim_pca:>10.4f} {real_pca:>10.4f} {sim_pca/max(real_pca,1e-9):>8.3f}")

        ratio_sim = sim_pca / sim_had
        ratio_real = real_pca / real_had
        print(f"  {'PCA/Had ratio':<20} {h:>4} {ratio_sim:>10.4f} {ratio_real:>10.4f}")
        print()
        rows.append((sim_had, sim_pca, real_had, real_pca))

    avg_sim_had  = np.mean([r[0] for r in rows])
    avg_sim_pca  = np.mean([r[1] for r in rows])
    avg_real_had = np.mean([r[2] for r in rows])
    avg_real_pca = np.mean([r[3] for r in rows])
    print("Head-averaged:")
    print(f"  Had: sim={avg_sim_had:.4f}  real={avg_real_had:.4f}  match ratio={avg_sim_had/max(avg_real_had,1e-9):.4f}")
    print(f"  PCA: sim={avg_sim_pca:.4f}  real={avg_real_pca:.4f}  match ratio={avg_sim_pca/max(avg_real_pca,1e-9):.4f}")
    print(f"  PCA improvement over Had: {(1 - avg_real_pca / avg_real_had) * 100:.1f}% (real), {(1 - avg_sim_pca / avg_sim_had) * 100:.1f}% (sim)")


def main():
    dump_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"F:\Repos\llama.cpp\kv_dumps_math")
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    mode = sys.argv[3] if len(sys.argv) > 3 else "K"
    if mode not in ("K", "V"):
        print(f"mode must be K or V, got {mode}")
        sys.exit(1)
    run(dump_dir, layer, mode)


if __name__ == "__main__":
    main()
