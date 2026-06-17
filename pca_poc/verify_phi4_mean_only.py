#!/usr/bin/env python3
"""
Verify Phi-4 mean-only mode: pure-numpy Q4_0 simulation vs real cached bytes.

Mean-only forward cache stores:
    K_cache = H_K (K - mean_k)        (per head)
    V_cache = H_V (V - mean_v)        (per head, H_V = block-diag H_64)

H_K is the dense Hadamard matrix for head_dim (H_128 for Phi-4).
H_V is H_64 applied to (head_dim/64) sub-blocks (2 sub-blocks for Phi-4 head_dim=128).
Hadamard matrices are symmetric + orthonormal, so forward = inverse.

Q4_0 quantization is applied to the rotated values.

Back-rotation:
    K = H_K K_cache + mean_k
    V = H_V V_cache + mean_v

Sim vs real comparison:
    - Take F16 K/V per head, apply (forward H, Q4_0, back H + mean) in pure numpy -> sim
    - Take mean-only cached K/V per head, back-rotate -> real
    - Compute RMS vs F16 for both, expect sim/real ratio = 1.000

Usage:
    python verify_phi4_mean_only.py
    python verify_phi4_mean_only.py --f16_dir <subdir> --mean_dir <subdir> --mean_bin <path>
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np


# ============================================================
# Constants
# ============================================================
MAGIC = b"PCAKV001"
QK = 32  # Q4_0 block size


# ============================================================
# Hadamard + Q4_0 sim (matches ggml semantics)
# ============================================================
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


def quantize_dequantize_q4_0(M: np.ndarray) -> np.ndarray:
    """Per-row block Q4_0 quant+dequant. M shape (n_rows, multiple_of_32)."""
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


# ============================================================
# Forward / back rotation helpers (per-head)
# ============================================================
def forward_k(M_per_head: np.ndarray, H_head: np.ndarray) -> np.ndarray:
    """M shape (n_tokens, head_dim). Full head_dim Hadamard."""
    return M_per_head @ H_head


def back_k(M_per_head: np.ndarray, H_head: np.ndarray) -> np.ndarray:
    """Hadamard is symmetric + orthonormal -> inverse = forward."""
    return M_per_head @ H_head


def forward_v_blocked(M_per_head: np.ndarray, H_64: np.ndarray, head_dim: int) -> np.ndarray:
    """M shape (n_tokens, head_dim). V uses H_64 applied to (head_dim/64) sub-blocks."""
    n = M_per_head.shape[0]
    n_sub = head_dim // 64
    return (M_per_head.reshape(n, n_sub, 64) @ H_64).reshape(n, head_dim)


def back_v_blocked(M_per_head: np.ndarray, H_64: np.ndarray, head_dim: int) -> np.ndarray:
    """Same as forward (Hadamard self-inverse)."""
    return forward_v_blocked(M_per_head, H_64, head_dim)


# ============================================================
# Mean .bin loader (PCAKV001 format)
# ============================================================
def load_mean_bin(path: Path):
    """Returns (means: dict[layer_idx -> (mean_k, mean_v)], n_kv_head, head_dim).

    Each mean_k/v has shape (n_kv_head, head_dim). W_k/W_v are skipped (mean-only
    mode uses identity; the loaded values are unused).
    """
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == MAGIC, f"bad magic: {magic!r}, expected {MAGIC!r}"
        version, n_layers, n_kv_head, head_dim = struct.unpack("<IIII", f.read(16))
        assert version == 1, f"unexpected version {version}"

        # W_k and W_v are each n_kv_head * head_dim * head_dim F32 values
        w_bytes = n_kv_head * head_dim * head_dim * 4
        mean_bytes = n_kv_head * head_dim * 4

        means = {}
        for _ in range(n_layers):
            layer_idx = struct.unpack("<I", f.read(4))[0]
            f.seek(w_bytes * 2, 1)  # skip W_k, W_v
            mean_k = np.frombuffer(f.read(mean_bytes), dtype=np.float32).reshape(n_kv_head, head_dim).copy()
            mean_v = np.frombuffer(f.read(mean_bytes), dtype=np.float32).reshape(n_kv_head, head_dim).copy()
            means[layer_idx] = (mean_k, mean_v)
    return means, n_kv_head, head_dim


# ============================================================
# Auto-discover dump subdirs
# ============================================================
def find_dump_subdir(base: Path, pca_enabled: bool, kv_type: str) -> Path:
    """Find the most recent matching subdir in base dir."""
    pca_str = "true" if pca_enabled else "false"
    prefix = f"pca_{pca_str}_ctk_{kv_type}_ctv_{kv_type}_"
    candidates = sorted([p for p in base.iterdir()
                         if p.is_dir() and p.name.startswith(prefix)],
                        key=lambda p: p.name)
    if not candidates:
        sys.exit(f"ERROR: no subdir matching {prefix}* in {base}")
    return candidates[-1]  # latest by name (timestamp suffix)


# ============================================================
# Per-layer verification
# ============================================================
def verify_layer(layer: int, f16_dir: Path, mean_dir: Path,
                 means: dict, H_head: np.ndarray, H_64: np.ndarray,
                 n_kv_head: int, head_dim: int, mode: str):
    """Verify one layer for K or V. Returns (N, div_idx, sim_rms_per_head, real_rms_per_head)."""
    f16 = np.load(f16_dir / f"{mode}_layer_{layer}.npy").astype(np.float32)
    mean_cached = np.load(mean_dir / f"{mode}_layer_{layer}.npy").astype(np.float32)

    n_tokens = min(f16.shape[0], mean_cached.shape[0])
    f16 = f16[:n_tokens]
    mean_cached = mean_cached[:n_tokens]

    # Reshape to per-head view: (n_tokens, n_kv_head, head_dim)
    f16_per_head = f16.reshape(n_tokens, n_kv_head, head_dim)
    mean_cached_per_head = mean_cached.reshape(n_tokens, n_kv_head, head_dim)

    mean_k, mean_v = means[layer]
    means_for_mode = mean_k if mode == "K" else mean_v

    # Back-rotate REAL cached values to original K/V space
    real_back = np.zeros_like(f16_per_head)
    for h in range(n_kv_head):
        stored = mean_cached_per_head[:, h, :]
        if mode == "K":
            real_back[:, h, :] = back_k(stored, H_head) + means_for_mode[h]
        else:
            real_back[:, h, :] = back_v_blocked(stored, H_64, head_dim) + means_for_mode[h]

    # Find first divergence token: where real_back starts to differ from F16
    # (above 2.5x the typical quantization noise floor)
    diff = real_back - f16_per_head
    per_tok_norm = np.linalg.norm(diff.reshape(n_tokens, -1), axis=1)
    ref_window = min(n_tokens, 64)
    median = float(np.median(per_tok_norm[:ref_window]))
    threshold = median * 2.5
    div_idx = n_tokens
    for i in range(n_tokens):
        if per_tok_norm[i] > threshold:
            div_idx = i
            break

    # Truncate to multiple of QK for clean Q4_0 block alignment
    N = (div_idx // QK) * QK
    if N < QK:
        N = QK

    # Sim + real RMS per head
    sim_rms = []
    real_rms = []
    for h in range(n_kv_head):
        K = f16_per_head[:N, h, :]
        mean_h = means_for_mode[h]

        if mode == "K":
            K_rot = forward_k(K - mean_h, H_head)
            K_back = back_k(quantize_dequantize_q4_0(K_rot), H_head) + mean_h
        else:
            K_rot = forward_v_blocked(K - mean_h, H_64, head_dim)
            K_back = back_v_blocked(quantize_dequantize_q4_0(K_rot), H_64, head_dim) + mean_h

        sim_rms.append(float(np.sqrt(np.mean((K_back - K) ** 2))))
        real_rms.append(float(np.sqrt(np.mean((real_back[:N, h, :] - K) ** 2))))

    return N, div_idx, sim_rms, real_rms


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f16_dir", type=Path,
                   default=Path(r"F:\Repos\llama.cpp\kv_dumps_phi4"),
                   help="Base dir containing F16 dump subdir")
    p.add_argument("--mean_dir", type=Path,
                   default=Path(r"F:\Repos\llama.cpp\kv_dumps_phi4"),
                   help="Base dir containing mean-only dump subdir")
    p.add_argument("--mean_bin", type=Path,
                   default=Path(r"F:\Repos\llama.cpp\kl_dumps\variants\phi4mini_mean.bin"))
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Layers to verify (default: all in mean_bin)")
    args = p.parse_args()

    means, n_kv_head, head_dim = load_mean_bin(args.mean_bin)
    print(f"Loaded means from {args.mean_bin.name}:")
    print(f"  {len(means)} layers, n_kv_head={n_kv_head}, head_dim={head_dim}")
    print()

    f16_subdir = find_dump_subdir(args.f16_dir, pca_enabled=False, kv_type="f16")
    mean_subdir = find_dump_subdir(args.mean_dir, pca_enabled=True, kv_type="q4_0")
    print(f"F16 baseline: {f16_subdir.name}")
    print(f"Mean-only Q4_0: {mean_subdir.name}")
    print()

    H_head = hadamard_matrix(head_dim)
    H_64 = hadamard_matrix(64)

    layers = args.layers if args.layers else sorted(means.keys())

    print(f"{'L':>3} {'mode':>4} {'N':>4} {'div':>4} "
          f"{'sim_rms':>9} {'real_rms':>9} {'ratio':>6}  per-head sim | per-head real")
    print("-" * 100)

    all_sim = []
    all_real = []
    for layer in layers:
        for mode in ["K", "V"]:
            N, div_idx, sim_rms, real_rms = verify_layer(
                layer, f16_subdir, mean_subdir, means, H_head, H_64,
                n_kv_head, head_dim, mode
            )
            avg_sim = float(np.mean(sim_rms))
            avg_real = float(np.mean(real_rms))
            ratio = avg_sim / max(avg_real, 1e-9)

            sim_str = " ".join(f"{v:.4f}" for v in sim_rms)
            real_str = " ".join(f"{v:.4f}" for v in real_rms)

            print(f"{layer:>3} {mode:>4} {N:>4} {div_idx:>4} "
                  f"{avg_sim:>9.4f} {avg_real:>9.4f} {ratio:>6.3f}  [{sim_str}] | [{real_str}]")

            all_sim.extend(sim_rms)
            all_real.extend(real_rms)

    print("-" * 100)
    avg_sim = float(np.mean(all_sim))
    avg_real = float(np.mean(all_real))
    ratio = avg_sim / max(avg_real, 1e-9)
    print(f"All layers, both K and V: sim={avg_sim:.4f}  real={avg_real:.4f}  ratio={ratio:.4f}")
    print()
    if 0.95 < ratio < 1.05:
        print("PASS: simulation matches real cached bytes within 5%.")
        print("Mean-only mode implementation is correct; high Phi-4 KL is fundamental")
        print("to KV distribution + Q4_0 quantization, not an implementation bug.")
    elif ratio < 0.95:
        print(f"FAIL: sim < real by {(1-ratio)*100:.1f}% — sim too optimistic, investigate back-rotation")
    else:
        print(f"FAIL: sim > real by {(ratio-1)*100:.1f}% — sim too pessimistic, investigate forward rotation")


if __name__ == "__main__":
    main()
