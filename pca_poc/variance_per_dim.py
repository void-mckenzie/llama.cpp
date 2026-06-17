"""
Per-dimension variance distribution — how concentrated is the variance, really?

Static PCA (calibrated offline, applied at runtime) only sorts variance
descending *approximately* — the calibration data isn't the same as the
inference prompt. This script shows how concentrated the variance actually is
for each rotation, on the real dumped K/V data.

Useful to dispel the misconception "PCA always concentrates variance in the
first few dims." That's true if you recompute PCA on the same data. For static
PCA, the concentration is approximate and depends on how well calibration
matches inference.

Usage:
    C:/Users/MukkeshGanesh/miniconda3/envs/kvpress_env/python.exe F:/Repos/llama.cpp/pca_poc/variance_per_dim.py [dump_dir] [layer] [mode]

    dump_dir  default F:\\Repos\\llama.cpp\\kv_dumps_math
    layer     default 3
    mode      default K  (K or V)
"""

import sys
from pathlib import Path

import numpy as np
import torch


PCA_PT = Path(r"C:\Users\MukkeshGanesh\Downloads\kv_pca_standalone\pca_scales.pt")
HEAD_DIM = 256
N_KV_HEAD = 2


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


def load_pca_scales(layer: int):
    raw = torch.load(str(PCA_PT), map_location="cpu", weights_only=False)
    s = raw[layer]
    return {
        "W_k":    s["W_k"].   float().numpy(),
        "W_v":    s["W_v"].   float().numpy(),
        "mean_k": s["mean_k"].float().numpy(),
        "mean_v": s["mean_v"].float().numpy(),
    }


def find_runs(dump_dir: Path):
    import json
    runs = {}
    for d in sorted(dump_dir.iterdir()):
        if not d.is_dir() or not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        kv = meta["kv_cache"]
        runs[(kv["pca_enabled"], kv["type_k"])] = d
    return runs


def find_divergence(M, baseline, noise_floor_mult=3.0):
    n = min(M.shape[0], baseline.shape[0])
    if n < 32:
        return n
    per_tok = np.linalg.norm(M[:n].astype(np.float64) - baseline[:n].astype(np.float64), axis=1)
    median = float(np.median(per_tok[:min(n, 64)]))
    if median < 1e-9:
        median = max(float(per_tok[:min(n, 64)].max()), 1e-6)
    threshold = median * noise_floor_mult
    for i in range(n):
        if per_tok[i] > threshold:
            return i
    return n


def back_rotate_hadamard(M, H256):
    n = M.shape[0]
    out = np.zeros_like(M, shape=(n, N_KV_HEAD, HEAD_DIM))
    Mr = M.reshape(n, N_KV_HEAD, HEAD_DIM)
    for h in range(N_KV_HEAD):
        out[:, h, :] = Mr[:, h, :] @ H256
    return out.reshape(n, -1)


def back_rotate_pca(M, W, mean):
    n = M.shape[0]
    out = np.zeros_like(M, shape=(n, N_KV_HEAD, HEAD_DIM))
    Mr = M.reshape(n, N_KV_HEAD, HEAD_DIM)
    for h in range(N_KV_HEAD):
        out[:, h, :] = Mr[:, h, :] @ W[h] + mean[h]
    return out.reshape(n, -1)


def variance_summary(M_per_head: np.ndarray) -> dict:
    """M_per_head shape (n_tokens, head_dim). Returns variance stats."""
    v = M_per_head.var(axis=0)
    total = float(v.sum())
    order = np.argsort(v)[::-1]
    return {
        "total":        total,
        "top_dims":     v[order[:5]],
        "bottom_dims":  v[order[-5:]],
        "top_bottom_ratio": float(v.max() / max(v.min(), 1e-12)),
        "top10_pct":    float(v[order[:10]].sum() / total),
        "top32_pct":    float(v[order[:32]].sum() / total),
        "top64_pct":    float(v[order[:64]].sum() / total),
        "bottom128_pct": float(v[order[128:]].sum() / total),
    }


def run(dump_dir: Path, layer: int, mode: str):
    runs = find_runs(dump_dir)
    if (False, "f16") not in runs:
        print(f"ERROR: no F16 baseline in {dump_dir}")
        sys.exit(1)

    base_dir = runs[(False, "f16")]
    f16 = np.load(base_dir / f"{mode}_layer_{layer}.npy").astype(np.float32)
    H256 = hadamard_matrix(HEAD_DIM)
    scales = load_pca_scales(layer)
    W = scales["W_k"] if mode == "K" else scales["W_v"]
    mean = scales["mean_k"] if mode == "K" else scales["mean_v"]

    print(f"dump_dir: {dump_dir}")
    print(f"layer:    {layer}")
    print(f"mode:     {mode}")
    print(f"n_tokens: {f16.shape[0]} (full; not truncated for variance analysis)")
    print()

    # Also load Q4 Had to find divergence truncation for fairness
    if (False, "q4_0") in runs:
        q4had = np.load(runs[(False, "q4_0")] / f"{mode}_layer_{layer}.npy").astype(np.float32)
        q4had_orig = back_rotate_hadamard(q4had, H256)
        div_idx = find_divergence(f16, q4had_orig)
        if div_idx < f16.shape[0]:
            print(f"  (divergence detected at token {div_idx}, but variance analysis uses full F16 K)")
    print()

    for h in range(N_KV_HEAD):
        K = f16.reshape(f16.shape[0], N_KV_HEAD, HEAD_DIM)[:, h, :]
        K_had = K @ H256
        K_pca = (K - mean[h]) @ W[h].T

        s_base = variance_summary(K)
        s_had  = variance_summary(K_had)
        s_pca  = variance_summary(K_pca)

        print(f"Head {h} — per-dim variance distribution:")
        print()
        print(f"  {'rotation':<18} {'total':>9} {'top dim':>9} {'bot dim':>9} {'ratio':>7} "
              f"{'top10%':>8} {'top32%':>8} {'top64%':>8} {'bot128%':>9}")
        print("  " + "-" * 92)
        for label, s in [("F16 baseline", s_base), ("Hadamard", s_had), ("PCA", s_pca)]:
            top_str = ", ".join(f"{x:.2f}" for x in s["top_dims"][:1])
            bot_str = ", ".join(f"{x:.3f}" for x in s["bottom_dims"][-1:])
            print(f"  {label:<18} {s['total']:>9.2f} {top_str:>9} {bot_str:>9} "
                  f"{s['top_bottom_ratio']:>6.0f}x {s['top10_pct']*100:>7.1f}% "
                  f"{s['top32_pct']*100:>7.1f}% {s['top64_pct']*100:>7.1f}% "
                  f"{s['bottom128_pct']*100:>8.1f}%")

        print()
        print(f"  Top 5 dims by variance:")
        for label, M, s in [("F16", K, s_base), ("Had", K_had, s_had), ("PCA", K_pca, s_pca)]:
            print(f"    {label}: " + ", ".join(f"{x:.2f}" for x in s["top_dims"]))
        print(f"  Bottom 5 dims by variance:")
        for label, M, s in [("F16", K, s_base), ("Had", K_had, s_had), ("PCA", K_pca, s_pca)]:
            print(f"    {label}: " + ", ".join(f"{x:.3f}" for x in s["bottom_dims"]))
        print()
        print(f"  Insights:")
        print(f"    - F16 baseline top/bottom ratio: {s_base['top_bottom_ratio']:.0f}x "
              f"(>100x indicates known transformer K-cache outlier dimensions)")
        print(f"    - Hadamard flattens to {s_had['top_bottom_ratio']:.0f}x; PCA to {s_pca['top_bottom_ratio']:.0f}x")
        print(f"    - Static PCA's concentration is approximate -- variance ordering only holds")
        print(f"      exactly if PCA is recomputed on the same data we're looking at.")
        print(f"    - The relevant metric for Q4_0 quantization is per-block form factor,")
        print(f"      not overall variance concentration. See form_factor.py.")
        print()


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
