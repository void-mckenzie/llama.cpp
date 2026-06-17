"""
Per-block form factor analysis — a heuristic for "why does PCA win at Q4_0?"

Q4_0 quantizes 32-element blocks. Each block has one FP16 scale and 32 4-bit
indices. Per-block SNR depends on the spread of values relative to the scale.

Two relevant form-factor metrics, both averaged across tokens within a block:

  RMS_form  = RMS / max_abs         (the original heuristic)
  std_form  = std / max_abs         (mean-subtracted version)

For centered data (PCA), RMS == std, so both metrics agree.
For uncentered data (Hadamard, F16 baseline), RMS > std because the mean
contributes to RMS but not to the deviation that the Q4_0 scale needs to cover.

The std_form is a better predictor of Q4_0 SNR than RMS_form because the Q4_0
scale is set by signed-max, which has to cover both mean and deviation for
uncentered data.

IMPORTANT: form factor is a *heuristic* for intuition, not a proof. The proof
that PCA beats Hadamard at Q4_0 is in verify_simulation.py, which reproduces
llama-cli's output to 4 decimals with pure-numpy Q4_0 simulation. Sometimes
form factor predicts the winner, sometimes it doesn't — when they disagree,
trust the simulation.

Usage:
    C:/Users/MukkeshGanesh/miniconda3/envs/kvpress_env/python.exe F:/Repos/llama.cpp/poca_poc/form_factor.py [dump_dir] [layer] [mode]

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
QK = 32  # Q4_0 block size


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


def per_block_form_factors(M_per_head: np.ndarray) -> dict:
    """M_per_head shape (n_tokens, head_dim). Returns dict with two lists
    of length 8 (one per Q4_0 block): 'rms' and 'std' form factors."""
    assert M_per_head.shape[1] == HEAD_DIM
    n_blocks = HEAD_DIM // QK
    rms_ff = []
    std_ff = []
    for b in range(n_blocks):
        block = M_per_head[:, b * QK:(b + 1) * QK]  # (n_tokens, 32)
        max_abs = np.abs(block).max(axis=1)
        rms = np.sqrt(np.mean(block ** 2, axis=1))
        std = np.std(block, axis=1)
        rms_ff.append(float((rms / np.maximum(max_abs, 1e-12)).mean()))
        std_ff.append(float((std / np.maximum(max_abs, 1e-12)).mean()))
    return {"rms": rms_ff, "std": std_ff}


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
    print(f"n_tokens: {f16.shape[0]} (full F16 K; analysis is on raw variance, not quantization)")
    print()

    print("std_form (std / max_abs) is the better Q4_0 predictor — Q4_0's scale has to")
    print("cover the mean too for uncentered data, which inflates max_abs without")
    print("adding signal. PCA subtracts the mean first; Hadamard does not.")
    print()

    for h in range(N_KV_HEAD):
        K = f16.reshape(f16.shape[0], N_KV_HEAD, HEAD_DIM)[:, h, :]
        K_had = K @ H256
        K_pca = (K - mean[h]) @ W[h].T

        ff_base = per_block_form_factors(K)
        ff_had  = per_block_form_factors(K_had)
        ff_pca  = per_block_form_factors(K_pca)

        print(f"Head {h}:")
        for metric in ("std", "rms"):
            ff_b = ff_base[metric]; ff_h = ff_had[metric]; ff_p = ff_pca[metric]
            avg_b = float(np.mean(ff_b)); avg_h = float(np.mean(ff_h)); avg_p = float(np.mean(ff_p))
            label = "std / max_abs (better Q4_0 predictor)" if metric == "std" else "RMS / max_abs (original heuristic)"
            print(f"  {label}")
            print(f"    F16 baseline avg: {avg_b:.4f}")
            print(f"    Hadamard     avg: {avg_h:.4f}    PCA vs Had: {(avg_p/avg_h - 1)*100:+.1f}%")
            print(f"    PCA          avg: {avg_p:.4f}")
        print()

        # also show per-block std/max_abs table
        print(f"  Per-block std/max_abs (8 blocks of 32 dims each, head {h}):")
        n_blocks = HEAD_DIM // QK
        header = "    rotation      " + "  ".join(f"b{b}" for b in range(n_blocks)) + "    avg"
        print(header)
        for label, ff in [("F16 baseline", ff_base["std"]),
                          ("Hadamard",     ff_had["std"]),
                          ("PCA",          ff_pca["std"])]:
            cells = "  ".join(f"{x:.3f}" for x in ff)
            print(f"    {label:<14} {cells}  {np.mean(ff):.3f}")
        print()
        print("  Note: form factor is a heuristic. The actual Q4_0 winner is determined")
        print("  by full quantization simulation. Run verify_simulation.py for ground truth.")
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
