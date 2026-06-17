#!/usr/bin/env python3
"""
Generate variant .bin files where per-head means are computed from N-token
prefixes of a llama-cli KV dump. W matrices are kept from the original .pt
unchanged (mean-only mode ignores them anyway).

Tests how few calibration tokens are needed to estimate a good per-head mean.

Layout assumption for the dump:
    K_layer_<il>.npy: shape (n_tokens, n_embd_gqa), F32 row-major
    n_embd_gqa = n_kv_head * head_dim  (for Qwen3.5-2B: 2 * 256 = 512)
    Reshape to (n_tokens, n_kv_head, head_dim) gives [token, head, dim].
    Per-head mean over the first N tokens = mean(arr[:N], axis=0) -> (n_kv_head, head_dim).

Usage:
    python pca_prefix_dump.py [<dump_dir>] [<pt_path>] [<out_dir>] [N1 N2 ...]

Defaults:
    dump_dir = F:/Repos/llama.cpp/kl_dumps/calib_dump
    pt_path  = C:/Users/MukkeshGanesh/Downloads/kv_pca_standalone/pca_scales.pt
    out_dir  = F:/Repos/llama.cpp/kl_dumps/variants
    prefixes = 32 128 512 2048 4096
"""

import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch

MAGIC = b"PCAKV001"
VERSION = 1
DEFAULT_PREFIXES = [32, 128, 512, 2048, 4096]


def load_pt_W(pt_path: Path):
    scales = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    layer_indices = sorted(int(k) for k in scales.keys())
    W = {li: {"W_k": scales[li]["W_k"].float().numpy(),
              "W_v": scales[li]["W_v"].float().numpy()}
         for li in layer_indices}
    return W, layer_indices


def load_dump(dump_dir: Path):
    # Pick most recent subdir if dump_dir contains subdirs
    subdirs = sorted([d for d in dump_dir.iterdir() if d.is_dir()])
    target = subdirs[-1] if subdirs else dump_dir
    print(f"  dump subdir: {target}")

    meta_path = target / "meta.json"
    layer_indices = None
    n_kv_head = None
    head_dim = None
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        kv = meta["kv_cache"]
        layer_indices = list(kv["kv_layer_indices"])
        n_kv_head = kv["n_kv_head"]
        head_dim = kv["head_dim_k"]
        assert kv["head_dim_v"] == head_dim, "head_dim_k != head_dim_v"

    if layer_indices is None:
        npys = list(target.glob("K_layer_*.npy"))
        layer_indices = sorted(int(re.search(r"K_layer_(\d+)\.npy", p.name).group(1))
                               for p in npys)
        n_kv_head = 2
        head_dim = 256

    out = {}
    for li in layer_indices:
        K = np.load(target / f"K_layer_{li}.npy")
        V = np.load(target / f"V_layer_{li}.npy")
        out[li] = {"K": K, "V": V}
    return out, layer_indices, n_kv_head, head_dim


def write_bin(W, prefix_means, layer_indices, n_kv_head, head_dim, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", VERSION, len(layer_indices), n_kv_head, head_dim))
        for li in layer_indices:
            f.write(struct.pack("<I", li))
            for key in ("W_k", "W_v"):
                arr = np.ascontiguousarray(W[li][key], dtype=np.float32)
                f.write(arr.tobytes())
            for key in ("mean_k", "mean_v"):
                arr = np.ascontiguousarray(prefix_means[li][key], dtype=np.float32)
                f.write(arr.tobytes())
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KiB)")


def main():
    args = sys.argv[1:]
    dump_dir = Path(args[0]) if len(args) > 0 else Path("F:/Repos/llama.cpp/kl_dumps/calib_dump")
    pt_path = Path(args[1]) if len(args) > 1 else Path("C:/Users/MukkeshGanesh/Downloads/kv_pca_standalone/pca_scales.pt")
    out_dir = Path(args[2]) if len(args) > 2 else Path("F:/Repos/llama.cpp/kl_dumps/variants")
    prefixes = sorted(set(int(x) for x in args[3:])) if len(args) > 3 else list(DEFAULT_PREFIXES)

    print(f"loading W from {pt_path}")
    W, layer_indices_W = load_pt_W(pt_path)

    print(f"loading dump from {dump_dir}")
    dumps, layer_indices_d, n_kv_head, head_dim = load_dump(dump_dir)

    assert layer_indices_W == layer_indices_d, \
        f"layer mismatch: W={layer_indices_W}, dump={layer_indices_d}"

    n_tokens = dumps[layer_indices_d[0]]["K"].shape[0]
    print(f"dump has {n_tokens} tokens, {len(layer_indices_d)} layers, "
          f"n_kv_head={n_kv_head}, head_dim={head_dim}")

    # Reshape to (n_tokens, n_kv_head, head_dim) for per-head slicing
    for li in layer_indices_d:
        K = dumps[li]["K"]
        V = dumps[li]["V"]
        assert K.shape[1] == n_kv_head * head_dim, \
            f"layer {li}: K shape {K.shape}, expected (*, {n_kv_head * head_dim})"
        dumps[li]["K"] = K.reshape(K.shape[0], n_kv_head, head_dim)
        dumps[li]["V"] = V.reshape(V.shape[0], n_kv_head, head_dim)

    # Clip prefix sizes to available tokens
    prefixes = sorted(set(min(p, n_tokens) for p in prefixes))
    print(f"prefix sizes (clipped to n_tokens): {prefixes}")

    for N in prefixes:
        print(f"[N={N}]")
        prefix_means = {}
        for li in layer_indices_d:
            K = dumps[li]["K"][:N]
            V = dumps[li]["V"][:N]
            prefix_means[li] = {
                "mean_k": K.mean(axis=0),
                "mean_v": V.mean(axis=0),
            }
        write_bin(W, prefix_means, layer_indices_d, n_kv_head, head_dim,
                  out_dir / f"pca_prefix_N{N}.bin")


if __name__ == "__main__":
    main()
