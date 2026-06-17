#!/usr/bin/env python3
"""
Generate variant .bin files for mean-only mode ablation.

All variants derive from the original PCA .pt (calibrated on calibration_datav5.txt).
W_k / W_v are kept UNCHANGED (mean-only mode ignores them anyway). Only mean_k and
mean_v are transformed. Each variant tests one hypothesis about what makes the
PCA-derived per-head mean effective.

Output layout matches pca_dump.py (PCAKV001 format).

Usage:
    python pca_variant_dump.py [--pt <path>] [--out_dir <path>] [--seed N]
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

MAGIC = b"PCAKV001"
VERSION = 1
EXPECTED_KEYS = ("W_k", "W_v", "mean_k", "mean_v")


def load_pt(pt_path: Path):
    scales = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    layer_indices = sorted(int(k) for k in scales.keys())
    return scales, layer_indices


def write_bin(scales, layer_indices, bin_path: Path):
    n_kv_head, head_dim, _ = scales[layer_indices[0]]["W_k"].shape
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bin_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", VERSION, len(layer_indices), n_kv_head, head_dim))
        for li in layer_indices:
            f.write(struct.pack("<I", li))
            for key in EXPECTED_KEYS:
                arr = np.ascontiguousarray(scales[li][key].float().numpy(), dtype=np.float32)
                f.write(arr.tobytes())
    print(f"  wrote {bin_path} ({bin_path.stat().st_size / 1024:.0f} KiB)")


def variant_zero(scales, layer_indices, rng):
    """mean_k = mean_v = 0. Sanity control — should match plain Hadamard."""
    out = {li: dict(scales[li]) for li in layer_indices}
    for li in layer_indices:
        for key in ("mean_k", "mean_v"):
            out[li][key] = torch.zeros_like(scales[li][key])
    return out


def variant_rand_dir(scales, layer_indices, rng):
    """Keep per-head L2 norm, replace direction with random unit vector.
    Tests whether direction matters or only magnitude."""
    out = {li: dict(scales[li]) for li in layer_indices}
    for li in layer_indices:
        for key in ("mean_k", "mean_v"):
            m = scales[li][key].numpy()  # (n_kv_head, head_dim)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            r = rng.standard_normal(m.shape)
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            out[li][key] = torch.tensor((r * norms).astype(np.float32))
    return out


def variant_rand_gauss(scales, layer_indices, rng):
    """Random Gaussian per (head, dim), scaled to match global std of real means.
    Tests whether per-head magnitude structure matters."""
    out = {li: dict(scales[li]) for li in layer_indices}
    all_vals = np.concatenate([
        scales[li][key].numpy().ravel()
        for li in layer_indices for key in ("mean_k", "mean_v")
    ])
    std = float(np.std(all_vals))
    for li in layer_indices:
        for key in ("mean_k", "mean_v"):
            shape = scales[li][key].shape
            out[li][key] = torch.tensor((rng.standard_normal(shape) * std).astype(np.float32))
    return out


def variant_shuffle_heads(scales, layer_indices, rng):
    """Shuffle mean vectors across heads within each layer.
    Tests whether the per-head mean-to-head correspondence matters."""
    out = {li: dict(scales[li]) for li in layer_indices}
    for li in layer_indices:
        for key in ("mean_k", "mean_v"):
            m = scales[li][key].numpy()
            perm = rng.permutation(m.shape[0])
            out[li][key] = torch.tensor(m[perm].astype(np.float32))
    return out


def variant_per_layer_avg(scales, layer_indices, rng):
    """Collapse head dimension: average across heads, broadcast back.
    Tests whether per-head specificity matters."""
    out = {li: dict(scales[li]) for li in layer_indices}
    for li in layer_indices:
        for key in ("mean_k", "mean_v"):
            m = scales[li][key].numpy()  # (n_kv_head, head_dim)
            avg = m.mean(axis=0, keepdims=True)
            out[li][key] = torch.tensor(np.broadcast_to(avg, m.shape).copy().astype(np.float32))
    return out


VARIANTS = {
    "zero":          variant_zero,
    "rand_dir":      variant_rand_dir,
    "rand_gauss":    variant_rand_gauss,
    "shuffle":       variant_shuffle_heads,
    "per_layer_avg": variant_per_layer_avg,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pt", default="C:/Users/MukkeshGanesh/Downloads/kv_pca_standalone/pca_scales.pt")
    p.add_argument("--out_dir", default="F:/Repos/llama.cpp/kl_dumps/variants")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    args = p.parse_args()

    scales, layer_indices = load_pt(Path(args.pt))
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)

    print(f"loaded {args.pt}: {len(layer_indices)} layers")
    for vname in args.variants:
        print(f"[{vname}]")
        modified = VARIANTS[vname](scales, layer_indices, rng)
        write_bin(modified, layer_indices, out_dir / f"pca_{vname}.bin")


if __name__ == "__main__":
    main()
