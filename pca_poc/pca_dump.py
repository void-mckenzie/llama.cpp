#!/usr/bin/env python3
"""
Convert pca_scales.pt (PyTorch) -> flat .bin for the llama.cpp PCA KV cache POC.

Binary layout (little-endian):
    magic           = 8 bytes: b"PCAKV001"
    version         = uint32
    n_layers        = uint32   (number of calibrated layers)
    n_kv_head       = uint32
    head_dim        = uint32
    for each layer (in ascending layer_idx order):
        layer_idx   = uint32
        W_k         = float32[n_kv_head * head_dim * head_dim]   # [n_kv_head, D, D] row-major
        W_v         = float32[n_kv_head * head_dim * head_dim]
        mean_k      = float32[n_kv_head * head_dim]
        mean_v      = float32[n_kv_head * head_dim]

Usage:
    python pca_dump.py [<pt_path>] [<bin_path>]

Defaults:
    pt_path  = C:/Users/MukkeshGanesh/Downloads/kv_pca_standalone/pca_scales.pt
    bin_path = <pt_path with .bin suffix>
"""

import struct
import sys
from pathlib import Path

import numpy as np
import torch

MAGIC = b"PCAKV001"
VERSION = 1
EXPECTED_KEYS = ("W_k", "W_v", "mean_k", "mean_v")


def pt_to_bin(pt_path: Path, bin_path: Path) -> None:
    print(f"loading {pt_path}")
    scales = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    if not isinstance(scales, dict):
        raise TypeError(f"expected dict at top level, got {type(scales).__name__}")

    layer_indices = sorted(int(k) for k in scales.keys())
    print(f"layer indices: {layer_indices}  (n_layers={len(layer_indices)})")

    first = scales[layer_indices[0]]
    for k in EXPECTED_KEYS:
        if k not in first:
            raise KeyError(f"layer {layer_indices[0]} missing key {k!r}")

    n_kv_head, head_dim, head_dim2 = first["W_k"].shape
    if head_dim != head_dim2:
        raise ValueError(f"W_k must be square, got {first['W_k'].shape}")
    print(f"inferred n_kv_head={n_kv_head}, head_dim={head_dim}")

    for li in layer_indices:
        s = scales[li]
        if s["W_k"].shape != (n_kv_head, head_dim, head_dim):
            raise ValueError(f"layer {li}: W_k shape {s['W_k'].shape}")
        if s["W_v"].shape != (n_kv_head, head_dim, head_dim):
            raise ValueError(f"layer {li}: W_v shape {s['W_v'].shape}")
        if s["mean_k"].shape != (n_kv_head, head_dim):
            raise ValueError(f"layer {li}: mean_k shape {s['mean_k'].shape}")
        if s["mean_v"].shape != (n_kv_head, head_dim):
            raise ValueError(f"layer {li}: mean_v shape {s['mean_v'].shape}")

    # Orthogonality sanity: W[h] @ W[h].T should be ~ I
    worst = 0.0
    for li in layer_indices:
        for name in ("W_k", "W_v"):
            W = scales[li][name].float().numpy()
            for h in range(n_kv_head):
                eye_err = float(np.linalg.norm(W[h] @ W[h].T - np.eye(head_dim)))
                worst = max(worst, eye_err)
    print(f"orthogonality check: max ||W W^T - I||_F = {worst:.3e}")
    if worst > 1e-3:
        print("WARN: rotation matrices are not orthogonal — back-rotation algebra breaks")

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bin_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(layer_indices)))
        f.write(struct.pack("<I", n_kv_head))
        f.write(struct.pack("<I", head_dim))
        for li in layer_indices:
            s = scales[li]
            f.write(struct.pack("<I", li))
            for key in EXPECTED_KEYS:
                arr = np.ascontiguousarray(s[key].float().numpy(), dtype=np.float32)
                f.write(arr.tobytes())

    size = bin_path.stat().st_size
    expected = 8 + 4 * 4 + len(layer_indices) * (
        4 + 2 * n_kv_head * head_dim * head_dim * 4 + 2 * n_kv_head * head_dim * 4
    )
    print(f"wrote {bin_path} ({size} bytes = {size/1024/1024:.2f} MiB); expected {expected}")

    verify(pt_path, bin_path, layer_indices, n_kv_head, head_dim)


def verify(pt_path: Path, bin_path: Path, expected_layers, expected_h, expected_d) -> None:
    scales_orig = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    with open(bin_path, "rb") as f:
        magic = f.read(8)
        assert magic == MAGIC, f"bad magic on readback: {magic!r}"
        version, n_layers, n_kv_head, head_dim = struct.unpack("<IIII", f.read(16))
        assert version == VERSION, f"bad version: {version}"
        assert n_layers == len(expected_layers), f"n_layers mismatch: {n_layers}"
        assert n_kv_head == expected_h, f"n_kv_head mismatch: {n_kv_head}"
        assert head_dim == expected_d, f"head_dim mismatch: {head_dim}"

        max_err = 0.0
        for expected_li in expected_layers:
            li = struct.unpack("<I", f.read(4))[0]
            assert li == expected_li, f"layer mismatch: {li} != {expected_li}"
            for key in EXPECTED_KEYS:
                shape = (n_kv_head, head_dim, head_dim) if key.startswith("W") else (n_kv_head, head_dim)
                nbytes = int(np.prod(shape)) * 4
                arr = np.frombuffer(f.read(nbytes), dtype=np.float32).reshape(shape)
                orig = np.ascontiguousarray(scales_orig[li][key].float().numpy())
                err = float(np.abs(arr - orig).max())
                max_err = max(max_err, err)
        # ensure nothing trailing
        trailing = f.read(1)
        assert not trailing, f"unexpected trailing bytes in {bin_path}"
    print(f"verify OK: round-trip max abs diff vs .pt = {max_err:.3e}")


if __name__ == "__main__":
    pt_path = Path(sys.argv[1] if len(sys.argv) > 1 else
                   "C:/Users/MukkeshGanesh/Downloads/kv_pca_standalone/pca_scales.pt")
    bin_path = Path(sys.argv[2] if len(sys.argv) > 2 else pt_path.with_suffix(".bin"))
    pt_to_bin(pt_path, bin_path)
