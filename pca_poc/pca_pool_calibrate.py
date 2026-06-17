#!/usr/bin/env python3
"""
Pool-based mean calibration. Mimics calibrate_pca.py's methodology using llama.cpp:

1. Slice calibration text into chunks of ~chunk_size_tokens
2. For each chunk: run llama-cli with F16 cache, dump KV
3. Load dump, accumulate per-layer per-head running sums
4. Delete dump (frees disk)
5. After all chunks: divide sums by total tokens to get per-head means
6. Write .bin (W matrices from existing pca_scales.pt, means from pool)

Streaming design: only per-layer sum_k[head,dim] and sum_v[head,dim] are kept
in memory (~6 KB total). Disk usage stays at one dump subdir at a time.

Usage:
    python pca_pool_calibrate.py --chunk_size_tokens 2048 --max_tokens all --tag exp1_2k_all
    python pca_pool_calibrate.py --chunk_size_tokens 2048 --max_tokens 100000 --tag exp2_2k_100k
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

MAGIC = b"PCAKV001"
VERSION = 1

# Defaults (overridable via CLI)
DEFAULT_CALIB = Path(r"C:/Users/MukkeshGanesh/Downloads/kv_pca_standalone/calibration_datav5.txt")
DEFAULT_PT    = Path(r"C:/Users/MukkeshGanesh/Downloads/kv_pca_standalone/pca_scales.pt")
DEFAULT_CLI   = Path(r"F:/Repos/llama.cpp/build-cuda/bin/Debug/llama-cli.exe")
DEFAULT_MODEL = Path(r"C:/Users/MukkeshGanesh/Downloads/Qwen3.5-2B-BF16.gguf")
DEFAULT_DUMP_BASE = Path(r"F:/Repos/llama.cpp/kl_dumps/pool_dumps")
DEFAULT_OUT_DIR   = Path(r"F:/Repos/llama.cpp/kl_dumps/variants")

# chars/token for this corpus (measured: 16K chars -> 4367 tokens)
CHARS_PER_TOKEN = 3.7

# Subprocess timeout (BF16 model load can be slow on first chunk)
SUBPROCESS_TIMEOUT = 600


def slice_into_chunks(text: str, chunk_chars: int):
    """Split text into chunks of approximately chunk_chars, on paragraph boundaries.
    Falls back to character-level split for oversized paragraphs."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur = []
    cur_size = 0
    for p in paragraphs:
        if len(p) > chunk_chars:
            # flush current first
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_size = [], 0
            # hard-split the oversized paragraph at character level
            for i in range(0, len(p), chunk_chars):
                chunks.append(p[i:i + chunk_chars])
        elif cur_size + len(p) > chunk_chars and cur:
            chunks.append("\n\n".join(cur))
            cur, cur_size = [p], len(p)
        else:
            cur.append(p)
            cur_size += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    # filter empty
    return [c for c in chunks if c.strip()]


def find_dump_subdir(dump_dir: Path, before_names: set) -> Path:
    """Find subdir created after snapshot `before_names`."""
    new_subdirs = [p for p in dump_dir.iterdir() if p.is_dir() and p.name not in before_names]
    if not new_subdirs:
        return None
    # most recent by mtime
    return max(new_subdirs, key=lambda p: p.stat().st_mtime)


def run_llama_cli(chunk_text: str, chunk_idx: int, dump_dir: Path, args) -> Path:
    """Run llama-cli on a single chunk, return path to dump subdir (or None on failure)."""
    prompt_file = dump_dir / f"chunk_{chunk_idx:04d}.txt"
    prompt_file.write_text(chunk_text, encoding="utf-8")

    before = set(p.name for p in dump_dir.iterdir() if p.is_dir())

    env = {
        **os.environ,
        "LLAMA_KV_DUMP_DIR": str(dump_dir),
        "LLAMA_KV_PCA_FILE": "",
        "LLAMA_KV_MEAN_ONLY": "",
    }
    cmd = [
        str(args.cli),
        "-m", str(args.model),
        "-f", str(prompt_file),
        "-n", "0",
        "-c", str(args.chunk_size_tokens + 512),  # margin for template tokens + safety
        "-b", "512",
        "-ub", "512",
        "-ngl", "999",
        "-t", "4",
        "--single-turn", "--reasoning", "off",
        "--temp", "0", "--top-k", "1",
        "--no-warmup",
    ]
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=SUBPROCESS_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"  [chunk {chunk_idx}] TIMEOUT after {SUBPROCESS_TIMEOUT}s")
        return None
    if result.returncode != 0:
        print(f"  [chunk {chunk_idx}] llama-cli FAILED rc={result.returncode}")
        print(f"    stderr tail: {result.stderr[-300:]}")
        return None

    sd = find_dump_subdir(dump_dir, before)
    if sd is None:
        print(f"  [chunk {chunk_idx}] no dump subdir created")
        return None

    # Clean up the prompt file
    try:
        prompt_file.unlink()
    except OSError:
        pass

    return sd


def load_dump_subdir(subdir: Path):
    """Load K/V arrays from a dump subdir. Returns {layer_idx: {"K", "V"}}."""
    meta_path = subdir / "meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    kv = meta["kv_cache"]
    layer_indices = list(kv["kv_layer_indices"])
    n_kv_head = kv["n_kv_head"]
    head_dim = kv["head_dim_k"]
    out = {}
    for li in layer_indices:
        K = np.load(subdir / f"K_layer_{li}.npy")
        V = np.load(subdir / f"V_layer_{li}.npy")
        out[li] = {
            "K": K.reshape(K.shape[0], n_kv_head, head_dim),
            "V": V.reshape(V.shape[0], n_kv_head, head_dim),
        }
    return out, layer_indices, n_kv_head, head_dim


def load_pt_W(pt_path: Path):
    scales = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    layer_indices = sorted(int(k) for k in scales.keys())
    W = {li: {"W_k": scales[li]["W_k"].float().numpy(),
              "W_v": scales[li]["W_v"].float().numpy()}
         for li in layer_indices}
    return W, layer_indices


def write_bin(W, means, layer_indices, n_kv_head, head_dim, out_path: Path):
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
                arr = np.ascontiguousarray(means[li][key], dtype=np.float32)
                f.write(arr.tobytes())
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KiB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--pt", type=Path, default=DEFAULT_PT)
    p.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dump_base", type=Path, default=DEFAULT_DUMP_BASE)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--chunk_size_tokens", type=int, required=True)
    p.add_argument("--max_tokens", default="all", help="'all' or integer token count")
    p.add_argument("--tag", required=True, help="name for dump subdir + .bin file")
    args = p.parse_args()

    # Verify paths
    for name, path in [("calib", args.calib), ("pt", args.pt), ("cli", args.cli),
                       ("model", args.model)]:
        if not path.exists():
            print(f"ERROR: {name} path does not exist: {path}")
            sys.exit(1)

    text = args.calib.read_text(encoding="utf-8")
    total_chars = len(text)
    chunk_chars = int(args.chunk_size_tokens * CHARS_PER_TOKEN)

    if args.max_tokens == "all":
        max_chars = total_chars
        max_label = "all"
    else:
        max_chars = int(int(args.max_tokens) * CHARS_PER_TOKEN)
        max_label = args.max_tokens

    src_text = text[:max_chars]
    chunks = slice_into_chunks(src_text, chunk_chars)
    print(f"=== {args.tag} ===")
    print(f"  chunk_size_tokens={args.chunk_size_tokens}, max_tokens={max_label}")
    print(f"  src_chars={len(src_text)}/{total_chars}, chunk_chars~={chunk_chars}")
    print(f"  -> {len(chunks)} chunks")

    dump_dir = args.dump_base / args.tag
    # Clean slate
    if dump_dir.exists():
        print(f"  cleaning existing dump_dir: {dump_dir}")
        shutil.rmtree(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Streaming accumulators
    sum_k = {}  # per-layer running sum, shape (n_kv_head, head_dim)
    sum_v = {}
    total_tokens = 0
    layer_indices = None
    n_kv_head = None
    head_dim = None
    failures = 0

    t_start = time.time()
    for i, chunk_text in enumerate(chunks):
        sd = run_llama_cli(chunk_text, i, dump_dir, args)
        if sd is None:
            failures += 1
            if failures > 20:
                print("  too many failures (>20), aborting")
                sys.exit(1)
            continue

        # Load + accumulate
        try:
            dump, li, nh, hd = load_dump_subdir(sd)
        except Exception as e:
            print(f"  [chunk {i}] load failed: {e}")
            print(f"    subdir: {sd}")
            failures += 1
            shutil.rmtree(sd, ignore_errors=True)  # clean up failed dump
            if failures > 20:
                print("  too many failures (>20), aborting")
                sys.exit(1)
            continue

        if layer_indices is None:
            layer_indices = li
            n_kv_head = nh
            head_dim = hd
            for l in layer_indices:
                sum_k[l] = np.zeros((n_kv_head, head_dim), dtype=np.float64)
                sum_v[l] = np.zeros((n_kv_head, head_dim), dtype=np.float64)
        else:
            assert li == layer_indices, f"layer mismatch at chunk {i}"
            assert nh == n_kv_head and hd == head_dim

        for l in layer_indices:
            sum_k[l] += dump[l]["K"].sum(axis=0)
            sum_v[l] += dump[l]["V"].sum(axis=0)
        total_tokens += dump[layer_indices[0]]["K"].shape[0]

        # Delete the dump subdir to free disk
        shutil.rmtree(sd, ignore_errors=True)

        # Progress
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(chunks) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(chunks)}] tokens={total_tokens} elapsed={elapsed:.0f}s eta={eta:.0f}s fail={failures}")

    if total_tokens == 0:
        print("ERROR: no tokens accumulated")
        sys.exit(1)

    print(f"  total: {total_tokens} tokens, {failures} failed chunks")

    # Compute means
    means = {}
    for l in layer_indices:
        means[l] = {
            "mean_k": (sum_k[l] / total_tokens).astype(np.float32),
            "mean_v": (sum_v[l] / total_tokens).astype(np.float32),
        }

    # Write .bin (W from existing .pt, means from pool)
    W, _ = load_pt_W(args.pt)
    out_bin = args.out_dir / f"{args.tag}.bin"
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    write_bin(W, means, layer_indices, n_kv_head, head_dim, out_bin)
    print(f"  done: {out_bin}")


if __name__ == "__main__":
    main()
