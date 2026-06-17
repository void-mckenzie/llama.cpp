#!/usr/bin/env python3
"""
End-to-end Qwen 3.5 (or similar) mean-only KL divergence experiment.

Given a BF16 GGUF and N quantized GGUFs, runs the full pipeline:
  1. Calibrate per-head K/V means from BF16 GGUF using a calibration text corpus
     (default: 200K tokens, 2K-token chunks — matches Exp 2 sweet spot).
  2. Generate F16-cache logits baselines for each model (BF16 + each quant).
  3. Run KL tests: each model x {Q4_0, Q8_0} x {plain Hadamard, mean-only},
     tested against both universal BF16+F16 baseline and same-model F16 baseline
     (for quantized models).
  4. Generate 3 Plotly charts + 2 printed tables.

Skips work that's already done (resumable).

Usage:
    python run_qwen35_experiment.py \\
        --bf16_gguf   C:/Users/.../Qwen3.5-2B-BF16.gguf \\
        --quant_ggufs C:/Users/.../Qwen3.5-2B-UD-Q8_K_XL.gguf C:/Users/.../Qwen3.5-2B-UD-Q4_K_XL.gguf \\
        --calib       C:/Users/.../calibration_datav5.txt \\
        --llama_dir   F:/Repos/llama.cpp \\
        --build_dir   F:/Repos/llama.cpp/build-cuda \\
        --eval_file   F:/Repos/llama.cpp/scripts/wikitext-2-raw/wiki.test.raw \\
        --tag         qwen35_2b
"""
import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# Constants
# ============================================================
MAGIC = b"PCAKV001"
VERSION = 1
CHARS_PER_TOKEN = 3.7  # measured for calibration_datav5.txt
SUBPROCESS_TIMEOUT = 600  # BF16 loads can be slow


# ============================================================
# Helpers
# ============================================================
def extract_metric(name, text):
    parts = [re.escape(p) for p in name.split(" ") if p]
    pattern = r"\s+".join(parts)
    pat = re.compile(rf"{pattern}\s*:\s*([-+]?\d+\.?\d*)")
    m = pat.search(text)
    return float(m.group(1)) if m else None


def extract_percentile(label, text):
    pat = re.compile(rf"{re.escape(label)}\s*%?\s*KLD\s*:\s*([-+]?\d+\.?\d*)")
    m = pat.search(text)
    return float(m.group(1)) if m else None


def quant_tag_from_path(path: Path) -> str:
    """Extract a short tag from a quantized GGUF filename.
    Qwen3.5-2B-UD-Q8_K_XL.gguf -> q8_k_xl
    """
    name = path.stem.lower()
    m = re.search(r"q(\d)_k_(\w+)", name)
    if m:
        return f"q{m.group(1)}_k_{m.group(2)}"
    m = re.search(r"q(\d)_\d+", name)  # q4_0, q5_1 etc
    if m:
        return f"q{m.group(1)}"
    return name.replace(".", "_").replace("-", "_")[-20:]


def slice_into_chunks(text: str, chunk_chars: int):
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur, cur_size = [], 0
    for p in paragraphs:
        if len(p) > chunk_chars:
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_size = [], 0
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
    return [c for c in chunks if c.strip()]


def prompt_rerun(existing, phase_name, tag):
    """If `existing` is a non-empty list of paths, ask the user whether to
    rerun (delete + redo) or skip (keep existing). Returns True if rerun."""
    existing = sorted(existing)
    if not existing:
        return False
    print(f"\n  Found {len(existing)} existing {phase_name} file(s) for tag '{tag}':")
    for p in existing[:10]:
        print(f"    {p.name}")
    if len(existing) > 10:
        print(f"    ... and {len(existing) - 10} more")
    while True:
        ans = input(f"  Rerun {phase_name}? [y] delete & rerun / [n] skip existing (default n): ").strip().lower()
        if ans in ('y', 'n', ''):
            break
        print("    please answer y or n")
    if ans == 'y':
        for p in existing:
            try:
                p.unlink()
            except OSError as e:
                print(f"    WARN: could not delete {p.name}: {e}")
        return True
    return False


# ============================================================
# Calibration (computes per-head means, writes .bin with identity W)
# ============================================================
def run_llama_cli_dump(chunk_text, chunk_idx, dump_dir, cli, model, chunk_size_tokens):
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
        str(cli), "-m", str(model), "-f", str(prompt_file),
        "-n", "0",
        "-c", str(chunk_size_tokens + 512),
        "-b", "512", "-ub", "512",
        "-ngl", "999", "-t", "4",
        "--single-turn", "--reasoning", "off",
        "--temp", "0", "--top-k", "1", "--no-warmup",
    ]
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=SUBPROCESS_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"    chunk {chunk_idx}: TIMEOUT")
        return None
    if result.returncode != 0:
        print(f"    chunk {chunk_idx}: llama-cli failed")
        return None
    new_dirs = [p for p in dump_dir.iterdir() if p.is_dir() and p.name not in before]
    if not new_dirs:
        return None
    try:
        prompt_file.unlink()
    except OSError:
        pass
    return max(new_dirs, key=lambda p: p.stat().st_mtime)


def load_dump_arrays(subdir):
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


def write_mean_bin(out_path: Path, layer_indices, n_kv_head, head_dim, means):
    """Write .bin with IDENTITY W matrices (mean-only mode ignores W)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    identity = np.tile(np.eye(head_dim, dtype=np.float32), (n_kv_head, 1, 1))
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", VERSION, len(layer_indices), n_kv_head, head_dim))
        for li in layer_indices:
            f.write(struct.pack("<I", li))
            for _ in ("W_k", "W_v"):
                f.write(np.ascontiguousarray(identity, dtype=np.float32).tobytes())
            for key in ("mean_k", "mean_v"):
                f.write(np.ascontiguousarray(means[li][key], dtype=np.float32).tobytes())
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KiB)")


def calibrate_means(bf16_gguf, calib_path, cli, out_dir, chunk_size, max_tokens, tag):
    print(f"\n=== Phase 1: calibrate means from BF16 GGUF ===")
    print(f"  chunk_size={chunk_size}, max_tokens={max_tokens}")

    mean_bin = out_dir / "variants" / f"{tag}_mean.bin"
    if mean_bin.exists():
        print(f"  {mean_bin} already exists, skipping calibration")
        return mean_bin

    text = calib_path.read_text(encoding="utf-8")
    chunk_chars = int(chunk_size * CHARS_PER_TOKEN)
    max_chars = len(text) if max_tokens == "all" else int(int(max_tokens) * CHARS_PER_TOKEN)
    chunks = slice_into_chunks(text[:max_chars], chunk_chars)
    print(f"  {len(chunks)} chunks")

    dump_dir = out_dir / "pool_dumps" / f"{tag}_calib"
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    sum_k, sum_v = {}, {}
    total_tokens = 0
    layer_indices = n_kv_head = head_dim = None
    failures = 0
    t_start = time.time()

    for i, chunk_text in enumerate(chunks):
        sd = run_llama_cli_dump(chunk_text, i, dump_dir, cli, bf16_gguf, chunk_size)
        if sd is None:
            failures += 1
            continue
        try:
            dump, li, nh, hd = load_dump_arrays(sd)
        except Exception as e:
            print(f"    chunk {i}: load failed: {e}")
            failures += 1
            shutil.rmtree(sd, ignore_errors=True)
            continue
        if layer_indices is None:
            layer_indices = li
            n_kv_head, head_dim = nh, hd
            for l in layer_indices:
                sum_k[l] = np.zeros((n_kv_head, head_dim), dtype=np.float64)
                sum_v[l] = np.zeros((n_kv_head, head_dim), dtype=np.float64)
        for l in layer_indices:
            sum_k[l] += dump[l]["K"].sum(axis=0)
            sum_v[l] += dump[l]["V"].sum(axis=0)
        total_tokens += dump[layer_indices[0]]["K"].shape[0]
        shutil.rmtree(sd, ignore_errors=True)

        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed else 0
        eta = (len(chunks) - i - 1) / rate if rate else 0
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{i+1}/{len(chunks)}] tokens={total_tokens} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s fail={failures}")

    print(f"  total: {total_tokens} tokens, {failures} failed chunks")
    if total_tokens == 0:
        sys.exit("ERROR: no tokens accumulated during calibration")

    means = {l: {"mean_k": (sum_k[l] / total_tokens).astype(np.float32),
                 "mean_v": (sum_v[l] / total_tokens).astype(np.float32)}
             for l in layer_indices}
    write_mean_bin(mean_bin, layer_indices, n_kv_head, head_dim, means)
    return mean_bin


# ============================================================
# Baselines and KL tests
# ============================================================
def generate_baseline(perplexity, model, eval_file, out_path, n_chunks):
    if out_path.exists():
        print(f"  {out_path.name} exists, skipping")
        return
    print(f"  generating {out_path.name}")
    env = {**os.environ, "LLAMA_KV_PCA_FILE": "", "LLAMA_KV_MEAN_ONLY": ""}
    cmd = [str(perplexity), "-m", str(model), "-f", str(eval_file),
           "-ngl", "999", "-c", "2048", "-b", "512",
           "--chunks", str(n_chunks), "--kl-divergence-base", str(out_path)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"    FAILED: {result.stderr[-300:]}")
        sys.exit(1)


def run_kl_test(perplexity, model, eval_file, baseline, out_log, n_chunks,
                kv_quant=None, mean_only=False, mean_path=None):
    if out_log.exists() and out_log.stat().st_size > 1000:
        print(f"  {out_log.name} exists, skipping")
        return
    print(f"  running {out_log.name}")
    env = {**os.environ,
           "LLAMA_KV_PCA_FILE": str(mean_path) if (mean_only and mean_path) else "",
           "LLAMA_KV_MEAN_ONLY": "1" if mean_only else ""}
    cmd = [str(perplexity), "-m", str(model), "-f", str(eval_file),
           "-ngl", "999", "-c", "2048", "-b", "512",
           "-fit", "off",
           "--chunks", str(n_chunks),
           "--kl-divergence", "--kl-divergence-base", str(baseline)]
    if kv_quant:
        cmd += ["-ctk", kv_quant, "-ctv", kv_quant]
    with open(out_log, "wb") as f:
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"    WARN: {out_log.name} returned {result.returncode}")
        try:
            text = out_log.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            tail = lines[-40:] if len(lines) > 40 else lines
            print(f"    --- tail of {out_log.name} ---")
            for line in tail:
                print(f"    {line}")
            print(f"    --- end tail ---")
        except Exception as e:
            print(f"    could not read log: {e}")


def parse_kl_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"expA_(\w+)_(f16|q4_0|q8_0)_(none|plain|mean)_vs_(\w+)\.log", path.name)
    if not m:
        return None
    model_tag, kv_quant, strategy, ref = m.groups()
    return {
        "model": model_tag,
        "kv_quant": kv_quant,
        "strategy": strategy,
        "reference": ref,
        "mean_kld": extract_metric("Mean    KLD", text),
        "median_kld": extract_metric("Median  KLD", text),
        "max_kld": extract_metric("Maximum KLD", text),
        "rms_dp": extract_metric("RMS Δp", text),
        "same_top_p": extract_metric("Same top p", text),
        "p50": extract_percentile("Median", text),
        "p95": extract_percentile("95.0%", text),
        "p99": extract_percentile("99.0%", text),
        "p999": extract_percentile("99.9%", text),
    }


# ============================================================
# Plotting
# ============================================================
def model_only_floor(df, model):
    row = df[(df["model"] == model) & (df["kv_quant"] == "f16") &
             (df["strategy"] == "none") & (df["reference"] == "bf16")]
    return row["mean_kld"].iloc[0] if len(row) else None


def get_kld(df, model, kv, strat, ref):
    row = df[(df["model"] == model) & (df["kv_quant"] == kv) &
             (df["strategy"] == strat) & (df["reference"] == ref)]
    return row["mean_kld"].iloc[0] if len(row) else None


def chart1(df, models):
    fig = go.Figure()
    x_labels = [f"{m}/{k}" for m in models for k in ["q4_0", "q8_0"]]
    for strat, color in [("plain", "#1f77b4"), ("mean", "#ff7f0e")]:
        ys = [get_kld(df, m, k, strat, "bf16") for m in models for k in ["q4_0", "q8_0"]]
        fig.add_trace(go.Bar(x=x_labels, y=ys, name=f"{strat} Hadamard",
                             marker_color=color,
                             hovertemplate="%{x}<br>" + strat + "<br>KLD=%{y:.5f}<extra></extra>"))
    for m, color in [(mm, c) for mm, c in zip(models[1:], ["#666", "#333", "#111", "#000"][:len(models)-1])]:
        floor = model_only_floor(df, m)
        if floor is not None:
            fig.add_hline(y=floor, line_dash="dash", line_color=color, line_width=1.5,
                          annotation_text=f"{m}+F16 floor ({floor:.5f})",
                          annotation_position="right", annotation_font_color=color)
    fig.update_layout(
        title=dict(text="<b>Chart 1: Mean-only wins everywhere</b><br>"
                        "<sup>Mean KLD vs universal BF16+F16 baseline. Lower = better. "
                        "Dashed = irreducible weight-quant floor.</sup>", x=0.5),
        xaxis_title="model / KV cache quant", yaxis_title="Mean KLD",
        barmode="group", legend_title="Strategy", height=550, hovermode="x unified",
    )
    return fig


def chart2(df, models):
    fig = go.Figure()
    x_labels, floors, kv_onlys, totals = [], [], [], []
    for m in models:
        floor = model_only_floor(df, m) or 0.0
        for k in ["q4_0", "q8_0"]:
            for s in ["plain", "mean"]:
                x_labels.append(f"{m}/{k}/{s}")
                total = get_kld(df, m, k, s, "bf16")
                kv = get_kld(df, m, k, s, "self")
                if kv is None:
                    kv = total if total is not None else 0.0
                floors.append(floor)
                kv_onlys.append(kv)
                totals.append(total if total is not None else 0.0)
    fig.add_trace(go.Bar(x=x_labels, y=floors, name="Model-only floor",
                         marker_color="#7f7f7f",
                         hovertemplate="%{x}<br>floor=%{y:.5f}<extra></extra>"))
    fig.add_trace(go.Bar(x=x_labels, y=kv_onlys, name="KV-cache + rotation",
                         marker_color="#1f77b4",
                         hovertemplate="%{x}<br>KV-only=%{y:.5f}<extra></extra>"))
    fig.add_trace(go.Bar(x=x_labels, y=totals, name="Total",
                         marker_color="#d62728",
                         hovertemplate="%{x}<br>total=%{y:.5f}<extra></extra>"))
    fig.update_layout(
        title=dict(text="<b>Chart 2: Where does the degradation come from?</b><br>"
                        "<sup>Gray=model weight quant floor, blue=KV-cache+rotation, red=total (vs BF16+F16). "
                        "Total ≈ floor + KV-only (KL not perfectly additive).</sup>", x=0.5),
        xaxis_title="config", yaxis_title="Mean KLD",
        barmode="group", legend_title="Metric", height=600, hovermode="x unified",
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01),
    )
    return fig


def chart3(df, models):
    pct_cols = ["p50", "p95", "p99", "p999", "max_kld"]
    pct_labels = ["p50<br>(median)", "p95", "p99", "p99.9", "max"]
    plot_df = df[(df["reference"] == "bf16") & (df["strategy"].isin(["plain", "mean"]))].copy()
    plot_df = plot_df.sort_values(["model", "kv_quant", "strategy"])
    y_labels = plot_df.apply(lambda r: f"{r['model']}/{r['kv_quant']}/{r['strategy']}", axis=1).tolist()
    z = plot_df[pct_cols].values.tolist()
    fig = go.Figure(data=go.Heatmap(
        z=z, x=pct_labels, y=y_labels, colorscale="Viridis",
        text=[[f"{v:.5f}" if pd.notna(v) else "-" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="config=%{y}<br>percentile=%{x}<br>KLD=%{z:.5f}<extra></extra>",
        colorbar=dict(title="KLD", tickformat=".4f"),
    ))
    fig.update_layout(
        title=dict(text="<b>Chart 3: KLD distribution shape per config</b><br>"
                        "<sup>Compare rows to see how mean-only compresses the tail.</sup>", x=0.5),
        xaxis_title="KLD percentile", yaxis_title="config",
        height=550, margin=dict(l=200, r=150), yaxis=dict(autorange="reversed"),
    )
    return fig


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bf16_gguf", type=Path, required=True)
    p.add_argument("--quant_ggufs", type=Path, nargs="+", required=True,
                   help="One or more quantized GGUF paths")
    p.add_argument("--calib", type=Path, required=True, help="Calibration text corpus")
    p.add_argument("--llama_dir", type=Path, default=Path("F:/Repos/llama.cpp"))
    p.add_argument("--build_dir", type=Path, default=Path("F:/Repos/llama.cpp/build-cuda"))
    p.add_argument("--eval_file", type=Path,
                   default=Path("F:/Repos/llama.cpp/scripts/wikitext-2-raw/wiki.test.raw"))
    p.add_argument("--out_dir", type=Path, default=Path("F:/Repos/llama.cpp/kl_dumps"))
    p.add_argument("--tag", default="qwen35_2b", help="Tag for naming files")
    p.add_argument("--chunk_size", type=int, default=2048)
    p.add_argument("--max_tokens", default="200000", help="'all' or integer")
    p.add_argument("--n_chunks_eval", type=int, default=10)
    args = p.parse_args()

    # Resolve binaries
    cli = args.build_dir / "bin" / "Debug" / "llama-cli.exe"
    perplexity = args.build_dir / "bin" / "Debug" / "llama-perplexity.exe"
    for name, path in [("bf16_gguf", args.bf16_gguf), ("calib", args.calib),
                       ("eval_file", args.eval_file), ("cli", cli), ("perplexity", perplexity)]:
        if not path.exists():
            sys.exit(f"ERROR: {name} not found: {path}")
    for i, qg in enumerate(args.quant_ggufs):
        if not qg.exists():
            sys.exit(f"ERROR: quant_ggufs[{i}] not found: {qg}")

    # Setup model tags
    quant_models = [(quant_tag_from_path(qg), qg) for qg in args.quant_ggufs]
    all_models = [("bf16", args.bf16_gguf)] + quant_models
    model_tags = [t for t, _ in all_models]
    print(f"Models to test:")
    for tag, path in all_models:
        print(f"  {tag:10s} -> {path}")

    # Setup output dirs
    for sub in ["pool_dumps", "logs", "variants"]:
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    # ===== Phase 1: calibrate means =====
    mean_bin_path = args.out_dir / "variants" / f"{args.tag}_mean.bin"
    prompt_rerun([mean_bin_path] if mean_bin_path.exists() else [],
                 "calibration", args.tag)
    mean_bin = calibrate_means(args.bf16_gguf, args.calib, cli, args.out_dir,
                               args.chunk_size, args.max_tokens, args.tag)

    # ===== Phase 2: generate baselines =====
    print(f"\n=== Phase 2: generate F16-cache baselines ===")
    baselines_dir = args.out_dir / "baselines" / args.tag
    baselines_dir.mkdir(parents=True, exist_ok=True)
    prompt_rerun(list(baselines_dir.glob("*.kld")), "baselines", args.tag)

    base_bf16 = baselines_dir / "baseline_f16.kld"
    generate_baseline(perplexity, args.bf16_gguf, args.eval_file, base_bf16, args.n_chunks_eval)

    baselines = {"bf16": base_bf16}
    for model_tag, path in quant_models:
        base = baselines_dir / f"baseline_f16_{model_tag}.kld"
        generate_baseline(perplexity, path, args.eval_file, base, args.n_chunks_eval)
        baselines[model_tag] = base

    # ===== Phase 3: KL tests =====
    print(f"\n=== Phase 3: KL tests ({len(model_tags)} models x 2 KV quants x 2 strategies) ===")
    logs_dir = args.out_dir / "logs" / args.tag
    logs_dir.mkdir(parents=True, exist_ok=True)
    prompt_rerun(list(logs_dir.glob("expA_*.log")), "KL tests", args.tag)

    for model_tag, model_path in all_models:
        is_bf16 = (model_tag == "bf16")
        # Model-only (F16 cache, vs BF16)
        if not is_bf16:
            log = logs_dir / f"expA_{model_tag}_f16_none_vs_bf16.log"
            run_kl_test(perplexity, model_path, args.eval_file, base_bf16, log,
                        args.n_chunks_eval, kv_quant=None, mean_only=False, mean_path=None)

        for kv in ["q4_0", "q8_0"]:
            for strategy, mean_on in [("plain", False), ("mean", True)]:
                # vs BF16
                log1 = logs_dir / f"expA_{model_tag}_{kv}_{strategy}_vs_bf16.log"
                run_kl_test(perplexity, model_path, args.eval_file, base_bf16, log1,
                            args.n_chunks_eval, kv_quant=kv,
                            mean_only=mean_on, mean_path=mean_bin)
                # vs self (only for quantized)
                if not is_bf16:
                    log2 = logs_dir / f"expA_{model_tag}_{kv}_{strategy}_vs_self.log"
                    run_kl_test(perplexity, model_path, args.eval_file, baselines[model_tag], log2,
                                args.n_chunks_eval, kv_quant=kv,
                                mean_only=mean_on, mean_path=mean_bin)

    # ===== Phase 4: parse logs + plot =====
    print(f"\n=== Phase 4: parse logs and plot ===")
    records = []
    for p in sorted(logs_dir.glob("expA_*.log")):
        r = parse_kl_log(p)
        if r:
            records.append(r)
    df = pd.DataFrame(records)
    if df.empty:
        sys.exit("ERROR: no KL logs parsed")

    cols = ["model", "kv_quant", "strategy", "mean_kld", "median_kld",
            "max_kld", "p95", "p99", "rms_dp", "same_top_p"]
    print("\n" + "=" * 110)
    print("TABLE 1: KL vs universal BF16+F16 baseline")
    print("=" * 110)
    print(df[df["reference"] == "bf16"][cols].sort_values(["model", "kv_quant", "strategy"])
          .to_string(index=False, float_format=lambda x: f"{x:.5f}" if pd.notna(x) else "-"))

    print("\n" + "=" * 110)
    print("TABLE 2: KL vs same-model F16 baseline")
    print("=" * 110)
    print(df[df["reference"] == "self"][cols].sort_values(["model", "kv_quant", "strategy"])
          .to_string(index=False, float_format=lambda x: f"{x:.5f}" if pd.notna(x) else "-"))

    fig1 = chart1(df, model_tags)
    fig2 = chart2(df, model_tags)
    fig3 = chart3(df, model_tags)

    for i, fig in enumerate([fig1, fig2, fig3], 1):
        out = args.out_dir.parent / "pca_poc" / f"{args.tag}_chart{i}.html"
        if not args.llama_dir.exists():
            out = args.out_dir / f"{args.tag}_chart{i}.html"
        else:
            out = args.llama_dir / "pca_poc" / f"{args.tag}_chart{i}.html"
        fig.write_html(str(out))
        print(f"wrote {out}")

    print("\n=== DONE ===")
    print(f"All outputs in: {args.out_dir}")
    print(f"Charts: {args.tag}_chart{{1,2,3}}.html")


if __name__ == "__main__":
    main()
