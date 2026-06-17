"""
Interactive KV cache visualizer for the PCA POC.

Loads all runs under F:\\Repos\\llama.cpp\\kv_dumps\\, back-rotates each K/V
matrix to a common "original" space (Hadamard inverse = Hadamard itself,
PCA inverse = W^T applied to the stored K_c), then shows side-by-side
heatmaps and per-layer RMS quantization errors.

Requires: numpy, matplotlib, torch  (kvpress_env has all three).
Run with kvpress_env python:
    C:/Users/MukkeshGanesh/miniconda3/envs/kvpress_env/python.exe F:/Repos/llama.cpp/pca_poc/visualize_kv.py

Controls (in the matplotlib window):
    Left / Right   cycle layer
    K / V          toggle K vs V cache
    R              toggle raw stored values vs back-rotated to original space
    S              save current figure to kv_dumps/<timestamp>.png
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

DUMPS_DIR  = Path(r"F:\Repos\llama.cpp\kv_dumps")
PCA_PT     = Path(r"C:\Users\MukkeshGanesh\Downloads\kv_pca_standalone\pca_scales.pt")


# ---------- loading ----------

def load_run(subdir: Path):
    meta = json.loads((subdir / "meta.json").read_text())
    kv = meta["kv_cache"]
    layers = {}
    for il in kv["kv_layer_indices"]:
        K = np.load(subdir / f"K_layer_{il}.npy").astype(np.float32)
        V = np.load(subdir / f"V_layer_{il}.npy").astype(np.float32)
        layers[il] = {"K": K, "V": V}
    return meta, layers, kv


def load_pca_scales():
    raw = torch.load(str(PCA_PT), map_location="cpu", weights_only=False)
    out = {}
    for k, v in raw.items():
        out[int(k)] = {
            "W_k":    v["W_k"].   float().numpy(),    # (n_kv_head, D, D)
            "W_v":    v["W_v"].   float().numpy(),
            "mean_k": v["mean_k"].float().numpy(),    # (n_kv_head, D)
            "mean_v": v["mean_v"].float().numpy(),
        }
    return out


def hadamard_matrix(n: int) -> np.ndarray:
    """Walsh-Hadamard orthonormal matrix; matches src/llama-kv-cache.cpp:ggml_gen_hadamard."""
    H = np.zeros((n, n), dtype=np.float32)
    H[0, 0] = 1.0 / np.sqrt(n)
    s = 1
    while s < n:
        for i in range(s):
            for j in range(s):
                v = H[i, j]
                H[i + s, j    ] =  v
                H[i    , j + s] =  v
                H[i + s, j + s] = -v
        s *= 2
    return H


# ---------- back-rotation to original space ----------

def back_rotate_pca(M: np.ndarray, W: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """M:(n_tokens, n_kv_head*D) in PCA space -> original space.

    Forward: K_c[h] = W[h] @ (K[h] - mean[h])
    Inverse: K[h]   = W[h].T @ K_c[h] + mean[h]   (W orthogonal)
    """
    n_tokens = M.shape[0]
    n_kv_head, D, _ = W.shape
    Mc = M.reshape(n_tokens, n_kv_head, D)
    # Kc[n,h,i] -> K[n,h,j] = sum_i W[h,i,j] * Kc[n,h,i]
    Mo = np.einsum("hij,nhi->nhj", W, Mc, optimize=True) + mean[None, :, :]
    return Mo.reshape(n_tokens, -1)


def back_rotate_hadamard(M: np.ndarray, n_kv_head: int, head_dim: int) -> np.ndarray:
    """Walsh-Hadamard is symmetric+orthonormal, so the inverse is the matrix itself."""
    H = hadamard_matrix(head_dim)
    n_tokens = M.shape[0]
    Mr = M.reshape(n_tokens, n_kv_head, head_dim)
    Mo = np.einsum("ij,nhj->nhi", H, Mr, optimize=True)
    return Mo.reshape(n_tokens, -1)


def to_original(M: np.ndarray, il: int, mode: str, kv_meta: dict, pca_scales: dict) -> np.ndarray:
    """Back-rotate M to original space according to this run's config."""
    cfg_type   = kv_meta["type_k"] if mode == "K" else kv_meta["type_v"]
    cfg_pca    = kv_meta["pca_enabled"]
    n_kv_head  = kv_meta["n_kv_head"]
    head_dim   = kv_meta["head_dim_k"] if mode == "K" else kv_meta["head_dim_v"]

    if cfg_pca and il in pca_scales:
        s = pca_scales[il]
        W    = s["W_k"]    if mode == "K" else s["W_v"]
        mean = s["mean_k"] if mode == "K" else s["mean_v"]
        return back_rotate_pca(M, W, mean)
    if cfg_type != "f16":
        return back_rotate_hadamard(M, n_kv_head, head_dim)
    return M  # F16 baseline already in original space


# ---------- metrics ----------

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def per_layer_rms(runs: dict, layer_ids, mode: str, pca_scales: dict):
    """For each layer, return {run_name: rms_error_vs_baseline}."""
    # find F16 baseline (no PCA)
    baseline_name = None
    for name, run in runs.items():
        kv = run[2]
        if kv["type_k"] == "f16" and not kv["pca_enabled"]:
            baseline_name = name
            break

    out = {il: {} for il in layer_ids}
    if baseline_name is None:
        return out, None
    for il in layer_ids:
        base_M = to_original(runs[baseline_name][1][il][mode], il, mode, runs[baseline_name][2], pca_scales)
        for name, run in runs.items():
            if name == baseline_name:
                continue
            M = to_original(run[1][il][mode], il, mode, run[2], pca_scales)
            # align length (different runs may have different n_tokens; truncate to min)
            n = min(M.shape[0], base_M.shape[0])
            out[il][name] = rms(M[:n] - base_M[:n])
    return out, baseline_name


# ---------- interactive viewer ----------

def explore(runs: dict, pca_scales: dict):
    layer_ids = sorted(next(iter(runs.values()))[1].keys())
    run_names = list(runs.keys())
    state = {"il_idx": 0, "mode": "K", "backrot": True}

    fig, axes = plt.subplots(2, max(len(run_names), 2), figsize=(6 * max(len(run_names), 2), 8))
    if len(axes.shape) == 1:
        axes = axes.reshape(2, -1)
    fig.suptitle("KV cache visualizer  (Left/Right: layer   K/V: toggle   R: raw/orig   S: save)")

    def render():
        il = layer_ids[state["il_idx"]]
        mode = state["mode"]
        space = "original" if state["backrot"] else "raw stored"

        # locate F16 baseline (in original space, it equals raw)
        baseline_M = None
        baseline_name = None
        for name, run in runs.items():
            kv = run[2]
            if kv["type_k"] == "f16" and not kv["pca_enabled"]:
                baseline_M = to_original(run[1][il][mode], il, mode, kv, pca_scales)
                baseline_name = name
                break

        for ax in axes.flat:
            ax.clear(); ax.set_xticks([]); ax.set_yticks([])

        for col, name in enumerate(run_names):
            run = runs[name]
            kv = run[2]
            M_stored = run[1][il][mode]
            M = to_original(M_stored, il, mode, kv, pca_scales) if state["backrot"] else M_stored

            # top row: heatmap
            vabs = max(abs(np.percentile(M, 1)), abs(np.percentile(M, 99)), 1e-3)
            im = axes[0, col].imshow(M.T, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
            axes[0, col].set_title(f"{name}\n{mode} L{il}  ({kv['type_k']}/{kv['type_v']}, pca={kv['pca_enabled']})\n"
                                   f"shape={M.shape}, space={space}", fontsize=9)
            if col == 0:
                axes[0, col].set_ylabel("head_dim")
            plt.colorbar(im, ax=axes[0, col], fraction=0.025)

            # bottom row: diff vs baseline
            if baseline_M is not None and name != baseline_name:
                n = min(M.shape[0], baseline_M.shape[0])
                diff = M[:n] - baseline_M[:n]
                dabs = max(abs(np.percentile(diff, 99)), 1e-6)
                im2 = axes[1, col].imshow(diff.T, aspect="auto", cmap="RdBu_r", vmin=-dabs, vmax=dabs)
                axes[1, col].set_title(f"diff vs F16 baseline ({baseline_name})\n"
                                       f"RMS={rms(diff):.4f}  max|d|={np.abs(diff).max():.4f}", fontsize=9)
                plt.colorbar(im2, ax=axes[1, col], fraction=0.025)
            else:
                axes[1, col].text(0.5, 0.5, "(this IS the baseline)" if name == baseline_name else "(no F16 baseline)",
                                  ha="center", va="center", transform=axes[1, col].transAxes)
                axes[1, col].set_title("baseline / no diff", fontsize=9)

        fig.canvas.draw_idle()

    def on_key(event):
        k = event.key
        if k == "right":
            state["il_idx"] = (state["il_idx"] + 1) % len(layer_ids); render()
        elif k == "left":
            state["il_idx"] = (state["il_idx"] - 1) % len(layer_ids); render()
        elif k in ("k", "K"):
            state["mode"] = "K"; render()
        elif k in ("v", "V"):
            state["mode"] = "V"; render()
        elif k in ("r", "R"):
            state["backrot"] = not state["backrot"]; render()
        elif k in ("s", "S"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = DUMPS_DIR / f"viz_L{layer_ids[state['il_idx']]}_{state['mode']}_{ts}.png"
            fig.savefig(out, dpi=110, bbox_inches="tight")
            print(f"saved: {out}")

    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.show()


# ---------- main ----------

def main():
    if not DUMPS_DIR.exists():
        print(f"DUMPS_DIR not found: {DUMPS_DIR}")
        sys.exit(1)

    runs = {}
    for d in sorted(DUMPS_DIR.iterdir()):
        if not d.is_dir():
            continue
        try:
            meta, layers, kv = load_run(d)
        except Exception as e:
            print(f"  skip {d.name}: {e}")
            continue
        runs[d.name] = (meta, layers, kv)
        print(f"loaded: {d.name}  pca={kv['pca_enabled']}  ctk={kv['type_k']}  ctv={kv['type_v']}  "
              f"n_tokens={kv['n_tokens_used']}  layers={kv['kv_layer_indices']}")

    if not runs:
        print("no runs found")
        sys.exit(1)

    print(f"\nloading PCA scales from {PCA_PT}")
    try:
        pca_scales = load_pca_scales()
        print(f"  layers: {sorted(pca_scales.keys())}")
    except Exception as e:
        print(f"  failed to load PCA scales ({e}); PCA back-rotation will be skipped")
        pca_scales = {}

    # per-layer RMS summary before opening the window
    layer_ids = sorted(next(iter(runs.values()))[1].keys())
    for mode in ("K", "V"):
        rms_table, baseline_name = per_layer_rms(runs, layer_ids, mode, pca_scales)
        if baseline_name is None:
            print(f"\n[{mode}] no F16 baseline found, skipping RMS table")
            continue
        print(f"\n[{mode}] RMS error vs F16 baseline ({baseline_name}):")
        print(f"  {'layer':>6}  " + "  ".join(f"{n:>40}" for n in runs if n != baseline_name))
        for il in layer_ids:
            row = "  ".join(f"{rms_table[il].get(n, float('nan')):>40.6f}" for n in runs if n != baseline_name)
            print(f"  {il:>6}  {row}")

    print("\nopening interactive viewer...")
    explore(runs, pca_scales)


if __name__ == "__main__":
    main()
