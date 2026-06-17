"""
Random rotation ablation: why does PCA beat Hadamard at Q4_0?

Tests six conditions, each a different `rotate -> Q4_0 quantize -> back-rotate`
pipeline applied to the dumped F16 baseline K/V:

  1. None (raw)         - quantize F16 K directly. No rotation, no mean subtract.
  2. Hadamard           - K @ H, quantize, @ H.   (data-independent rotation)
  3. Hadamard + mean    - (K - mean) @ H, quantize, @ H + mean.
  4. Random             - K @ W_rand, quantize, @ W_rand.T. (avg over N trials)
  5. Random + mean      - (K - mean) @ W_rand, quantize, @ W_rand.T + mean.
  6. PCA                - (K - mean) @ W_pca.T, quantize, @ W_pca + mean.

Conditions 2 and 6 must match the real dumped Q4 Had / Q4 PCA runs to 4 decimals
(self-check). The other four are simulations of what llama-cli *would* produce
if it supported those rotation modes.

The decomposition tells you what drives PCA's win:
  - Mean-subtraction effect:     compare (2) vs (3), or (4) vs (5)
  - Rotation-choice effect:      compare (3) vs (6)  (same mean, different rotation)
  - Calibration effect:          compare (5) vs (6)  (same mean+orthonormal, PCA is calibrated)

Usage:
    C:/Users/MukkeshGanesh/miniconda3/envs/kvpress_env/python.exe F:/Repos/llama.cpp/pca_poc/ablation_rotations.py [dump_dir] [layer] [mode] [n_random_trials]

    dump_dir         default F:\\Repos\\llama.cpp\\kv_dumps_math
    layer            default 3
    mode             default K  (K or V)
    n_random_trials  default 10
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch


PCA_PT = Path(r"C:\Users\MukkeshGanesh\Downloads\kv_pca_standalone\pca_scales.pt")
HEAD_DIM = 256
N_KV_HEAD = 2
QK = 32
SEED = 0  # for reproducible random rotations


# ---- Q4_0 quantizer (matches ggml's quantize_row_q4_0) ----

def quantize_dequantize_q4_0(M: np.ndarray) -> np.ndarray:
    """Per-row block Q4_0 quant+dequant. M shape (n_rows, multiple_of_32)."""
    assert M.shape[1] % QK == 0
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


_H256 = hadamard_matrix(256)
_H64  = hadamard_matrix(64)


def apply_hadamard(M: np.ndarray, mode: str) -> np.ndarray:
    """Apply the correct Hadamard rotation per cache type.

    K uses H_256 (dense, full head_dim) — matches src/llama-kv-cache.cpp:1761 build_input_k_rot
    V uses H_64 applied to 4 sub-blocks of 64 dims each (block-diagonal, NOT dense)
        — matches src/llama-kv-cache.cpp:1786 build_input_v_rot which fixes nrot=64.

    Symmetric + orthonormal in both cases, so forward = inverse.

    M shape: (n_rows, head_dim=256).
    """
    if mode == "K":
        return M @ _H256
    n = M.shape[0]
    return (M.reshape(n, 4, 64) @ _H64).reshape(n, HEAD_DIM)


def random_orthonormal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthonormal matrix via QR of a random Gaussian."""
    A = rng.standard_normal((n, n)).astype(np.float32)
    Q, R = np.linalg.qr(A)
    # normalize sign so the diagonal of R is positive (makes QR unique)
    sign = np.sign(np.diag(R))
    Q = Q * sign[None, :]
    return Q.astype(np.float32)


def load_pca_scales(layer: int):
    raw = torch.load(str(PCA_PT), map_location="cpu", weights_only=False)
    s = raw[layer]
    return {
        "W_k":    s["W_k"].   float().numpy(),
        "W_v":    s["W_v"].   float().numpy(),
        "mean_k": s["mean_k"].float().numpy(),
        "mean_v": s["mean_v"].float().numpy(),
    }


# ---- pipeline ----

def pipeline(K, W, mean, mode, use_hadamard, quantize=True):
    """Apply forward rotation with optional mean subtraction, quantize, back-rotate.

    If use_hadamard: apply mode-specific Hadamard (H_256 for K, blocked H_64 for V)
                    via apply_hadamard(). W is ignored.
    Else (PCA / random dense): use full matrix W. Forward = (K - mean) @ W.T,
                    inverse = K_q @ W + mean.

    Pass mean=None to skip mean subtraction.
    """
    Kc = K - mean if mean is not None else K
    if use_hadamard:
        K_rot = apply_hadamard(Kc, mode)
    else:
        K_rot = Kc @ W.T
    if quantize:
        K_rot = quantize_dequantize_q4_0(K_rot)
    if use_hadamard:
        return apply_hadamard(K_rot, mode) + (mean if mean is not None else 0.0)
    return K_rot @ W + (mean if mean is not None else 0.0)


def rms(M):
    return float(np.sqrt(np.mean(M.astype(np.float64) ** 2)))


def rms_error(recon, K):
    return rms(recon - K)


# ---- dump loaders (for self-check) ----

def find_runs(dump_dir: Path):
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


def back_rotate_real(stored, W, mean, mode, used_pca, used_had):
    """Back-rotate a dumped stored-tensor to original space.
    For Hadamard, uses apply_hadamard (mode-aware: H_256 for K, blocked H_64 for V).
    For PCA, uses full W + mean."""
    n = stored.shape[0]
    Sr = stored.reshape(n, N_KV_HEAD, HEAD_DIM)
    out = np.zeros_like(Sr)
    if used_pca:
        for h in range(N_KV_HEAD):
            out[:, h, :] = Sr[:, h, :] @ W[h] + mean[h]
    elif used_had:
        for h in range(N_KV_HEAD):
            out[:, h, :] = apply_hadamard(Sr[:, h, :], mode)
    else:
        out = Sr
    return out.reshape(n, -1)


# ---- main ----

def run(dump_dir: Path, layer: int, mode: str, n_random: int):
    runs = find_runs(dump_dir)
    if (False, "f16") not in runs:
        print(f"ERROR: no F16 baseline in {dump_dir}")
        sys.exit(1)
    if (False, "q4_0") not in runs or (True, "q4_0") not in runs:
        print(f"ERROR: need both Q4_0 Had and Q4_0 PCA runs in {dump_dir}")
        print(f"       found: {list(runs.keys())}")
        sys.exit(1)

    base_dir   = runs[(False, "f16")]
    q4had_dir  = runs[(False, "q4_0")]
    q4pca_dir  = runs[(True, "q4_0")]

    f16   = np.load(base_dir  / f"{mode}_layer_{layer}.npy").astype(np.float32)
    q4had = np.load(q4had_dir / f"{mode}_layer_{layer}.npy").astype(np.float32)
    q4pca = np.load(q4pca_dir / f"{mode}_layer_{layer}.npy").astype(np.float32)

    H256 = hadamard_matrix(HEAD_DIM)
    scales = load_pca_scales(layer)
    W_pca = scales["W_k"] if mode == "K" else scales["W_v"]
    mean_pca = scales["mean_k"] if mode == "K" else scales["mean_v"]

    # find divergence to truncate F16 K
    q4had_back = back_rotate_real(q4had, W_pca, mean_pca, mode, used_pca=False, used_had=True)
    div_idx = find_divergence(f16, q4had_back)
    N = (div_idx // 32) * 32
    if N < 32: N = 32

    print(f"dump_dir:    {dump_dir}")
    print(f"layer:       {layer}")
    print(f"mode:        {mode}")
    print(f"N tokens:    {N}  (truncated from {f16.shape[0]} via divergence detection)")
    print(f"PCA scales:  layer {layer} present (W shape {W_pca.shape}, mean shape {mean_pca.shape})")
    print(f"random seed: {SEED}, {n_random} trials per random condition")
    print()

    rng = np.random.default_rng(SEED)

    # Per-head sim for each condition
    results = {}
    for h in range(N_KV_HEAD):
        K = f16[:N].reshape(N, N_KV_HEAD, HEAD_DIM)[:, h, :]
        W_pca_h = W_pca[h]
        mean_h  = mean_pca[h]

        # 1. None (no rotation, no mean) -- just quantize K directly
        K_none = quantize_dequantize_q4_0(K)
        results.setdefault("1. None (raw)", []).append(rms_error(K_none, K))

        # 2. Hadamard (no mean) -- K uses H_256, V uses blocked H_64 (matches llama.cpp)
        K_had = pipeline(K, None, mean=None, mode=mode, use_hadamard=True)
        results.setdefault("2. Hadamard", []).append(rms_error(K_had, K))

        # 3. Hadamard + mean
        K_hadm = pipeline(K, None, mean=mean_h, mode=mode, use_hadamard=True)
        results.setdefault("3. Hadamard + mean", []).append(rms_error(K_hadm, K))

        # 4. Random (no mean) -- dense 256x256 orthonormal, averaged over trials
        rand_errs = []
        for _ in range(n_random):
            W_rand = random_orthonormal(HEAD_DIM, rng)
            K_rand = pipeline(K, W_rand, mean=None, mode=mode, use_hadamard=False)
            rand_errs.append(rms_error(K_rand, K))
        results.setdefault("4. Random (no mean)", []).append(float(np.mean(rand_errs)))

        # 5. Random + mean -- dense 256x256 orthonormal, averaged over trials
        rand_m_errs = []
        for _ in range(n_random):
            W_rand = random_orthonormal(HEAD_DIM, rng)
            K_randm = pipeline(K, W_rand, mean=mean_h, mode=mode, use_hadamard=False)
            rand_m_errs.append(rms_error(K_randm, K))
        results.setdefault("5. Random + mean", []).append(float(np.mean(rand_m_errs)))

        # 6. PCA (calibrated + mean) -- always full W matrix per head
        K_pca = pipeline(K, W_pca_h, mean=mean_h, mode=mode, use_hadamard=False)
        results.setdefault("6. PCA", []).append(rms_error(K_pca, K))

    # average across heads
    avg = {k: float(np.mean(v)) for k, v in results.items()}

    # self-check: conditions 2 and 6 vs real dumps
    real_had = rms_error(q4had_back[:N], f16[:N])
    q4pca_back = back_rotate_real(q4pca, W_pca, mean_pca, mode, used_pca=True, used_had=False)[:N]
    real_pca = rms_error(q4pca_back, f16[:N])

    # print results
    print(f"{'condition':<26} {'head 0':>10} {'head 1':>10} {'avg':>10}   {'vs None':>10}")
    print("-" * 76)
    base_avg = avg["1. None (raw)"]
    for cond in ["1. None (raw)", "2. Hadamard", "3. Hadamard + mean",
                 "4. Random (no mean)", "5. Random + mean", "6. PCA"]:
        h0, h1 = results[cond][0], results[cond][1]
        a = avg[cond]
        delta = (a / base_avg - 1.0) * 100
        print(f"{cond:<26} {h0:>10.4f} {h1:>10.4f} {a:>10.4f}   {delta:>+9.1f}%")
    print()

    # self-check
    sim_had = avg["2. Hadamard"]
    sim_pca = avg["6. PCA"]
    print("Self-check vs real llama-cli dumps:")
    print(f"  Hadamard:  sim={sim_had:.4f}  real={real_had:.4f}  match={'YES' if abs(sim_had-real_had)<0.005 else 'NO -- investigate'}")
    print(f"  PCA:       sim={sim_pca:.4f}  real={real_pca:.4f}  match={'YES' if abs(sim_pca-real_pca)<0.005 else 'NO -- investigate'}")
    print()

    # decomposition
    had = avg["2. Hadamard"]; hadm = avg["3. Hadamard + mean"]
    rnd = avg["4. Random (no mean)"]; rndm = avg["5. Random + mean"]
    pca = avg["6. PCA"]
    print("Decomposition (what drives PCA's win):")
    print(f"  Mean subtraction effect on Hadamard:  {had:.4f} -> {hadm:.4f}  ({(hadm/had - 1)*100:+.1f}%)")
    print(f"  Mean subtraction effect on Random:    {rnd:.4f} -> {rndm:.4f}  ({(rndm/rnd - 1)*100:+.1f}%)")
    print(f"  Rotation choice (PCA vs Had+mean):    {hadm:.4f} -> {pca:.4f}  ({(pca/hadm - 1)*100:+.1f}%)")
    print(f"  Calibration (PCA vs Rand+mean):       {rndm:.4f} -> {pca:.4f}  ({(pca/rndm - 1)*100:+.1f}%)")
    print()

    # interpretation
    print("Interpretation:")
    if abs(pca - rndm) / pca < 0.05:
        print("  -> PCA's win is essentially fully explained by 'any orthonormal rotation + mean")
        print("     subtraction'. The calibration to data covariance adds little on top.")
    elif abs(pca - hadm) / pca < 0.05:
        print("  -> PCA's win is essentially fully explained by 'mean subtraction'. The choice")
        print("     of rotation (Hadamard vs PCA) adds little if both subtract the mean.")
    elif pca < rndm and pca < hadm:
        print("  -> PCA's win requires BOTH calibration AND mean subtraction. Neither alone")
        print("     closes the gap.")
    else:
        print("  -> Result is ambiguous; check the decomposition numbers above.")


def main():
    dump_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"F:\Repos\llama.cpp\kv_dumps_math")
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    mode = sys.argv[3] if len(sys.argv) > 3 else "K"
    n_random = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    if mode not in ("K", "V"):
        print(f"mode must be K or V, got {mode}")
        sys.exit(1)
    run(dump_dir, layer, mode, n_random)


if __name__ == "__main__":
    main()
