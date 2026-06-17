"""
KV cache dashboard for the PCA POC.

Web-based (Dash + Plotly). Run, then open http://127.0.0.1:8050 in a browser.

    C:/Users/MukkeshGanesh/miniconda3/envs/kvpress_env/python.exe F:/Repos/llama.cpp/pca_poc/kv_dashboard.py

Loads every run under F:\\Repos\\llama.cpp\\kv_dumps\\, picks the F16 no-PCA run
as the baseline, back-rotates Hadamard / PCA caches into the original K/V space
(Walsh-Hadamard inverse = itself; PCA inverse = W^T + mean), and shows:

  * Sidebar — layer / K-or-V / raw-vs-original controls, plus a per-method
    metrics table (RMS, max|err|, mean|err|, cosine sim, SNR, rel Frobenius)
    for the currently selected layer.
  * Tab "Current layer" — three side-by-side K/V heatmaps + two diff-vs-baseline
    heatmaps + a per-head RMS bar chart.
  * Tab "Across layers" — RMS error (and other metrics) for every layer in one
    grouped bar chart; click a bar to jump to that layer.
  * Tab "Distributions" — per-method value histograms to see how the tails
    change under quantization.

Back-rotation matches what src/llama-kv-cache.cpp does at attention time, so
the diffs you see are the actual quantization error each method introduces.
"""

from pathlib import Path
import sys
import json

import numpy as np
import torch
import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DUMPS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"F:\Repos\llama.cpp\kv_dumps")
PCA_PT    = Path(r"C:\Users\MukkeshGanesh\Downloads\kv_pca_standalone\pca_scales.pt")


# ---------- loading + back-rotation ----------

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
            "W_k":    v["W_k"].   float().numpy(),
            "W_v":    v["W_v"].   float().numpy(),
            "mean_k": v["mean_k"].float().numpy(),
            "mean_v": v["mean_v"].float().numpy(),
        }
    return out


def hadamard_matrix(n: int) -> np.ndarray:
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


def back_rotate_pca(M, W, mean):
    """M:(n_tokens, n_kv_head*D) in PCA space -> original space."""
    n_tokens = M.shape[0]
    n_kv_head, D, _ = W.shape
    Mc = M.reshape(n_tokens, n_kv_head, D)
    Mo = np.einsum("hij,nhi->nhj", W, Mc, optimize=True) + mean[None, :, :]
    return Mo.reshape(n_tokens, -1)


def back_rotate_hadamard(M, n_kv_head, head_dim):
    H = hadamard_matrix(head_dim)
    n_tokens = M.shape[0]
    Mr = M.reshape(n_tokens, n_kv_head, head_dim)
    Mo = np.einsum("ij,nhj->nhi", H, Mr, optimize=True)
    return Mo.reshape(n_tokens, -1)


def to_original(M, il, mode, kv_meta, pca_scales):
    cfg_type = kv_meta["type_k"] if mode == "K" else kv_meta["type_v"]
    cfg_pca  = kv_meta["pca_enabled"]
    n_kv_head = kv_meta["n_kv_head"]
    head_dim  = kv_meta["head_dim_k"] if mode == "K" else kv_meta["head_dim_v"]
    if cfg_pca and il in pca_scales:
        s = pca_scales[il]
        W    = s["W_k"]    if mode == "K" else s["W_v"]
        mean = s["mean_k"] if mode == "K" else s["mean_v"]
        return back_rotate_pca(M, W, mean)
    if cfg_type != "f16":
        return back_rotate_hadamard(M, n_kv_head, head_dim)
    return M


# ---------- metrics ----------

def find_divergence(M: np.ndarray, baseline: np.ndarray, noise_floor_mult: float = 3.0) -> int:
    """First token index where per-token L2 diff exceeds noise_floor_mult * median
    of the first 64 tokens. Returns len if no divergence detected. Used to truncate
    metric computation to the pre-divergence prefix — past the divergence point the
    two runs have different tokens, so the K matrices are not directly comparable."""
    n = min(M.shape[0], baseline.shape[0])
    if n < 32:
        return n
    per_tok = np.linalg.norm(M[:n].astype(np.float64) - baseline[:n].astype(np.float64), axis=1)
    ref_n = min(n, 64)
    median = float(np.median(per_tok[:ref_n]))
    if median < 1e-9:
        # near-zero noise (e.g. F16 PCA vs F16 baseline): flag any large jump
        median = max(float(per_tok[:ref_n].max()), 1e-6)
    threshold = median * noise_floor_mult
    for i in range(n):
        if per_tok[i] > threshold:
            return i
    return n


def compute_metrics(M, baseline):
    """Error metrics of M vs baseline. Both shape (n_tokens, n_dims).
    Auto-truncates to pre-divergence prefix — past that point runs have
    different tokens and the diff stops being quantization error."""
    n_init = min(M.shape[0], baseline.shape[0])
    M = M[:n_init].astype(np.float64)
    baseline = baseline[:n_init].astype(np.float64)
    div_idx = find_divergence(M, baseline)
    M = M[:div_idx]
    baseline = baseline[:div_idx]
    if div_idx == 0:
        return {
            "RMS error": 0.0, "Max |error|": 0.0, "Mean |error|": 0.0,
            "RMS baseline": 0.0, "Relative RMS": 0.0, "Cosine sim": 1.0,
            "SNR (dB)": 0.0, "Rel Frobenius": 0.0,
            "N compared": 0, "diverged at": None,
        }
    diff = M - baseline
    rms_err   = float(np.sqrt(np.mean(diff ** 2)))
    rms_base  = float(np.sqrt(np.mean(baseline ** 2)))
    max_abs   = float(np.abs(diff).max())
    mean_abs  = float(np.abs(diff).mean())
    dots      = np.sum(M * baseline, axis=1)
    denom     = np.linalg.norm(M, axis=1) * np.linalg.norm(baseline, axis=1)
    cos       = float(np.mean(np.where(denom > 1e-12, dots / np.maximum(denom, 1e-12), 0.0)))
    snr       = float(10.0 * np.log10(max(rms_base ** 2, 1e-24) / max(rms_err ** 2, 1e-24)))
    frob_d    = float(np.linalg.norm(diff))
    frob_b    = float(np.linalg.norm(baseline))
    return {
        "RMS error":       rms_err,
        "Max |error|":     max_abs,
        "Mean |error|":    mean_abs,
        "RMS baseline":    rms_base,
        "Relative RMS":    rms_err / max(rms_base, 1e-12),
        "Cosine sim":      cos,
        "SNR (dB)":        snr,
        "Rel Frobenius":   frob_d / max(frob_b, 1e-12),
        "N compared":      int(div_idx),
        "diverged at":     int(div_idx) if div_idx < n_init else None,
    }


def per_head_rms(M, baseline, n_kv_head, head_dim):
    """Return list of per-head RMS errors. Length = n_kv_head."""
    n = min(M.shape[0], baseline.shape[0])
    M = M[:n].astype(np.float64).reshape(n, n_kv_head, head_dim)
    B = baseline[:n].astype(np.float64).reshape(n, n_kv_head, head_dim)
    out = []
    for h in range(n_kv_head):
        out.append(float(np.sqrt(np.mean((M[:, h, :] - B[:, h, :]) ** 2))))
    return out


def first_word_divergence(a: str, b: str):
    """Return number of leading whitespace-separated tokens that match between a and b."""
    wa, wb = a.split(), b.split()
    n = 0
    for x, y in zip(wa, wb):
        if x == y:
            n += 1
        else:
            break
    return n


def get_assistant_text(meta: dict) -> str:
    if meta.get("assistant_output"):
        return meta["assistant_output"]
    out = []
    for m in meta.get("messages", []):
        if m.get("role") == "assistant":
            out.append(m.get("content", ""))
    return "".join(out)


def get_shared_prompt(meta: dict, role: str) -> str:
    out = []
    for m in meta.get("messages", []):
        if m.get("role") == role:
            out.append(m.get("content", ""))
    return "\n".join(out)


def render_text_with_divergence(text: str, baseline_text: str, is_baseline: bool):
    """Render text as a list of html spans; matched prefix in black, diverged suffix in red.
    Baseline itself is rendered all-green."""
    if is_baseline:
        return [html.Span(text or "(empty)", style={"color": "#178a17"})]
    if not text:
        return [html.Span("(empty)", style={"color": "#999"})]
    n_match = first_word_divergence(baseline_text, text)
    words = text.split()
    matched = " ".join(words[:n_match])
    diverged = " ".join(words[n_match:])
    spans = []
    if matched:
        spans.append(html.Span(matched, style={"color": "#333"}))
    if diverged:
        spans.append(html.Span(" " + diverged if matched else diverged,
                               style={"color": "#c62828", "fontWeight": "bold"}))
    return spans


# ---------- helpers ----------

def find_baseline(runs):
    for name, (meta, layers, kv) in runs.items():
        if kv["type_k"] == "f16" and not kv["pca_enabled"]:
            return name
    return None


def short_name(run_name: str, kv: dict) -> str:
    """Compact label like 'Q4_0 Had' / 'Q4_0 PCA' / 'F16 base'."""
    tk = kv["type_k"]
    if kv["pca_enabled"]:
        return f"{tk} PCA"
    if tk == "f16":
        return "F16 base"
    return f"{tk} Had"


# ---------- Dash app ----------

def build_text_tab(runs, baseline_name, short_name_fn):
    """Static side-by-side of each run's generated text. Words after the first
    divergence from the F16 baseline are highlighted red."""
    baseline_text = ""
    if baseline_name:
        baseline_text = get_assistant_text(runs[baseline_name][0])

    first_meta = next(iter(runs.values()))[0]
    sys_text  = get_shared_prompt(first_meta, "system")
    user_text = get_shared_prompt(first_meta, "user")

    shared_box = {
        "border": "1px solid #ddd", "background": "#fafafa", "padding": "8px",
        "marginBottom": "8px", "maxHeight": "120px", "overflowY": "auto",
        "fontFamily": "monospace", "fontSize": "12px", "whiteSpace": "pre-wrap",
    }

    children = [
        html.H3("Conversation", style={"marginTop": "0"}),
        html.H4("System prompt (shared)", style={"fontSize": "14px"}),
        html.Pre(sys_text or "(none)", style=shared_box),
        html.H4("User prompt (shared)", style={"fontSize": "14px"}),
        html.Pre(user_text, style=shared_box),
        html.H4("Assistant output per run", style={"fontSize": "14px", "marginTop": "16px"}),
        html.Div("Matched prefix is black; diverged suffix from baseline is red. Baseline shown green.",
                 style={"color": "#666", "fontSize": "12px", "marginBottom": "12px"}),
    ]

    panels = []
    for name, (meta, layers, kv) in runs.items():
        text = get_assistant_text(meta)
        is_base = (name == baseline_name)
        n_words_total = len(text.split())
        n_words_match = first_word_divergence(baseline_text, text) if not is_base else n_words_total
        div_summary = ("baseline" if is_base else
                       (f"matches first {n_words_match}/{n_words_total} words"
                        if n_words_match < n_words_total else
                        f"identical ({n_words_total} words)"))
        panel = html.Div([
            html.Div([
                html.Strong(short_name_fn(name, kv)),
                html.Span(f"  · ctk={kv['type_k']} ctv={kv['type_v']} pca={kv['pca_enabled']} "
                          f"· n_tokens={kv['n_tokens_used']} · {len(text)} chars",
                          style={"color": "#666", "fontSize": "11px", "marginLeft": "8px"}),
            ], style={"marginBottom": "4px"}),
            html.Div(render_text_with_divergence(text, baseline_text, is_base),
                     style={"fontFamily": "monospace", "fontSize": "12px", "whiteSpace": "pre-wrap",
                            "maxHeight": "300px", "overflowY": "auto",
                            "background": "#fcfcfc", "padding": "6px", "border": "1px solid #eee"}),
            html.Div(div_summary, style={"color": "#666", "fontSize": "11px", "marginTop": "4px",
                                         "fontStyle": "italic"}),
        ], style={"border": "1px solid #ddd", "padding": "8px", "backgroundColor": "white"})
        panels.append(panel)

    children.append(html.Div(panels, style={
        "display": "grid", "gridTemplateColumns": "1fr 1fr",
        "gap": "10px", "alignItems": "start",
    }))
    return children


def build_app(runs, pca_scales):
    layer_ids = sorted(next(iter(runs.values()))[1].keys())
    baseline_name = find_baseline(runs)
    non_baseline = [n for n in runs if n != baseline_name]
    app = dash.Dash(__name__)
    app.title = f"KV cache visualizer — {DUMPS_DIR.name}"

    sidebar_style = {
        "width": "340px", "flexShrink": "0", "padding": "16px",
        "background": "#f7f7f7", "borderRight": "1px solid #ddd",
        "overflowY": "auto", "height": "100vh", "boxSizing": "border-box",
    }
    main_style = {"flexGrow": "1", "padding": "16px", "overflowX": "hidden"}
    row_flex   = {"display": "flex", "gap": "8px", "alignItems": "center", "marginBottom": "12px"}
    table_style = {"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"}
    td_th = {"border": "1px solid #ccc", "padding": "4px 6px", "textAlign": "right"}
    td_th_l = dict(td_th, textAlign="left")

    app.layout = html.Div([
        html.Div([
            html.H3("KV cache visualizer", style={"marginTop": "0"}),
            html.Div(f"dir: {DUMPS_DIR}",
                     style={"color": "#666", "fontSize": "11px", "marginBottom": "4px", "fontFamily": "monospace"}),
            html.Div(f"{len(runs)} runs loaded · baseline = {baseline_name or 'NONE'}",
                     style={"color": "#666", "fontSize": "12px", "marginBottom": "12px"}),

            html.Label("Layer"),
            dcc.Dropdown(
                id="layer-dd",
                options=[{"label": f"Layer {il}", "value": il} for il in layer_ids],
                value=layer_ids[0],
                clearable=False,
            ),

            html.Label("Cache", style={"marginTop": "12px"}),
            dcc.Dropdown(
                id="mode-dd",
                options=[{"label": "K (keys)", "value": "K"},
                         {"label": "V (values)", "value": "V"}],
                value="K", clearable=False,
            ),

            html.Label("Space", style={"marginTop": "12px"}),
            dcc.RadioItems(
                id="space-dd",
                options=[{"label": "Original (back-rotated)", "value": "orig"},
                         {"label": "Raw stored",                "value": "raw"}],
                value="orig", labelStyle={"display": "block"},
            ),
            html.Div("Original space: each run back-rotated so K/V values are directly "
                     "comparable to F16 baseline. Raw: shows what's literally stored "
                     "(PCA-rotated / Hadamard-rotated).",
                     style={"fontSize": "11px", "color": "#666", "marginTop": "4px"}),

            html.Div([
                html.H4("Reading this dashboard", style={"fontSize": "12px", "marginTop": "16px", "marginBottom": "6px"}),
                html.Ul([
                    html.Li("Baseline = F16 with no PCA. All other runs are compared against it."),
                    html.Li("Metrics auto-truncate to the pre-divergence prefix when runs fork in their generated tokens. "
                            "Past the fork, K/V matrices are for different tokens and the diff stops being quantization error."),
                    html.Li("Red dashed line on heatmaps = divergence point. Left of it = real quantization noise. "
                            "Right of it = different-token noise."),
                    html.Li("SNR (dB) higher = better. PCA wins if its SNR > Hadamard's SNR."),
                ], style={"fontSize": "11px", "color": "#444", "paddingLeft": "18px", "margin": "0"}),
            ], style={"marginTop": "12px", "padding": "8px", "background": "#fffaeb",
                       "border": "1px solid #f0d000", "borderRadius": "4px"}),

            html.Hr(),
            html.H4("Metrics vs F16 baseline", style={"marginBottom": "8px"}),
            html.Div(id="stats-table"),
            html.Div([
                html.H5("Per-head RMS error", style={"marginTop": "12px"}),
                dcc.Graph(id="per-head-bar", config={"displayModeBar": False},
                          style={"height": "180px"}),
            ]),
        ], style=sidebar_style),

        html.Div([
            dcc.Tabs([
                dcc.Tab(label="Current layer", children=[
                    html.H3(id="cur-layer-title", style={"marginTop": "0"}),
                    html.H4("Heatmaps"),
                    dcc.Graph(id="hm-row", style={"height": "360px"}),
                    html.H4("Diff vs F16 baseline"),
                    dcc.Graph(id="hm-diff-row", style={"height": "360px"}),
                ]),
                dcc.Tab(label="Across layers", children=[
                    html.H3("Cross-layer metrics", style={"marginTop": "0"}),
                    dcc.RadioItems(
                        id="cross-metric",
                        options=[{"label": v, "value": v} for v in
                                 ["RMS error", "Max |error|", "Mean |error|", "Relative RMS",
                                  "Cosine sim", "SNR (dB)", "Rel Frobenius", "N compared"]],
                        value="RMS error", inline=True,
                    ),
                    html.Div([
                        dcc.Graph(id="cross-bar", style={"height": "500px", "flex": "1"}),
                    ], style={"display": "flex"}),
                    html.H4("Per-layer values"),
                    html.Div("Tip: click a bar to jump to that layer. "
                             "Metrics auto-truncate to pre-divergence prefix per layer; "
                             "use 'N compared' to see how many tokens each layer actually used.",
                             style={"color": "#666", "marginBottom": "8px", "fontSize": "12px"}),
                    dash_table.DataTable(
                        id="cross-table",
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "right", "padding": "4px 8px", "fontFamily": "monospace", "fontSize": "12px"},
                        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
                        style_cell_conditional=[{"if": {"column_id": "layer"}, "textAlign": "left"}],
                    ),
                    html.H4("Full breakdown (all metrics × all layers)", style={"marginTop": "16px"}),
                    dash_table.DataTable(
                        id="cross-full-table",
                        style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
                        style_cell={"textAlign": "right", "padding": "3px 6px", "fontFamily": "monospace", "fontSize": "11px"},
                        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
                        style_cell_conditional=[{"if": {"column_id": "layer"}, "textAlign": "left"},
                                                {"if": {"column_id": "metric"}, "textAlign": "left"}],
                    ),
                ]),
                dcc.Tab(label="Distributions", children=[
                    html.H3("Value distributions", style={"marginTop": "0"}),
                    dcc.Graph(id="hist-plot", style={"height": "500px"}),
                ]),
                dcc.Tab(label="Text", children=build_text_tab(runs, baseline_name, short_name)),
            ]),
        ], style=main_style),
    ], style={"display": "flex", "minHeight": "100vh"})

    # -------- callbacks --------

    def current_matrices(il, mode, space):
        """Return list of (short_label, M_orig_or_raw, run_name) for all runs."""
        out = []
        for name, (meta, layers, kv) in runs.items():
            M_stored = layers[il][mode]
            if space == "orig":
                M = to_original(M_stored, il, mode, kv, pca_scales)
            else:
                M = M_stored
            out.append((short_name(name, kv), M, name))
        return out

    def baseline_M(il, mode, space):
        if baseline_name is None:
            return None
        kv = runs[baseline_name][2]
        M_stored = runs[baseline_name][1][il][mode]
        if space == "orig":
            return to_original(M_stored, il, mode, kv, pca_scales)
        return M_stored

    @app.callback(
        Output("stats-table", "children"),
        Input("layer-dd", "value"), Input("mode-dd", "value"), Input("space-dd", "value"),
    )
    def update_stats(il, mode, space):
        if baseline_name is None or space == "raw":
            return html.Div("(stats only meaningful in original space — switch via the Space radio)")
        B = baseline_M(il, mode, space)
        rows = []
        for name, (meta, layers, kv) in runs.items():
            if name == baseline_name:
                continue
            M = to_original(layers[il][mode], il, mode, kv, pca_scales)
            m = compute_metrics(M, B)
            m["__label"] = short_name(name, kv)
            rows.append(m)
        header = ["Metric"] + [r["__label"] for r in rows]
        keys = ["RMS error", "Max |error|", "Mean |error|", "RMS baseline",
                "Relative RMS", "Cosine sim", "SNR (dB)", "Rel Frobenius", "N compared"]
        body = []
        for k in keys:
            cells = [html.Td(k, style=td_th_l)]
            for r in rows:
                v = r[k]
                if k == "N compared":
                    cells.append(html.Td(str(int(v)), style=td_th))
                elif k == "SNR (dB)":
                    cells.append(html.Td(f"{v:.2f}", style=td_th))
                elif abs(v) < 100:
                    cells.append(html.Td(f"{v:.4f}", style=td_th))
                else:
                    cells.append(html.Td(f"{v:.2f}", style=td_th))
            body.append(html.Tr(cells))

        # divergence note
        div_note = None
        for r in rows:
            d = r.get("diverged at")
            if d is not None:
                div_note = (f"⚠ Runs diverged at ~token {d}. Metrics above averaged over the "
                            f"pre-divergence prefix only — past token {d} the runs are computing K/V "
                            f"for different tokens, so the diff stops being quantization error.")
                break

        return html.Div([
            html.Table(
                [html.Tr([html.Th(h, style=td_th_l if i == 0 else td_th) for i, h in enumerate(header)])] + body,
                style=table_style,
            ),
            html.Div(div_note, style={"color": "#c62828", "fontSize": "11px",
                                       "marginTop": "8px", "fontStyle": "italic"}) if div_note
            else html.Div("✓ All runs agree on token stream through the compared range.",
                          style={"color": "#178a17", "fontSize": "11px", "marginTop": "8px"}),
        ])

    @app.callback(
        Output("per-head-bar", "figure"),
        Input("layer-dd", "value"), Input("mode-dd", "value"), Input("space-dd", "value"),
    )
    def update_per_head(il, mode, space):
        if baseline_name is None or space == "raw":
            return go.Figure(layout={"title": "(original space only)"})
        B = baseline_M(il, mode, space)
        n_kv_head = runs[baseline_name][2]["n_kv_head"]
        head_dim  = runs[baseline_name][2]["head_dim_k" if mode == "K" else "head_dim_v"]
        fig = go.Figure()
        for name, (meta, layers, kv) in runs.items():
            if name == baseline_name:
                continue
            M = to_original(layers[il][mode], il, mode, kv, pca_scales)
            errs = per_head_rms(M, B, n_kv_head, head_dim)
            fig.add_trace(go.Bar(x=[f"h{h}" for h in range(n_kv_head)], y=errs,
                                 name=short_name(name, kv)))
        fig.update_layout(barmode="group", margin={"l": 8, "r": 8, "t": 8, "b": 8},
                          legend={"orientation": "h", "y": -0.2}, height=180)
        return fig

    @app.callback(
        Output("hm-row", "figure"), Output("hm-diff-row", "figure"),
        Output("cur-layer-title", "children"),
        Input("layer-dd", "value"), Input("mode-dd", "value"), Input("space-dd", "value"),
    )
    def update_heatmaps(il, mode, space):
        mats = current_matrices(il, mode, space)
        B = baseline_M(il, mode, space) if space == "orig" else None

        # shared color limits across all runs (so the eye can compare directly)
        all_vals = np.concatenate([M.flatten() for _, M, _ in mats if M is not None])
        vabs = max(abs(np.percentile(all_vals, 1)), abs(np.percentile(all_vals, 99)), 1e-3)

        # divergence point (using F16 baseline vs first non-baseline run, in original space)
        div_idx = None
        if B is not None:
            for label, M, name in mats:
                if M is None or name is None or name == baseline_name:
                    continue
                div_idx = find_divergence(M, B)
                if div_idx < min(M.shape[0], B.shape[0]):
                    break  # found a real divergence
                div_idx = None

        n = len(mats)
        fig_hm = make_subplots(
            rows=1, cols=n,
            subplot_titles=[label for label, _, _ in mats],
            horizontal_spacing=0.04 / max(n, 1) * 2,
        )
        for i, (label, M, _) in enumerate(mats):
            if M is None:
                continue
            fig_hm.add_trace(go.Heatmap(
                z=M.T, colorscale="RdBu", zmid=0, zmin=-vabs, zmax=vabs,
                colorbar={"thickness": 8} if i == n - 1 else None,
                showscale=(i == n - 1),
                hovertemplate="token=%{x}<br>dim=%{y}<br>v=%{z:.4f}<extra></extra>",
            ), row=1, col=i + 1)
            fig_hm.update_xaxes(title_text="token", row=1, col=i + 1)
            if i == 0:
                fig_hm.update_yaxes(title_text="head_dim", row=1, col=i + 1)
            if div_idx is not None and space == "orig":
                fig_hm.add_vline(x=div_idx, line=dict(color="red", width=1.5, dash="dash"),
                                 row=1, col=i + 1)
        title_hm = "Heatmaps (red dashed = divergence point)"
        fig_hm.update_layout(
            margin={"l": 8, "r": 8, "t": 40, "b": 8}, height=360,
            annotations=[dict(font=dict(size=11))],
        )

        # diffs: skip the baseline run itself
        diff_entries = []
        for label, M, name in mats:
            if M is None or name is None or name == baseline_name or B is None:
                continue
            nn = min(M.shape[0], B.shape[0])
            diff = M[:nn] - B[:nn]
            rms_full = float(np.sqrt(np.mean(diff ** 2)))
            # recompute RMS on pre-divergence prefix for the title
            div_local = find_divergence(M[:nn], B[:nn])
            rms_prediv = float(np.sqrt(np.mean(diff[:div_local] ** 2))) if div_local > 0 else 0.0
            diff_entries.append((label, diff, rms_prediv, div_local, nn))

        nd = max(len(diff_entries), 1)
        fig_diff = make_subplots(
            rows=1, cols=nd,
            subplot_titles=[
                (f"{lab} − F16  (pre-div RMS={rms:.4f}" +
                 (f", diverges @ {dl})" if dl < nn else ")"))
                for lab, _, rms, dl, nn in diff_entries
            ] or ["(no non-baseline runs)"],
            horizontal_spacing=0.04 / max(nd, 1) * 2,
        )
        if not diff_entries:
            fig_diff.update_layout(annotations=[dict(text="(no baseline / no diffs)", showarrow=False)])
        for i, (label, diff, rms, div_local, nn) in enumerate(diff_entries):
            dabs = max(abs(np.percentile(diff, 99)), 1e-6)
            fig_diff.add_trace(go.Heatmap(
                z=diff.T, colorscale="RdBu", zmid=0, zmin=-dabs, zmax=dabs,
                colorbar={"thickness": 8} if i == len(diff_entries) - 1 else None,
                showscale=(i == len(diff_entries) - 1),
                hovertemplate="token=%{x}<br>dim=%{y}<br>d=%{z:.4f}<extra></extra>",
            ), row=1, col=i + 1)
            fig_diff.update_xaxes(title_text="token", row=1, col=i + 1)
            if i == 0:
                fig_diff.update_yaxes(title_text="head_dim", row=1, col=i + 1)
            if div_local < nn:
                fig_diff.add_vline(x=div_local, line=dict(color="red", width=1.5, dash="dash"),
                                   row=1, col=i + 1)
        fig_diff.update_layout(margin={"l": 8, "r": 8, "t": 40, "b": 8}, height=360)

        title = f"Layer {il} · {mode}"
        if div_idx is not None:
            title += f"  ·  ⚠ runs diverge at token {div_idx} (left of red line = real quantization error)"
        return fig_hm, fig_diff, title

    @app.callback(
        Output("cross-bar", "figure"),
        Output("cross-table", "data"),
        Output("cross-table", "columns"),
        Output("cross-full-table", "data"),
        Output("cross-full-table", "columns"),
        Input("mode-dd", "value"), Input("cross-metric", "value"), Input("space-dd", "value"),
    )
    def update_cross(mode, metric, space):
        empty_fig = go.Figure(layout={"annotations": [{"text": "(original space only)", "showarrow": False}]})
        if space == "raw" or baseline_name is None:
            return empty_fig, [], [], [], []

        # compute metrics for every (layer, non-baseline run)
        per_layer = {il: {} for il in layer_ids}  # il -> run_name -> metrics dict
        B_per_layer = {}
        for il in layer_ids:
            B_per_layer[il] = to_original(runs[baseline_name][1][il][mode], il, mode,
                                          runs[baseline_name][2], pca_scales)
        for name, (meta, layers, kv) in runs.items():
            if name == baseline_name:
                continue
            for il in layer_ids:
                M = to_original(layers[il][mode], il, mode, kv, pca_scales)
                per_layer[il][name] = compute_metrics(M, B_per_layer[il])

        # bar chart for the selected metric
        fig = go.Figure()
        for name, (meta, layers, kv) in runs.items():
            if name == baseline_name:
                continue
            ys = [per_layer[il][name][metric] for il in layer_ids]
            fig.add_trace(go.Bar(x=[f"L{il}" for il in layer_ids], y=ys, name=short_name(name, kv)))
        fig.update_layout(barmode="group", title=f"{metric} per layer ({mode})",
                          xaxis_title="layer", yaxis_title=metric,
                          legend={"orientation": "h", "y": -0.15}, height=500)

        # single-metric table: rows = layers, columns = methods
        non_base_runs = [n for n in runs if n != baseline_name]
        short_labels = {n: short_name(n, runs[n][2]) for n in non_base_runs}
        columns = [{"name": "layer", "id": "layer"}] + [{"name": short_labels[n], "id": n} for n in non_base_runs]
        rows = []
        for il in layer_ids:
            row = {"layer": f"L{il}"}
            for n in non_base_runs:
                v = per_layer[il][n][metric]
                row[n] = f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}"
            rows.append(row)

        # full breakdown: rows = (layer, metric), columns = methods
        metric_keys = ["RMS error", "Max |error|", "Mean |error|", "RMS baseline",
                       "Relative RMS", "Cosine sim", "SNR (dB)", "Rel Frobenius"]
        full_columns = [{"name": "layer", "id": "layer"},
                        {"name": "metric", "id": "metric"}] + \
                       [{"name": short_labels[n], "id": n} for n in non_base_runs]
        full_rows = []
        for il in layer_ids:
            for mk in metric_keys:
                row = {"layer": f"L{il}", "metric": mk}
                for n in non_base_runs:
                    v = per_layer[il][n][mk]
                    if mk == "Cosine sim":
                        row[n] = f"{v:.6f}"
                    elif mk == "SNR (dB)":
                        row[n] = f"{v:.2f}"
                    elif abs(v) < 100:
                        row[n] = f"{v:.4f}"
                    else:
                        row[n] = f"{v:.2f}"
                full_rows.append(row)

        return fig, rows, columns, full_rows, full_columns

    @app.callback(
        Output("hist-plot", "figure"),
        Input("layer-dd", "value"), Input("mode-dd", "value"), Input("space-dd", "value"),
    )
    def update_hist(il, mode, space):
        mats = current_matrices(il, mode, space)
        fig = go.Figure()
        for label, M, _ in mats:
            if M is None:
                continue
            flat = M.flatten()
            lo, hi = np.percentile(flat, 0.5), np.percentile(flat, 99.5)
            bins = np.linspace(lo, hi, 80)
            hist, edges = np.histogram(flat, bins=bins)
            fig.add_trace(go.Bar(x=edges[:-1], y=hist, name=label, opacity=0.7))
        fig.update_layout(title=f"Value distribution · {mode} L{il} · {space}",
                          barmode="overlay", xaxis_title="value", yaxis_title="count")
        return fig

    # cross-tab click: jump to layer (uses clickData on cross-bar)
    @app.callback(
        Output("layer-dd", "value"),
        Input("cross-bar", "clickData"),
        State("layer-dd", "value"),
        prevent_initial_call=True,
    )
    def cross_click(clickData, current):
        if not clickData:
            return current
        try:
            label = clickData["points"][0]["label"]  # e.g. "L11"
            return int(label[1:])
        except Exception:
            return current

    return app


def main():
    if not DUMPS_DIR.exists():
        raise SystemExit(f"DUMPS_DIR not found: {DUMPS_DIR}")
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
        print(f"loaded: {d.name}  pca={kv['pca_enabled']}  ctk={kv['type_k']}  n_tokens={kv['n_tokens_used']}")
    if not runs:
        raise SystemExit("no runs found")
    print(f"loading PCA scales from {PCA_PT}")
    try:
        pca_scales = load_pca_scales()
        print(f"  layers: {sorted(pca_scales.keys())}")
    except Exception as e:
        print(f"  failed ({e}) — PCA back-rotation will be skipped")
        pca_scales = {}
    app = build_app(runs, pca_scales)
    print("\nopen  http://127.0.0.1:8050  in a browser\n")
    app.run(debug=False, port=8050)


if __name__ == "__main__":
    main()
