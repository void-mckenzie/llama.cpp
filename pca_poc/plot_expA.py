#!/usr/bin/env python3
"""
Parse Exp A KL logs and produce 3 separate story-driven charts:

  Chart 1: "Mean-only wins everywhere" — Mean KLD by config, plain vs mean,
           with model-only floor reference lines.
  Chart 2: "Where does the degradation come from?" — Stacked bars per quantized
           model config showing model-only floor + KV-cache+rotation contribution.
  Chart 3: "Distribution shape" — Heatmap of KLD percentiles per config.

Plus printed tables (Table 1: vs BF16+F16, Table 2: vs same-model F16).

Usage:
    python plot_expA.py
"""
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LOGS_DIR = Path(r"F:/Repos/llama.cpp/kl_dumps/logs")

FILENAME_RE = re.compile(
    r"expA_(bf16|q8xl|q4xl)_(f16|q4_0|q8_0)_(none|plain|mean)_vs_(bf16|self)\.log"
)

PERCENTILE_KEYS = [
    ("Median", "p50"),
    ("95.0%", "p95"),
    ("99.0%", "p99"),
    ("99.9%", "p999"),
]


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


def parse_log(path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    model, kv_quant, strategy, ref = m.groups()
    text = path.read_text(encoding="utf-8", errors="replace")
    rec = {
        "model": model,
        "kv_quant": kv_quant,
        "strategy": strategy,
        "reference": ref,
        "mean_kld": extract_metric("Mean    KLD", text),
        "median_kld": extract_metric("Median  KLD", text),
        "max_kld": extract_metric("Maximum KLD", text),
        "rms_dp": extract_metric("RMS Δp", text),
        "same_top_p": extract_metric("Same top p", text),
        "mean_ppl_q": extract_metric("Mean PPL(Q)", text),
    }
    for label, key in PERCENTILE_KEYS:
        rec[key] = extract_percentile(label, text)
    return rec


def load_all():
    records = []
    for p in sorted(LOGS_DIR.glob("expA_*.log")):
        r = parse_log(p)
        if r:
            r["file"] = p.name
            records.append(r)
    return pd.DataFrame(records)


def print_tables(df):
    cols = ["model", "kv_quant", "strategy", "mean_kld", "median_kld",
            "max_kld", "p95", "p99", "rms_dp", "same_top_p"]

    print("\n" + "=" * 110)
    print("TABLE 1: KL vs universal BF16+F16 baseline (total degradation from gold standard)")
    print("=" * 110)
    t1 = df[df["reference"] == "bf16"][cols].sort_values(["model", "kv_quant", "strategy"])
    print(t1.to_string(index=False, float_format=lambda x: f"{x:.5f}" if pd.notna(x) else "-"))

    print("\n" + "=" * 110)
    print("TABLE 2: KL vs same-model F16 baseline (isolates KV-cache + rotation contribution)")
    print("=" * 110)
    t2 = df[df["reference"] == "self"][cols].sort_values(["model", "kv_quant", "strategy"])
    print(t2.to_string(index=False, float_format=lambda x: f"{x:.5f}" if pd.notna(x) else "-"))


def model_only_floor(df, model):
    row = df[(df["model"] == model) & (df["kv_quant"] == "f16") &
             (df["strategy"] == "none") & (df["reference"] == "bf16")]
    return row["mean_kld"].iloc[0] if len(row) else None


def get_kld(df, model, kv_quant, strategy, reference):
    row = df[(df["model"] == model) & (df["kv_quant"] == kv_quant) &
             (df["strategy"] == strategy) & (df["reference"] == reference)]
    return row["mean_kld"].iloc[0] if len(row) else None


# ============================================================
# CHART 1: Mean-only wins everywhere
# ============================================================
def chart1_plain_vs_mean(df):
    models = ["bf16", "q8xl", "q4xl"]
    kv_quants = ["q4_0", "q8_0"]
    strategies = ["plain", "mean"]
    colors = {"plain": "#1f77b4", "mean": "#ff7f0e"}

    fig = go.Figure()
    x_labels = [f"{m} / {k}" for m in models for k in kv_quants]

    for strategy in strategies:
        ys = []
        for m in models:
            for k in kv_quants:
                v = get_kld(df, m, k, strategy, "bf16")
                ys.append(v)
        fig.add_trace(go.Bar(
            x=x_labels, y=ys,
            name=f"{strategy} Hadamard",
            marker_color=colors[strategy],
            hovertemplate="%{x}<br>" + f"{strategy}<br>" + "Mean KLD=%{y:.5f}<extra></extra>",
        ))

    # Model-only floor lines (only for quantized models)
    for model, color in [("q8xl", "#666"), ("q4xl", "#333")]:
        floor = model_only_floor(df, model)
        if floor is None:
            continue
        fig.add_hline(
            y=floor, line_dash="dash", line_color=color, line_width=1.5,
            annotation_text=f"{model} + F16 floor ({floor:.5f})",
            annotation_position="right",
            annotation_font_color=color,
        )

    fig.update_layout(
        title=dict(
            text="<b>Chart 1: Mean-only wins everywhere</b><br>"
                 "<sup>Mean KLD vs universal BF16+F16 baseline. Lower = better. "
                 "Dashed lines = irreducible weight-quant floor.</sup>",
            x=0.5,
        ),
        xaxis_title="model weight quant / KV cache quant",
        yaxis_title="Mean KLD (vs BF16+F16 baseline)",
        barmode="group",
        legend_title="Rotation strategy",
        height=550,
        hovermode="x unified",
    )
    return fig


# ============================================================
# CHART 2: Where does the degradation come from?
# ============================================================
def chart2_attribution(df):
    """Three side-by-side bars per config:
       - Floor: model weight quant only (KL of model+F16 vs BF16+F16)
       - KV-only: KL of config vs same-model F16 (= cache quant + rotation effect)
       - Total:  KL of config vs universal BF16+F16 (= floor + KV-only, approximately)

       For BF16 model, floor = 0 (it IS the reference), so total == KV-only.
    """
    models = ["bf16", "q8xl", "q4xl"]
    kv_quants = ["q4_0", "q8_0"]
    strategies = ["plain", "mean"]

    fig = go.Figure()

    x_labels = []
    floors = []
    kv_onlys = []
    totals = []

    for m in models:
        floor = model_only_floor(df, m) or 0.0  # BF16 has no row → floor 0
        for k in kv_quants:
            for s in strategies:
                x_labels.append(f"{m} / {k} / {s}")
                total = get_kld(df, m, k, s, "bf16")
                kv_only = get_kld(df, m, k, s, "self")
                # For BF16 model, vs_self wasn't run (would be 0); use total
                if kv_only is None:
                    kv_only = total if total is not None else 0.0
                floors.append(floor)
                kv_onlys.append(kv_only)
                totals.append(total if total is not None else 0.0)

    # Three side-by-side bars
    fig.add_trace(go.Bar(
        x=x_labels, y=floors,
        name="Model-only floor (weight quant)",
        marker_color="#7f7f7f",
        hovertemplate="%{x}<br>floor=%{y:.5f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=x_labels, y=kv_onlys,
        name="KV-cache + rotation (vs same-model F16)",
        marker_color="#1f77b4",
        hovertemplate="%{x}<br>KV-only=%{y:.5f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=x_labels, y=totals,
        name="Total (vs BF16+F16)",
        marker_color="#d62728",
        hovertemplate="%{x}<br>total=%{y:.5f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="<b>Chart 2: Where does the degradation come from?</b><br>"
                 "<sup>Three bars per config: gray = model weight quant floor, "
                 "blue = KV-cache+rotation contribution (vs same-model F16), "
                 "red = total (vs BF16+F16). Total ≈ floor + KV-only (KL not perfectly additive).</sup>",
            x=0.5,
        ),
        xaxis_title="config (model / KV quant / strategy)",
        yaxis_title="Mean KLD",
        barmode="group",
        legend_title="Metric",
        height=600,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01),
    )
    return fig


# ============================================================
# CHART 3: Distribution shape (heatmap of percentiles)
# ============================================================
def chart3_distribution(df):
    """Heatmap: rows = configs (vs_bf16), cols = percentiles."""
    pct_cols = ["p50", "p95", "p99", "p999", "max_kld"]
    pct_labels = ["p50<br>(median)", "p95", "p99", "p99.9", "max"]

    plot_df = df[(df["reference"] == "bf16") & (df["strategy"].isin(["plain", "mean"]))].copy()
    plot_df = plot_df.sort_values(["model", "kv_quant", "strategy"])

    y_labels = plot_df.apply(
        lambda r: f"{r['model']} / {r['kv_quant']} / {r['strategy']}", axis=1
    ).tolist()

    z_values = plot_df[pct_cols].values.tolist()

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=pct_labels,
        y=y_labels,
        colorscale="Viridis",
        text=[[f"{v:.5f}" if pd.notna(v) else "-" for v in row] for row in z_values],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="config=%{y}<br>percentile=%{x}<br>KLD=%{z:.5f}<extra></extra>",
        colorbar=dict(title="KLD", tickformat=".4f"),
    ))

    # Add model-only floor as annotation on right margin
    for i, row in plot_df.reset_index(drop=True).iterrows():
        floor = model_only_floor(df, row["model"])
        if floor is not None:
            fig.add_annotation(
                x=len(pct_cols) - 0.5, y=i,
                text=f"floor: {floor:.5f}",
                showarrow=False, xanchor="left", xshift=10,
                font=dict(size=9, color="#666"),
            )

    fig.update_layout(
        title=dict(
            text="<b>Chart 3: KLD distribution shape per config</b><br>"
                 "<sup>Log-color scale. Compare rows to see how mean-only compresses the tail. "
                 "'floor' = that model's weight-quant KLD floor for reference.</sup>",
            x=0.5,
        ),
        xaxis_title="KLD percentile",
        yaxis_title="config (model / KV quant / strategy)",
        height=550,
        margin=dict(l=200, r=150),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def main():
    if not LOGS_DIR.exists():
        print(f"ERROR: {LOGS_DIR} does not exist")
        return

    df = load_all()
    if df.empty:
        print(f"No expA_*.log files found in {LOGS_DIR}")
        return

    print(f"Loaded {len(df)} log files")
    print_tables(df)

    fig1 = chart1_plain_vs_mean(df)
    fig2 = chart2_attribution(df)
    fig3 = chart3_distribution(df)

    out_dir = Path(r"F:/Repos/llama.cpp/pca_poc")
    for i, fig in enumerate([fig1, fig2, fig3], 1):
        out_html = out_dir / f"expA_chart{i}.html"
        fig.write_html(str(out_html))
        print(f"Wrote {out_html}")

    # Open all three in browser
    print("\nOpening charts in browser...")
    for fig in [fig1, fig2, fig3]:
        fig.show()


if __name__ == "__main__":
    main()
