"""Static figures (matplotlib) for the poster/paper and interactive ones (plotly) for the app.

Colour roles are fixed across every figure:
  honest / pushback       blue   #2a78d6
  sycophantic             red    #e34948
  baseline run            gray   #898781
  steered run             orange #eb6834
  neutral                 gray   #c3c2b7
"""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HONEST, SYC, BASE, STEER, NEUTRAL = "#2a78d6", "#e34948", "#898781", "#eb6834", "#c3c2b7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2)


def plot_layer_sweep(sep: pd.DataFrame, chosen_layer: int, path: Path) -> Path:
    """Held-out AUROC and pairwise accuracy per layer: which layers encode the distinction."""
    hold = sep[sep.split == "holdout"].sort_values("layer")
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=180)
    ax.plot(hold.layer, hold.auroc, color=HONEST, lw=2, marker="o", ms=4, label="Held-out AUROC")
    ax.plot(hold.layer, hold.pairwise_accuracy, color=STEER, lw=2, marker="s", ms=4, label="Held-out pairwise accuracy")
    ax.axhline(0.5, color=BASE, ls=":", lw=1)
    ax.text(hold.layer.max(), 0.51, "chance", color=BASE, ha="right", va="bottom", fontsize=8)
    ax.axvline(chosen_layer, color=INK2, ls="--", lw=1)
    ax.text(chosen_layer + 0.2, 0.03, f"steering layer {chosen_layer}", color=INK2, fontsize=8, rotation=90, va="bottom")
    ax.set_ylim(0, 1.05)
    _style(ax, "Which layers separate honest from sycophantic replies?", "Decoder layer (output)", "Score on held-out pairs")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_separation(art, layer: int, path: Path) -> Path:
    """Distribution of held-out reply projections on the honest..sycophantic scale."""
    mask = art.is_holdout
    h = art.score(layer, art.honest_proj[mask, layer]).numpy()
    s = art.score(layer, art.syc_proj[mask, layer]).numpy()
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=180)
    lo, hi = float(min(h.min(), s.min(), -1.5)), float(max(h.max(), s.max(), 1.5))
    bins = np.linspace(lo, hi, 25)
    for vals, color, name in ((h, HONEST, "Honest"), (s, SYC, "Sycophantic")):
        ax.hist(vals, bins=bins, histtype="stepfilled", color=color, alpha=0.30, lw=0)
        ax.hist(vals, bins=bins, histtype="step", color=color, lw=2, label=f"{name} replies (n={len(vals)})")
    for x in (-1, 1):
        ax.axvline(x, color=BASE, ls=":", lw=1)
    from .extraction import auroc
    _style(ax, f"Held-out separation at layer {layer} (AUROC {auroc(s, h):.2f})",
           "Sycophancy score  (−1 = typical honest reply, +1 = typical sycophantic reply)", "Replies")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_dose_response(summary: pd.DataFrame, path: Path, judge_name: str = "") -> Path:
    """Two panels on one x-axis: what the judge saw, and what the internal readout saw."""
    df = summary[summary.source == "all"].sort_values("alpha")
    has_labels = df["n_labeled"].sum() > 0
    fig, axes = plt.subplots(1, 2 if has_labels else 1, figsize=(11 if has_labels else 6, 4), dpi=180, squeeze=False)
    axes = axes[0]
    if has_labels:
        ax = axes[0]
        for label, color, marker in (("pushback", HONEST, "o"), ("sycophantic", SYC, "s"), ("neutral", NEUTRAL, "^")):
            y = df[f"{label}_rate"] * 100
            if label != "neutral":  # keep the bands readable: CIs for the two poles only
                ax.fill_between(df.alpha, df[f"{label}_lo"] * 100, df[f"{label}_hi"] * 100, color=color, alpha=0.12, lw=0)
            ax.plot(df.alpha, y, color=color, lw=2, marker=marker, ms=6, label=label.capitalize())
        ax.axvline(0, color=BASE, ls=":", lw=1)
        ax.set_ylim(-3, 103)
        n = int(df["n_labeled"].max())
        _style(ax, "Behaviour: judged stance vs steering strength", "Steering strength α  (− pushback, + sycophancy)",
               f"% of {n} prompts (95% CI)")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
        if judge_name:
            ax.text(1, 1.01, f"judge: {judge_name}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=BASE)
    ax = axes[-1]
    ax.errorbar(df.alpha, df.mean_score, yerr=1.96 * df.mean_score_sem.fillna(0), color=STEER, lw=2, marker="o",
                ms=6, capsize=3)
    ax.axhspan(-1.2, -0.8, color=HONEST, alpha=0.08, lw=0)
    ax.axhspan(0.8, 1.2, color=SYC, alpha=0.08, lw=0)
    ax.axhline(0, color=BASE, ls=":", lw=1)
    ax.text(0.01, 1.0, "typical sycophantic reply", transform=ax.get_yaxis_transform(), color=SYC, fontsize=8, va="bottom")
    ax.text(0.01, -1.0, "typical honest reply", transform=ax.get_yaxis_transform(), color=HONEST, fontsize=8, va="top")
    _style(ax, "Internals: readout-layer sycophancy score", "Steering strength α", "Mean per-token score (±95% CI)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- app visuals
def token_trace_figure(base, steered, readout_layer: int):
    """Per-token sycophancy score for the baseline and steered replies (plotly)."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_hrect(y0=0.6, y1=1.4, fillcolor=SYC, opacity=0.07, line_width=0)
    fig.add_hrect(y0=-1.4, y1=-0.6, fillcolor=HONEST, opacity=0.07, line_width=0)
    fig.add_hline(y=0, line=dict(color=BASE, dash="dot", width=1))
    for gen, name, color, dash in ((base, "Baseline (α = 0)", BASE, "dash"), (steered, f"Steered (α = {steered.alpha:+g})", STEER, "solid")):
        fig.add_trace(go.Scatter(
            x=list(range(1, len(gen.scores) + 1)), y=gen.scores, mode="lines+markers", name=name,
            line=dict(color=color, width=2, dash=dash), marker=dict(size=6),
            customdata=[t.replace("\n", "⏎") for t in gen.tokens],
            hovertemplate="token %{x}: <b>%{customdata}</b><br>score %{y:+.2f}<extra>" + name + "</extra>"))
    fig.add_annotation(xref="paper", x=0.005, y=1.0, text="typical sycophantic reply", showarrow=False,
                       font=dict(size=10, color=SYC), xanchor="left")
    fig.add_annotation(xref="paper", x=0.005, y=-1.0, text="typical honest reply", showarrow=False,
                       font=dict(size=10, color=HONEST), xanchor="left")
    fig.update_layout(
        title=dict(text=f"Sycophancy score per generated token (read at layer {readout_layer})", x=0, font=dict(size=14)),
        xaxis_title="Generated token #", yaxis_title="Score (−1 honest … +1 sycophantic)",
        template="plotly_white", height=360, margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.25), hovermode="closest")
    return fig


def highlighted_html(gen, title: str) -> str:
    """Reply text with each token tinted by its score: blue toward honest, red toward sycophantic."""
    spans = []
    for tok, s in zip(gen.tokens, gen.scores):
        a = min(abs(s), 1.5) / 1.5 * 0.55
        rgb = "227,73,72" if s > 0 else "42,120,214"
        safe = html.escape(tok).replace("\n", "<br>")
        spans.append(f'<span title="{s:+.2f}" style="background:rgba({rgb},{a:.2f});border-radius:3px;'
                     f'padding:1px 0">{safe}</span>')
    mean = gen.mean_score
    verdict = "leans sycophantic" if mean > 0.25 else "leans honest" if mean < -0.25 else "in between"
    return (f'<div style="font-family:system-ui,sans-serif;line-height:1.7">'
            f'<div style="font-weight:600;margin-bottom:4px">{html.escape(title)}</div>'
            f'<div style="font-size:12px;opacity:.75;margin-bottom:6px">mean score {mean:+.2f} · {verdict}</div>'
            f'<div>{"".join(spans) or "<i>(empty reply)</i>"}</div></div>')


def colour_legend_html() -> str:
    return ('<div style="font-family:system-ui,sans-serif;font-size:12px;opacity:.8">'
            '<span style="background:rgba(42,120,214,.5);padding:1px 6px;border-radius:3px">honest-like</span> '
            '<span style="padding:1px 6px">neutral</span> '
            '<span style="background:rgba(227,73,72,.5);padding:1px 6px;border-radius:3px">sycophantic-like</span>'
            ' &nbsp;Hover a token to see its score.</div>')
