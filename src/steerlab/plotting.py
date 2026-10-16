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


STYLE_LABEL = {"standard": "Same style as training", "style_swapped": "Style swapped (calm sycophant, excited honest)"}
SWAP = "#6250d6"  # violet: the style-swapped test set


def plot_layer_sweep(sep: pd.DataFrame, chosen_layer: int, path: Path) -> Path:
    """Held-out AUROC per layer. With style subsets, one line per subset: if the
    style-swapped line collapses toward chance, the vector is mostly reading tone."""
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=180)
    styled = sorted(x for x in sep.split.unique() if x.startswith("holdout["))
    if styled:
        for split, color, marker, ls in zip(styled, (HONEST, SWAP, STEER), ("o", "s", "^"), ("-", "--", ":")):
            d = sep[sep.split == split].sort_values("layer")
            style = split[len("holdout["):-1]
            ax.plot(d.layer, d.auroc, color=color, lw=2, ls=ls, marker=marker, ms=4,
                    label=f"{STYLE_LABEL.get(style, style)}")
        title = "Does the vector track sycophancy, or just tone?"
    else:
        hold = sep[sep.split == "holdout"].sort_values("layer")
        ax.plot(hold.layer, hold.auroc, color=HONEST, lw=2, marker="o", ms=4, label="Held-out AUROC")
        ax.plot(hold.layer, hold.pairwise_accuracy, color=STEER, lw=2, marker="s", ms=4, label="Held-out pairwise accuracy")
        title = "Which layers separate honest from sycophantic replies?"
    last = int(sep.layer.max())
    ax.axhline(0.5, color=BASE, ls=":", lw=1)
    ax.text(last, 0.51, "chance", color=BASE, ha="right", va="bottom", fontsize=8)
    ax.axvline(chosen_layer, color=INK2, ls="--", lw=1)
    ax.text(chosen_layer + 0.2, 0.03, f"steering layer {chosen_layer}", color=INK2, fontsize=8, rotation=90, va="bottom")
    ax.set_ylim(0, 1.05)
    _style(ax, title, "Decoder layer (output)", "Held-out AUROC" if styled else "Score on held-out pairs")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_separation(art, layer: int, path: Path) -> Path:
    """Held-out honest vs sycophantic replies on the score scale, one panel per style subset."""
    from .extraction import auroc
    styles = np.array(art.styles or ["standard"] * len(art.pair_ids))
    hold = art.is_holdout.numpy()
    groups = sorted(set(styles[hold]))
    allh = art.score(layer, art.honest_proj[:, layer]).numpy()
    alls = art.score(layer, art.syc_proj[:, layer]).numpy()
    lo = float(min(allh[hold].min(), alls[hold].min(), -1.5))
    hi = float(max(allh[hold].max(), alls[hold].max(), 1.5))
    bins = np.linspace(lo, hi, 25)
    fig, axes = plt.subplots(1, len(groups), figsize=(8 if len(groups) == 1 else 12, 3.8), dpi=180,
                             squeeze=False, sharey=True)
    for ax, g in zip(axes[0], groups):
        m = hold & (styles == g)
        h, s = allh[m], alls[m]
        for vals, color, name in ((h, HONEST, "Honest"), (s, SYC, "Sycophantic")):
            ax.hist(vals, bins=bins, histtype="stepfilled", color=color, alpha=0.30, lw=0)
            ax.hist(vals, bins=bins, histtype="step", color=color, lw=2, label=f"{name} replies (n={len(vals)})")
        for x in (-1, 1):
            ax.axvline(x, color=BASE, ls=":", lw=1)
        name = STYLE_LABEL.get(g, g) if len(groups) > 1 else "Held-out"
        _style(ax, f"{name}: AUROC {auroc(s, h):.2f}", "Sycophancy score  (−1 typical honest … +1 typical sycophantic)",
               "Replies" if ax is axes[0][0] else "")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Held-out separation at layer {layer}", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


CONFIRM, FALSEPB = "#008300", "#6250d6"


def _rate_lines(ax, summary, subset, series, title, n_note):
    """Steered rates as lines with 95% bands; random-direction control as hollow markers."""
    st = summary[(summary.condition == "steered") & (summary.subset == subset)].sort_values("alpha")
    rd = summary[(summary.condition == "random") & (summary.subset == subset)].sort_values("alpha")
    if st.empty or st.n_labeled.sum() == 0:
        ax.text(0.5, 0.5, "No judge labels for this subset\n(run evaluate with a judge)", ha="center", va="center",
                transform=ax.transAxes, color=INK2)
        _style(ax, title, "Steering strength α", "% of prompts")
        return
    for label, color, marker in series:
        ax.fill_between(st.alpha, st[f"{label}_lo"] * 100, st[f"{label}_hi"] * 100, color=color, alpha=0.12, lw=0)
        ax.plot(st.alpha, st[f"{label}_rate"] * 100, color=color, lw=2, marker=marker, ms=6,
                label=label.replace("_", " ").capitalize())
        if not rd.empty:
            ax.plot(rd.alpha, rd[f"{label}_rate"] * 100, ls="none", marker=marker, ms=9, mfc="none", mec=color, mew=2)
    if not rd.empty:
        ax.plot([], [], ls="none", marker="o", ms=9, mfc="none", mec=INK2, mew=2, label="Random direction (control)")
    ax.axvline(0, color=BASE, ls=":", lw=1)
    ax.set_ylim(-3, 103)
    _style(ax, title, "Steering strength α  (− pushback … + agreement)", f"% of {n_note} prompts (95% CI)")
    ax.legend(frameon=False, fontsize=8, loc="best")


def plot_evaluation(summary: pd.DataFrame, path: Path, judge_name: str = "") -> Path:
    """Four panels sharing the α axis: the benefit, the over-correction risk, fluency, and the internal readout."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=180)
    (a, b), (c, d) = axes
    n_h = int(summary[summary.subset == "holdout"].n_labeled.max() or 0) if (summary.subset == "holdout").any() else 0
    n_u = int(summary[summary.subset == "user_correct"].n_labeled.max() or 0) if (summary.subset == "user_correct").any() else 0
    _rate_lines(a, summary, "holdout", (("pushback", HONEST, "o"), ("sycophantic", SYC, "s")),
                "1  User is wrong: does steering reduce sycophancy?", n_h)
    _rate_lines(b, summary, "user_correct", (("confirms", CONFIRM, "o"), ("false_pushback", FALSEPB, "s")),
                "2  User is right: does it start arguing anyway?", n_u)

    def metric(ax, col, title, ylabel, bands=False):
        sub = summary[summary.subset == "holdout"]
        st = sub[sub.condition == "steered"].sort_values("alpha")
        rd = sub[sub.condition == "random"].sort_values("alpha")
        if st.empty or col not in st or st[col].isna().all():
            ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes, color=INK2)
        else:
            err = 1.96 * st["mean_score_sem"].fillna(0) if col == "mean_score" else None
            ax.errorbar(st.alpha, st[col], yerr=err, color=STEER, lw=2, marker="o", ms=6, capsize=3,
                        label="Sycophancy direction")
            if not rd.empty:
                ax.plot(rd.alpha, rd[col], ls="none", marker="o", ms=9, mfc="none", mec=INK2, mew=2,
                        label="Random direction (control)")
            ax.legend(frameon=False, fontsize=8)
        if bands:
            ax.axhspan(-1.2, -0.8, color=HONEST, alpha=0.08, lw=0)
            ax.axhspan(0.8, 1.2, color=SYC, alpha=0.08, lw=0)
            ax.text(0.01, 1.0, "typical sycophantic reply", transform=ax.get_yaxis_transform(), color=SYC, fontsize=8, va="bottom")
            ax.text(0.01, -1.0, "typical honest reply", transform=ax.get_yaxis_transform(), color=HONEST, fontsize=8, va="top")
            ax.axhline(0, color=BASE, ls=":", lw=1)
        ax.axvline(0, color=BASE, ls=":", lw=1)
        _style(ax, title, "Steering strength α", ylabel)

    metric(c, "perplexity_median", "3  Is the text still fluent?", "Median perplexity (lower = more natural)")
    metric(d, "mean_score", "4  Internals: readout-layer sycophancy score", "Mean per-token score (±95% CI)", bands=True)
    if judge_name:
        fig.text(0.99, 0.005, f"judge: {judge_name}", ha="right", fontsize=7, color=BASE)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_dose_response(summary: pd.DataFrame, path: Path, judge_name: str = "") -> Path:  # backwards-compatible name
    return plot_evaluation(summary, path, judge_name)


# app visuals
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
