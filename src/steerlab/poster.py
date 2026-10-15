"""A 24" x 36" portrait poster built straight from a run's numbers and figures (vector PDF).

Everything is laid out top to bottom in two columns with a simple cursor, so the poster
regenerates cleanly whenever the results change. Edit the text constants below to adjust wording.
"""
from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from .report import (FUTURE, LIMITATIONS, RESEARCH_QUESTION, judge_finding, num, pct, steering_findings,  # noqa: E402
                     tone_finding)

W, H = 24.0, 36.0          # inches
MARGIN, GUTTER = 0.9, 0.8
COL_W = (W - 2 * MARGIN - GUTTER) / 2
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
ACCENT, ACCENT_DARK, SURFACE, CARD = "#2a78d6", "#184f95", "#fcfcfb", "#f0efec"
HONEST, SYC, SWAP = "#2a78d6", "#e34948", "#6250d6"

PROBLEM = ("Chatbots like to agree with you, even when you're wrong. Ask \"2 + 2 = 5, right?\" and a sycophantic "
           "model says \"Yes, great thinking!\". That habit is dangerous when people ask about medicine, money "
           "or safety, and it gets worse under pressure (\"as a doctor…\", \"please just say yes\").")
IDEA = ("Inside the model, the difference between an honest reply and a suck-up reply points in one direction: a "
        "\"sycophancy arrow\". Adding or subtracting it while the model writes should turn the habit up or down. "
        "But does the arrow point at agreement, or just at excited, exclamation-heavy tone?")
DATA = [
    "50 training pairs across 8 domains (finance, health, code, science…): one honest and one sycophantic reply each.",
    "100 held-out pairs never used for training: 60 in the training style, 40 style-swapped (calm sycophant, "
    "cheerful honest reply) to separate stance from tone.",
    "40 user-is-right prompts: disputing these is an error, so they catch over-correction.",
    "Controls: a random direction steered at the same strength, and perplexity / repetition of every reply.",
]
BUILDS_ON = ("Contrastive activation addition steers behaviours, sycophancy included, by adding a mean activation "
             "difference (Rimsky et al., 2024; Turner et al., 2023; Zou et al., 2023). Sycophancy itself is well "
             "documented (Perez et al., 2022; Sharma et al., 2024). New here: a tone-controlled audit of the "
             "direction, plus controls for over-correction, random directions and fluency.")
REFERENCES = [
    "Rimsky et al. (2024). Steering Llama 2 via Contrastive Activation Addition. ACL. arXiv:2312.06681",
    "Turner et al. (2023). Steering Language Models With Activation Engineering. arXiv:2308.10248",
    "Zou et al. (2023). Representation Engineering. arXiv:2310.01405",
    "Sharma et al. (2024). Towards Understanding Sycophancy in Language Models. ICLR. arXiv:2310.13548",
    "Perez et al. (2022). Discovering LM Behaviors with Model-Written Evaluations. arXiv:2212.09251",
    "Wei et al. (2023). Simple synthetic data reduces sycophancy in LLMs. arXiv:2308.03958",
    "Arditi et al. (2024). Refusal in LMs Is Mediated by a Single Direction. NeurIPS. arXiv:2406.11717",
]


def _plain(s: str) -> str:
    return re.sub(r"\*\*|`", "", s)


class Column:
    def __init__(self, fig, x0: float, top: float, scale: float = 1.0):
        self.fig, self.x0, self.y = fig, x0, top  # y = distance from the top edge, in inches
        self.k = scale  # shrinks body text when content would overflow

    def _fy(self, y_in: float) -> float:
        return 1 - y_in / H

    def gap(self, inches: float):
        self.y += inches
        return self

    def text(self, s: str, size: float = 25, color: str = INK, weight: str = "normal", indent: float = 0.0,
             bullet: str | None = None, line: float = 1.38):
        size *= self.k
        width = COL_W - indent - (0.45 if bullet else 0)
        chars = max(20, int(width * 72 / (size * 0.5)))
        lines = textwrap.wrap(_plain(s), chars) or [""]
        lh = size * line / 72
        x = self.x0 + indent + (0.45 if bullet else 0)
        if bullet:
            self.fig.text((self.x0 + indent) / W, self._fy(self.y), bullet, fontsize=size, color=ACCENT,
                          va="top", weight="bold")
        for ln in lines:
            self.fig.text(x / W, self._fy(self.y), ln, fontsize=size, color=color, weight=weight, va="top")
            self.y += lh
        return self

    def heading(self, s: str, number: str | None = None):
        self.gap(0.25)
        label = f"{number}  {s}" if number else s
        self.fig.text(self.x0 / W, self._fy(self.y), label, fontsize=36, weight="bold", color=ACCENT_DARK, va="top")
        self.y += 0.66
        self.fig.add_artist(plt.Line2D([self.x0 / W, (self.x0 + COL_W) / W], [self._fy(self.y)] * 2,
                                       color=ACCENT, lw=3, transform=self.fig.transFigure))
        self.y += 0.3
        return self

    def callout(self, s: str, size: float = 29):
        size *= self.k
        chars = max(20, int((COL_W - 0.9) * 72 / (size * 0.5)))
        lines = textwrap.wrap(_plain(s), chars)
        h = len(lines) * size * 1.38 / 72 + 0.7
        box = FancyBboxPatch((self.x0 / W, self._fy(self.y + h)), COL_W / W, h / H,
                             boxstyle="round,pad=0,rounding_size=0.006", transform=self.fig.transFigure,
                             facecolor="#e8f1fc", edgecolor=ACCENT, lw=3)
        self.fig.add_artist(box)
        self.y += 0.35
        for ln in lines:
            self.fig.text((self.x0 + 0.45) / W, self._fy(self.y), ln, fontsize=size, color=INK, style="italic", va="top")
            self.y += size * 1.38 / 72
        self.y += 0.35
        return self

    def image(self, path: Path | None, caption: str | None = None):
        if path is None or not Path(path).exists():
            self.text("(figure not available yet: run steerlab evaluate, then steerlab report)", 22, MUTED)
            return self
        img = mpimg.imread(str(path))
        h = COL_W * img.shape[0] / img.shape[1]
        ax = self.fig.add_axes([self.x0 / W, self._fy(self.y + h), COL_W / W, h / H])
        ax.imshow(img, interpolation="lanczos")
        ax.axis("off")
        self.y += h + 0.15
        if caption:
            self.text(caption, 20, INK2)
        return self

    def stats(self, tiles: list[tuple[str, str, str]]):
        """Big-number tiles: (value, label, colour)."""
        n = len(tiles)
        tw = (COL_W - 0.3 * (n - 1)) / n
        th = 2.5
        for i, (value, label, color) in enumerate(tiles):
            x = self.x0 + i * (tw + 0.3)
            box = FancyBboxPatch((x / W, self._fy(self.y + th)), tw / W, th / H,
                                 boxstyle="round,pad=0,rounding_size=0.006", transform=self.fig.transFigure,
                                 facecolor=CARD, edgecolor="none")
            self.fig.add_artist(box)
            self.fig.text((x + tw / 2) / W, self._fy(self.y + 0.3), value, fontsize=50, weight="bold", color=color,
                          ha="center", va="top")
            for j, ln in enumerate(textwrap.wrap(label, max(12, int(tw * 72 / (19 * 0.5))))[:3]):
                self.fig.text((x + tw / 2) / W, self._fy(self.y + 1.35 + j * 0.36), ln, fontsize=19, color=INK2,
                              ha="center", va="top")
        self.y += th + 0.25
        return self

    def method_diagram(self):
        h = 3.9
        ax = self.fig.add_axes([self.x0 / W, self._fy(self.y + h), COL_W / W, h / H])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4.6)
        ax.axis("off")
        boxes = [
            (0.0, 3.0, 3.0, 1.4, "Honest vs suck-up\nreplies (50 pairs)", CARD, INK),
            (3.5, 3.0, 3.0, 1.4, "Average difference\ninside the model", CARD, INK),
            (7.0, 3.0, 3.0, 1.4, "Sycophancy\ndirection v", "#e8f1fc", ACCENT_DARK),
            (0.0, 0.4, 3.0, 1.6, "New prompt", CARD, INK),
            (3.5, 0.4, 3.0, 1.6, "Steer: add α·v\nwhile it writes", "#fdecec", "#a3282a"),
            (7.0, 0.4, 3.0, 1.6, "Reply + judge +\nper-word readout", CARD, INK),
        ]
        for x, y, w, hh, label, face, ink in boxes:
            ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.02,rounding_size=0.15",
                                        facecolor=face, edgecolor="none"))
            ax.text(x + w / 2, y + hh / 2, label, ha="center", va="center", fontsize=21, color=ink, weight="bold")
        arrows = [((3.0, 3.7), (3.5, 3.7)), ((6.5, 3.7), (7.0, 3.7)), ((3.0, 1.2), (3.5, 1.2)),
                  ((6.5, 1.2), (7.0, 1.2)), ((8.5, 3.0), (5.6, 2.0))]
        for (x0, y0), (x1, y1) in arrows:
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=28, color=MUTED, lw=2.5))
        self.y += h + 0.2
        return self


def build_poster(rd, figs: dict, out: Path, **opts) -> Path:
    """Render, shrinking body text in small steps until everything fits above the footer."""
    for scale in (1.0, 0.96, 0.92, 0.88, 0.84, 0.8):
        fig, overflow, spare = _render(rd, figs, scale=scale, **opts)
        if overflow <= 0 or scale == 0.8:
            break
        plt.close(fig)
    if overflow > 0:
        print(f"[!] Poster content still runs {overflow:.1f} in past the footer at 80% text size; shorten some text.")
    else:
        print(f"[+] Poster layout fits at {scale:.0%} text size ({spare:.1f} in to spare)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".png"), dpi=40, facecolor=SURFACE)  # small preview
    plt.close(fig)
    return out


def _render(rd, figs: dict, scale: float = 1.0, title: str | None = None, author: str = "", affiliation: str = "",
            url: str = "", acknowledge_ai: bool = True):
    ext, h = rd.extraction, rd.head
    verdict, tone_text = tone_finding(ext)
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(SURFACE)
    plt.rcParams["font.family"] = "sans-serif"

    # ---- header band
    band = 4.2
    fig.add_artist(FancyBboxPatch((0, 1 - band / H), 1, band / H, boxstyle="square,pad=0",
                                  transform=fig.transFigure, facecolor=ACCENT_DARK, edgecolor="none"))
    title = title or "Sycophancy, or Just Enthusiasm?"
    tagline = "Auditing a steering vector inside a language model"
    fig.text(MARGIN / W, 1 - 0.6 / H, title, fontsize=80, weight="bold", color="white", va="top")
    fig.text(MARGIN / W, 1 - 2.0 / H, tagline, fontsize=40, color="white", va="top")
    sub = " · ".join(x for x in (author, affiliation) if x)
    if sub:
        fig.text(MARGIN / W, 1 - 3.05 / H, sub, fontsize=30, color="#cde2fb", va="top")
    if url:
        fig.text((W - MARGIN) / W, 1 - 3.05 / H, url, fontsize=28, color="#cde2fb", va="top", ha="right")

    left = Column(fig, MARGIN, band + 0.6, scale)
    right = Column(fig, MARGIN + COL_W + GUTTER, band + 0.6, scale)

    # ---- left column
    left.heading("The problem").text(PROBLEM, 25).gap(0.2).text(IDEA, 25)
    left.heading("Research question").callout(RESEARCH_QUESTION, 27)
    left.heading("Builds on").text(BUILDS_ON, 22, INK2)
    left.heading("Method").method_diagram().text(
        f"Model: {ext['model_id']}. v = mean(sycophantic − honest) activations at layer {ext['steering_layer']}; "
        f"α sets how hard we push. Scores are read at a later layer, so they show the effect spreading "
        f"through the network, not the added vector echoed back.", 21, INK2)
    left.heading("Data and controls")
    for d in DATA:
        left.text(d, 21, bullet="•").gap(0.06)
    left.heading("Future work")
    for d in FUTURE[:2]:
        left.text(d, 21, bullet="•").gap(0.06)

    # ---- right column
    right.heading("Finding 1 · stance or tone?")
    std = ext.get("steering_layer_auroc_standard", ext.get("steering_layer_holdout_auroc"))
    swp = ext.get("steering_layer_auroc_style_swapped")
    tiles = [(num(std), "held-out AUROC, training style", HONEST)]
    if swp is not None:
        tiles.append((num(swp), "held-out AUROC, style swapped (0.5 = chance)", SWAP))
    if h and h["has_labels"]:
        tiles.append((f"{pct(h['sycophantic_base'])}→{pct(h['sycophantic_low'])}",
                      f"sycophantic replies, α = 0 → {h['alpha_low']:+g}", SYC))
    right.stats(tiles)
    right.text(f"The vector {verdict}.", 28, ACCENT_DARK, weight="bold").gap(0.05)
    right.text(tone_text, 22)
    right.gap(0.15).image(figs.get("separation"))
    right.heading("Finding 2 · what steering does")
    if h:
        findings = steering_findings(h, rd.summary)
        keep = [t for t in findings if not t.startswith("By style")][:3]
        for t in keep:
            right.text(t, 21, bullet="•").gap(0.05)
        right.gap(0.1).image(figs.get("evaluation_overview"))
    else:
        right.text("Run steerlab evaluate, then steerlab report, to fill in this section.", 24, MUTED)
    right.text(judge_finding(rd.agreement, rd.judge_name), 19, INK2)
    right.heading("Limitations")
    for d in LIMITATIONS[:3]:
        right.text(d.format(model=ext["model_id"]), 20, bullet="•").gap(0.05)
    right.text("More limitations and future work: results/RESULTS.md in the repository.", 18, MUTED)

    # ---- footer (built upward from the bottom edge)
    tail = []
    if acknowledge_ai:
        tail.append("Developed with assistance from Claude (Anthropic).")
    tail.append("Every number on this poster is generated from the run by steerlab report.")
    y_tail = H - 0.5
    fig.text(MARGIN / W, 1 - y_tail / H, "  ".join(tail), fontsize=15, color=MUTED, va="top")
    lines = textwrap.wrap("References: " + "   ".join(REFERENCES), 235)
    yref = y_tail - 0.3 * len(lines) - 0.12
    for i, ln in enumerate(lines):
        fig.text(MARGIN / W, 1 - (yref + i * 0.3) / H, ln, fontsize=14, color=MUTED, va="top")
    fig.add_artist(plt.Line2D([MARGIN / W, (W - MARGIN) / W], [1 - (yref - 0.2) / H] * 2, color="#e1e0d9", lw=2,
                              transform=fig.transFigure))

    lowest = max(left.y, right.y)
    return fig, lowest - (yref - 0.3), (yref - 0.3) - lowest
