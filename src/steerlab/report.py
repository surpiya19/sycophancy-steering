"""Turn an extraction + evaluation run into shareable results: results/RESULTS.md, figures, and a poster PDF.

Every number and every sentence of interpretation is computed from the run, so the write-up
cannot drift from the data. Interpretations are deliberately hedged: they describe what the
numbers show with their 95% intervals, not significance claims the sample size can't support.
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .evaluation import headline

RESEARCH_QUESTION = ("Does a contrastive steering vector for sycophancy capture agreement with false or risky "
                     "premises, or just the surface tone of flattery, and can steering against it reduce "
                     "sycophancy without making the model contrarian or incoherent?")


@dataclass
class RunData:
    extraction: dict
    separation: pd.DataFrame
    summary: pd.DataFrame | None
    head: dict | None
    judge_name: str
    agreement: dict | None
    config: dict
    artifact_dir: Path
    run_dir: Path | None


def load_run(artifact_dir: Path, run_dir: Path | None) -> RunData:
    import yaml
    ext = json.loads((artifact_dir / "extraction_summary.json").read_text())
    sep = pd.read_csv(artifact_dir / "layer_separation.csv")
    summary = head = agreement = None
    judge_name = "none"
    cfg_path = artifact_dir / "config.yaml"
    if run_dir is not None:
        summary = pd.read_csv(run_dir / "summary.csv")
        head = headline(summary)
        if (run_dir / "judge.txt").exists():
            judge_name = (run_dir / "judge.txt").read_text().strip()
        if (run_dir / "judge_agreement.json").exists():
            agreement = json.loads((run_dir / "judge_agreement.json").read_text())
        if (run_dir / "config.yaml").exists():
            cfg_path = run_dir / "config.yaml"
    config = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    return RunData(ext, sep, summary, head, judge_name, agreement, config, artifact_dir, run_dir)


# ----------------------------------------------------------------------------- formatting helpers
def pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.0f}%"


def num(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _ci(summary, condition, subset, alpha, label) -> str:
    r = summary[(summary.condition == condition) & (summary.subset == subset) & ((summary.alpha - alpha).abs() < 1e-9)]
    if r.empty or r[f"{label}_rate"].isna().all():
        return "n/a"
    r = r.iloc[0]
    return f"{r[f'{label}_rate'] * 100:.0f}% [{r[f'{label}_lo'] * 100:.0f}–{r[f'{label}_hi'] * 100:.0f}]"


def _change(a: float, b: float) -> str:
    if any(isinstance(x, float) and math.isnan(x) for x in (a, b)):
        return "could not be measured"
    d = (b - a) * 100
    if abs(d) < 5:
        return f"stayed about the same ({pct(a)} → {pct(b)})"
    return f"{'fell' if d < 0 else 'rose'} from {pct(a)} to {pct(b)}"


# ----------------------------------------------------------------------------- interpretations
def tone_finding(ext: dict) -> tuple[str, str]:
    std = ext.get("steering_layer_auroc_standard", ext.get("steering_layer_holdout_auroc"))
    swp = ext.get("steering_layer_auroc_style_swapped")
    if swp is None:
        return ("not tested", f"Held-out AUROC at the steering layer is {num(std)}. No style-swapped pairs were "
                              f"evaluated, so tone and stance can't be told apart.")
    if std >= 0.8 and swp >= 0.8:
        verdict = "survives the style swap"
        text = (f"The vector separates held-out honest from sycophantic replies whether the sycophantic reply "
                f"is gushing (AUROC {num(std)}) or calm (AUROC {num(swp)}). That is evidence it encodes "
                f"agreement with the premise, not only enthusiastic tone.")
    elif std >= 0.7 and swp < 0.6:
        verdict = "mostly tracks tone"
        text = (f"The vector separates replies written in the training style (AUROC {num(std)}), but not when "
                f"the tone is flipped (AUROC {num(swp)}; 0.5 is chance"
                f"{', below 0.5 means it rates excited honest replies as more sycophantic' if swp < 0.5 else ''}). "
                f"It is mostly detecting enthusiastic tone rather than agreement with a false premise.")
    else:
        verdict = "partly tracks tone"
        text = (f"Separation drops from AUROC {num(std)} on standard pairs to {num(swp)} on style-swapped pairs, "
                f"so tone explains part, but not all, of what the vector reads.")
    return verdict, text


def steering_findings(h: dict, summary: pd.DataFrame) -> list[str]:
    lo = h["alpha_low"]
    out = []
    if h["has_labels"]:
        out.append(f"**User is wrong ({h['n_holdout_eval']} held-out prompts).** At α = {lo:+g}, sycophantic replies "
                   f"{_change(h['sycophantic_base'], h['sycophantic_low'])} and pushback "
                   f"{_change(h['pushback_base'], h['pushback_low'])}. Steering the same strength along a random "
                   f"direction gave {pct(h['random_sycophantic_low'])} sycophantic and "
                   f"{pct(h['random_pushback_low'])} pushback.")
        if not math.isnan(h.get("sycophantic_standard_low", float("nan"))):
            out.append(f"By style at α = {lo:+g}: sycophantic replies were {pct(h['sycophantic_standard_low'])} on "
                       f"standard prompts and {pct(h['sycophantic_swapped_low'])} on style-swapped prompts "
                       f"(baseline {pct(h['sycophantic_standard_base'])} and {pct(h['sycophantic_swapped_base'])}).")
        if h["n_user_correct_eval"]:
            out.append(f"**User is right ({h['n_user_correct_eval']} prompts).** False pushback, where the model "
                       f"disputes a correct claim, {_change(h['false_pushback_base'], h['false_pushback_low'])} at "
                       f"α = {lo:+g} (random direction: {pct(h['random_false_pushback_low'])}). Confirmations "
                       f"{_change(h['confirms_base'], h['confirms_low'])}.")
    else:
        out.append("No judge labels in this run, so behavioural rates are not reported. Re-run `steerlab evaluate` "
                   "with a judge to fill them in.")
    if not math.isnan(h["perplexity_base"]):
        ratio = h["perplexity_low"] / h["perplexity_base"] if h["perplexity_base"] else float("nan")
        out.append(f"**Fluency.** Median perplexity of the replies (under the unsteered model) was "
                   f"{num(h['perplexity_base'], 1)} at baseline and {num(h['perplexity_low'], 1)} at α = {lo:+g} "
                   f"({num(ratio, 1)}×); the random direction gave {num(h['random_perplexity_low'], 1)}."
                   + (" A large rise means the text is degrading, so behavioural changes at that strength should be "
                      "read with caution." if ratio == ratio and ratio > 1.5 else ""))
    out.append(f"**Internal readout.** The mean sycophancy score moved from {num(h['score_base'])} to "
               f"{num(h['score_low'])} along the extracted direction, versus {num(h['random_score_low'])} along the "
               f"random one (−1 = typical honest reply, +1 = typical sycophantic reply).")
    return out


def judge_finding(agreement: dict | None, judge_name: str) -> str:
    if agreement is None:
        return (f"Judge: `{judge_name}`. Not yet validated against human labels; run `steerlab label` to label a "
                f"sample yourself and measure agreement.")
    k = agreement["cohens_kappa"]
    word = "substantial" if k >= 0.6 else "moderate" if k >= 0.4 else "weak"
    return (f"Judge: `{judge_name}`. On {agreement['n']} replies labelled blind by the author, it agreed "
            f"{pct(agreement['agreement'])} of the time (Cohen's κ = {num(k)}, {word} agreement).")


# ----------------------------------------------------------------------------- RESULTS.md
LIMITATIONS = [
    "In the 50 training pairs, tone and label line up perfectly: every sycophantic reply has an exclamation mark and "
    "no honest reply does. The style-swapped test measures this confound; it does not remove it.",
    "One model and one steering layer ({model}); effects may not transfer to larger or differently trained models.",
    "All 190 conversations (training, held-out, user-is-right) were written in one voice, with AI assistance; "
    "they are varied but not a naturalistic sample of how people talk to chatbots.",
    "Behaviour is graded by an LLM judge; its labels are only as good as its agreement with humans.",
    "Small evaluation sets give wide 95% intervals; differences inside overlapping intervals are not evidence.",
    "The pressure-tactic detector is a list of regular expressions and has not been evaluated.",
]
FUTURE = [
    "Rebuild the training pairs so tone is balanced against the label, and check whether the style-swap gap closes.",
    "Repeat across model sizes and families, and across steering layers chosen from the layer sweep.",
    "Compare with non-steering fixes: an honesty system prompt and fine-tuning on synthetic data (Wei et al., 2023).",
    "Test on naturalistic multi-turn conversations where users push back over several turns.",
    "A user study: do people trust, or notice, a steered assistant's pushback?",
]


def write_results_md(rd: RunData, out_dir: Path, figs: dict[str, str]) -> Path:
    ext, h = rd.extraction, rd.head
    verdict, tone_text = tone_finding(ext)
    ev = rd.config.get("eval", {}) if rd.config else {}
    lines = [
        "# Results",
        "",
        f"Generated by `steerlab report` from `{rd.run_dir or rd.artifact_dir}`. Every number below comes from that run.",
        "",
        "## Research question",
        "",
        f"> {RESEARCH_QUESTION}",
        "",
        "## Setup",
        "",
        "| | |",
        "|---|---|",
        f"| Model | `{ext['model_id']}` |",
        f"| Steering layer | {ext['steering_layer']} (best held-out layer by AUROC: {ext['best_layer_by_holdout_auroc']}) |",
        f"| Training pairs | {ext['n_train']} |",
        f"| Held-out pairs | {ext['n_holdout']} ({ext.get('n_holdout_standard', '–')} standard, "
        f"{ext.get('n_holdout_style_swapped', '–')} style-swapped) |",
    ]
    if h:
        lines += [
            f"| Evaluation prompts | {h['n_holdout_eval']} user-is-wrong, {h['n_user_correct_eval']} user-is-right |",
            f"| Steering strengths | α ∈ {ev.get('alphas', '?')}; random-direction control at {ev.get('random_control_alphas', '?')} |",
            f"| Judge | `{rd.judge_name}` |",
        ]
    lines += [
        "",
        f"## Finding 1: the vector {verdict}",
        "",
        tone_text,
        "",
        f"![Layer sweep]({figs['layer_sweep']})",
        "",
        f"![Held-out separation]({figs['separation']})",
        "",
    ]
    if h and rd.summary is not None:
        lo = h["alpha_low"]
        s = rd.summary
        lines += ["## Finding 2: what steering does to behaviour", ""]
        lines += [f"- {t}" for t in steering_findings(h, s)]
        lines += ["", f"![Evaluation overview]({figs['evaluation_overview']})", ""]
        if h["has_labels"]:
            lines += [
                "| | baseline (α = 0) | steered (α = %+g) | random direction (α = %+g) |" % (lo, lo),
                "|---|---|---|---|",
                f"| Sycophantic, user wrong | {_ci(s, 'steered', 'holdout', 0.0, 'sycophantic')} | "
                f"{_ci(s, 'steered', 'holdout', lo, 'sycophantic')} | {_ci(s, 'random', 'holdout', lo, 'sycophantic')} |",
                f"| Pushback, user wrong | {_ci(s, 'steered', 'holdout', 0.0, 'pushback')} | "
                f"{_ci(s, 'steered', 'holdout', lo, 'pushback')} | {_ci(s, 'random', 'holdout', lo, 'pushback')} |",
                f"| False pushback, user right | {_ci(s, 'steered', 'user_correct', 0.0, 'false_pushback')} | "
                f"{_ci(s, 'steered', 'user_correct', lo, 'false_pushback')} | "
                f"{_ci(s, 'random', 'user_correct', lo, 'false_pushback')} |",
                "",
                "Percentages with 95% Wilson intervals.",
                "",
            ]
    lines += ["## Judge validation", "", judge_finding(rd.agreement, rd.judge_name), ""]
    lines += ["## Limitations", ""] + [f"- {x.format(model=ext['model_id'])}" for x in LIMITATIONS] + [""]
    lines += ["## Future work", ""] + [f"- {x}" for x in FUTURE] + [""]
    path = out_dir / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_report(artifact_dir: Path, run_dir: Path | None, out_dir: Path, poster: bool = True,
                 poster_opts: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    rd = load_run(artifact_dir, run_dir)
    figs = {}
    for name, src in (("layer_sweep", artifact_dir / "layer_sweep.png"), ("separation", artifact_dir / "separation.png"),
                      ("evaluation_overview", run_dir / "evaluation_overview.png" if run_dir else None)):
        if src is not None and src.exists():
            shutil.copy(src, fig_dir / f"{name}.png")
            figs[name] = f"figures/{name}.png"
    md = write_results_md(rd, out_dir, figs)
    (out_dir / "headline.json").write_text(json.dumps({"extraction": rd.extraction, "evaluation": rd.head,
                                                       "judge": rd.judge_name, "judge_agreement": rd.agreement},
                                                      indent=2, default=float))
    result = {"results_md": md, "figures": {k: out_dir / v for k, v in figs.items()}}
    if poster:
        from .poster import build_poster
        result["poster"] = build_poster(rd, {k: out_dir / v for k, v in figs.items()}, out_dir / "poster.pdf",
                                        **(poster_opts or {}))
    return result
