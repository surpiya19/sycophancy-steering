"""Alpha sweep with controls: does steering change behaviour, and only in the way we want?

For every steering strength alpha we generate replies to three kinds of prompt:

  holdout       the user is WRONG (100 never-trained pairs, split into standard and style_swapped)
                -> we want less sycophancy and more pushback as alpha goes negative
  user_correct  the user is RIGHT -> pushing back here is an error (false pushback / over-correction)
  custom        your own prompts, judged with the rubric their `premise` names

and run two conditions:

  steered  along the extracted sycophancy direction
  random   along a random unit direction, scaled exactly the same: separates "this direction does
           something specific" from "any push of this size changes the output"

Each reply also gets fluency metrics (perplexity under the unsteered model, distinct-2), so
broken text cannot pass as pushback.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import EvalPrompt, Pair, load_prompts
from .extraction import SteeringArtifacts
from .generation import SteerSpec, generate
from .judges import ALL_LABELS, Judge
from .metrics import distinct_2, perplexity, reply_nll, tone_markers
from .modeling import LoadedModel


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; sensible even at 0/n and n/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def balanced_subset(prompts: list[EvalPrompt], limit: int | None, key: str = "style") -> list[EvalPrompt]:
    """First `limit` prompts, taken round-robin across `key` groups so each group stays represented."""
    if not limit or limit >= len(prompts):
        return prompts
    groups: dict[str, list[EvalPrompt]] = {}
    for p in prompts:
        groups.setdefault(getattr(p, key), []).append(p)
    out, queues = [], [list(v) for _, v in sorted(groups.items())]
    while len(out) < limit and any(queues):
        for q in queues:
            if q and len(out) < limit:
                out.append(q.pop(0))
    return out


def build_eval_prompts(cfg, holdout: list[Pair], extra_file: str | None = None) -> list[EvalPrompt]:
    prompts: list[EvalPrompt] = []
    if cfg.eval.include_holdout:
        held = [EvalPrompt(p.id, p.prompt, p.domain, "holdout", p.style, "false") for p in holdout]
        prompts += balanced_subset(held, cfg.eval.holdout_limit)
    if cfg.eval.user_correct_file:
        uc = [EvalPrompt(p.id, p.prompt, p.domain, "user_correct", "", "true")
              for p in load_prompts(cfg.resolve_path(cfg.eval.user_correct_file))]
        prompts += balanced_subset(uc, cfg.eval.user_correct_limit, key="domain")
    path = extra_file or cfg.eval.prompts_file
    if path:
        prompts += load_prompts(cfg.resolve_path(path))
    if not prompts:
        raise ValueError("No evaluation prompts: enable eval.include_holdout or pass a prompts file")
    return prompts


def random_direction(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def run_alpha_sweep(lm: LoadedModel, art: SteeringArtifacts, cfg, prompts: list[EvalPrompt], judge: Judge,
                    alphas: list[float] | None = None, random_alphas: list[float] | None = None) -> pd.DataFrame:
    alphas = alphas if alphas is not None else cfg.eval.alphas
    random_alphas = random_alphas if random_alphas is not None else cfg.eval.random_control_alphas
    rand_vec = random_direction(art.hidden_size, cfg.eval.random_seed)
    plan = [("steered", a, None) for a in alphas] + [("random", a, rand_vec) for a in random_alphas if a != 0]
    texts = [p.prompt for p in prompts]
    rows = []
    for i, (condition, alpha, direction) in enumerate(plan, 1):
        spec = SteerSpec(cfg.steering.layer, cfg.steering.readout_layer, alpha,
                         cfg.steering.scale_fraction, cfg.steering.positions, direction)
        gens = generate(lm, art, texts, spec, max_new_tokens=cfg.eval.max_new_tokens, greedy=True,
                        batch_size=cfg.generation.batch_size)
        labels = judge.judge_many([(g.prompt, g.text, p.premise) for p, g in zip(prompts, gens)])
        nlls = (reply_nll(lm, texts, [g.text for g in gens], cfg.generation.batch_size)
                if cfg.eval.fluency else [float("nan")] * len(gens))
        for p, g, label, nll in zip(prompts, gens, labels, nlls):
            rows.append({"condition": condition, "alpha": alpha, "prompt_id": p.id, "source": p.source,
                         "style": p.style, "premise": p.premise, "domain": p.domain,
                         "prompt": p.prompt, "response": g.text, "label": label,
                         "mean_score": g.mean_score, "first_token_score": g.scores[0] if g.scores else float("nan"),
                         "n_tokens": len(g.tokens), "nll": nll, "perplexity": perplexity(nll),
                         "distinct_2": distinct_2(g.text), **tone_markers(g.text)})
        df = pd.DataFrame(rows[-len(gens):])
        msg = f"[*] {i}/{len(plan)} {condition:8s} α={alpha:+.2f}  score={df.mean_score.mean():+.2f}"
        if cfg.eval.fluency:
            msg += f"  ppl={df.perplexity.median():.1f}"
        held = df[df.source == "holdout"]
        if held.label.notna().any():
            msg += f"  sycophantic {(held.label == 'sycophantic').sum()}/{held.label.notna().sum()}"
        uc = df[df.source == "user_correct"]
        if uc.label.notna().any():
            msg += f"  false pushback {(uc.label == 'false_pushback').sum()}/{uc.label.notna().sum()}"
        print(msg)
    return pd.DataFrame(rows)


def _subsets(responses: pd.DataFrame):
    held = responses[responses.source == "holdout"]
    if len(held):
        yield "holdout", held
        if held["style"].nunique() > 1:
            for s, g in held.groupby("style"):
                yield f"holdout[{s}]", g
    for source in ("user_correct", "custom"):
        g = responses[responses.source == source]
        if len(g):
            yield source, g


def summarize(responses: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, subset, alpha): label rates with 95% Wilson intervals, internal score, fluency."""
    out = []
    if "condition" not in responses:
        responses = responses.assign(condition="steered")
    for subset, df in _subsets(responses):
        for (condition, alpha), g in df.groupby(["condition", "alpha"]):
            labeled = g[g["label"].notna()]
            n = len(labeled)
            row = {"condition": condition, "subset": subset, "alpha": alpha, "n_prompts": len(g), "n_labeled": n,
                   "mean_score": g["mean_score"].mean(),
                   "mean_score_sem": g["mean_score"].std(ddof=1) / math.sqrt(len(g)) if len(g) > 1 else float("nan")}
            if "perplexity" in g:
                row["perplexity_median"] = g["perplexity"].median()
                row["distinct_2_mean"] = g["distinct_2"].mean()
            for col in ("exclamations", "emoji"):
                if col in g:
                    row[f"{col}_mean"] = pd.to_numeric(g[col], errors="coerce").mean()
            for label in ALL_LABELS:
                k = int((labeled["label"] == label).sum())
                lo, hi = wilson(k, n)
                row[f"{label}_rate"] = k / n if n else float("nan")
                row[f"{label}_lo"], row[f"{label}_hi"] = lo, hi
            out.append(row)
    return pd.DataFrame(out)


def relabel(responses: pd.DataFrame, judge: Judge) -> pd.DataFrame:
    """Re-grade saved replies with another judge, without generating anything again."""
    out = responses.copy()
    items = list(zip(out["prompt"], out["response"].fillna(""), out["premise"]))
    labels = []
    for i, item in enumerate(items, 1):
        labels.append(judge.judge(*item))
        if i % 25 == 0 or i == len(items):
            print(f"\r[*] judged {i}/{len(items)}", end="", flush=True)
    print()
    out["label"] = labels
    if "response" in out:  # older runs have no tone columns
        tm = out["response"].fillna("").map(tone_markers)
        out["exclamations"] = [t["exclamations"] for t in tm]
        out["emoji"] = [t["emoji"] for t in tm]
    return out


def read_responses(path: str | Path) -> pd.DataFrame:
    """Load responses.csv with text columns kept as text (pandas would turn "true"/"false" into booleans)."""
    df = pd.read_csv(path, dtype={"premise": str, "style": str, "label": str, "response": str, "prompt": str},
                     keep_default_na=False, na_values={"label": [""], "mean_score": [""], "nll": [""],
                                                       "perplexity": [""], "distinct_2": [""], "first_token_score": [""]})
    df["premise"] = df["premise"].str.lower().replace({"": "false"})
    for col in ("alpha", "exclamations", "emoji", "n_tokens", "distinct_2"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label"] = df["label"].where(df["label"].notna() & (df["label"] != ""), None)
    return df


def save_eval(out_dir: Path, responses: pd.DataFrame, summary: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    responses.to_csv(out_dir / "responses.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)


def headline(summary: pd.DataFrame, alpha_low: float | None = None) -> dict:
    """Key numbers for the report: baseline (alpha=0) vs the most negative steered alpha,
    and the random control at the same strength when it exists."""
    def get(condition, subset, alpha, col):
        r = summary[(summary.condition == condition) & (summary.subset == subset) & np.isclose(summary.alpha, alpha)]
        return float(r[col].iloc[0]) if len(r) and col in r else float("nan")

    steered = summary[summary.condition == "steered"]
    lo = alpha_low if alpha_low is not None else float(steered.alpha.min())
    hi = float(steered.alpha.max())
    out = {"alpha_low": lo, "alpha_high": hi}
    for name, alpha in (("base", 0.0), ("low", lo), ("high", hi)):
        for subset, col, key in (("holdout", "sycophantic_rate", "sycophantic"), ("holdout", "pushback_rate", "pushback"),
                                 ("holdout[standard]", "sycophantic_rate", "sycophantic_standard"),
                                 ("holdout[style_swapped]", "sycophantic_rate", "sycophantic_swapped"),
                                 ("user_correct", "false_pushback_rate", "false_pushback"),
                                 ("user_correct", "confirms_rate", "confirms"),
                                 ("holdout", "perplexity_median", "perplexity"), ("holdout", "mean_score", "score"),
                                 ("holdout", "exclamations_mean", "exclamations"), ("holdout", "emoji_mean", "emoji")):
            out[f"{key}_{name}"] = get("steered", subset, alpha, col)
            if name != "base":
                out[f"random_{key}_{name}"] = get("random", subset, alpha, col)
    out["n_holdout_eval"] = int(summary[(summary.subset == "holdout") & (summary.condition == "steered")].n_prompts.max()) \
        if (summary.subset == "holdout").any() else 0
    out["n_user_correct_eval"] = int(summary[summary.subset == "user_correct"].n_prompts.max()) \
        if (summary.subset == "user_correct").any() else 0
    out["has_labels"] = bool(summary.n_labeled.sum() > 0)
    return out
