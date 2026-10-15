"""Alpha sweep: behaviour (judge labels) and internal readout as steering strength changes."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .data import EvalPrompt, Pair, load_prompts
from .extraction import SteeringArtifacts
from .generation import SteerSpec, generate
from .judges import LABELS, Judge
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


def build_eval_prompts(cfg, holdout: list[Pair], extra_file: str | None = None) -> list[EvalPrompt]:
    prompts: list[EvalPrompt] = []
    if cfg.eval.include_holdout:
        prompts += [EvalPrompt(p.id, p.prompt, p.domain, "holdout") for p in holdout]
    path = extra_file or cfg.eval.prompts_file
    if path:
        prompts += load_prompts(cfg.resolve_path(path))
    if not prompts:
        raise ValueError("No evaluation prompts: enable eval.include_holdout or pass a prompts file")
    return prompts


def run_alpha_sweep(lm: LoadedModel, art: SteeringArtifacts, cfg, prompts: list[EvalPrompt], judge: Judge,
                    alphas: list[float] | None = None) -> pd.DataFrame:
    alphas = alphas if alphas is not None else cfg.eval.alphas
    rows = []
    texts = [p.prompt for p in prompts]
    for alpha in alphas:
        spec = SteerSpec(cfg.steering.layer, cfg.steering.readout_layer, alpha,
                         cfg.steering.scale_fraction, cfg.steering.positions)
        gens = generate(lm, art, texts, spec, max_new_tokens=cfg.eval.max_new_tokens, greedy=True,
                        batch_size=cfg.generation.batch_size)
        labels = judge.judge_many([(g.prompt, g.text) for g in gens])
        for p, g, label in zip(prompts, gens, labels):
            rows.append({"alpha": alpha, "prompt_id": p.id, "source": p.source, "domain": p.domain,
                         "prompt": p.prompt, "response": g.text, "label": label,
                         "mean_score": g.mean_score, "first_token_score": g.scores[0] if g.scores else float("nan"),
                         "n_tokens": len(g.tokens)})
        n_lab = sum(l is not None for l in labels)
        syc = sum(l == "sycophantic" for l in labels)
        print(f"[*] alpha={alpha:+.2f}  mean score={pd.Series([g.mean_score for g in gens]).mean():+.2f}"
              + (f"  sycophantic {syc}/{n_lab}" if n_lab else ""))
    return pd.DataFrame(rows)


def summarize(responses: pd.DataFrame) -> pd.DataFrame:
    """Per (source, alpha): label rates with 95% Wilson intervals and the mean internal score."""
    out = []
    groups = [("all", responses)] + [(s, g) for s, g in responses.groupby("source")]
    if responses["source"].nunique() == 1:
        groups = groups[:1]
    for source, df in groups:
        for alpha, g in df.groupby("alpha"):
            labeled = g[g["label"].notna()]
            n = len(labeled)
            row = {"source": source, "alpha": alpha, "n_prompts": len(g), "n_labeled": n,
                   "mean_score": g["mean_score"].mean(),
                   "mean_score_sem": g["mean_score"].std(ddof=1) / math.sqrt(len(g)) if len(g) > 1 else float("nan")}
            for label in LABELS:
                k = int((labeled["label"] == label).sum())
                lo, hi = wilson(k, n)
                row[f"{label}_rate"] = k / n if n else float("nan")
                row[f"{label}_lo"], row[f"{label}_hi"] = lo, hi
            out.append(row)
    return pd.DataFrame(out)


def save_eval(out_dir: Path, responses: pd.DataFrame, summary: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    responses.to_csv(out_dir / "responses.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
