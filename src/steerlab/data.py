"""Dataset loading: contrastive pairs and free-form prompt lists."""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REQUIRED_KEYS = ("prompt", "honest", "sycophantic")


@dataclass(frozen=True)
class Pair:
    """One user message with an honest reply and a sycophantic reply."""
    id: str
    domain: str
    prompt: str
    honest: str
    sycophantic: str
    style: str = "standard"   # standard | style_swapped (honest reply exclamatory, sycophantic reply calm)


@dataclass(frozen=True)
class EvalPrompt:
    id: str
    prompt: str
    domain: str = "custom"
    source: str = "custom"   # holdout | user_correct | custom
    style: str = ""
    premise: str = "false"   # "false": the user is wrong (honest = push back); "true": the user is right (honest = confirm)


def load_pairs(path: str | Path) -> list[Pair]:
    """Load contrastive pairs from JSONL (one object per line).

    Required keys: prompt, honest, sycophantic. Optional: id, domain.
    """
    pairs: list[Pair] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            missing = [k for k in REQUIRED_KEYS if not str(rec.get(k, "")).strip()]
            if missing:
                raise ValueError(f"{path}:{lineno} missing {missing}")
            pid = str(rec.get("id") or f"pair-{lineno:05d}")
            if pid in seen:
                raise ValueError(f"{path}:{lineno} duplicate id {pid!r}")
            seen.add(pid)
            pairs.append(Pair(pid, str(rec.get("domain") or "general"),
                              rec["prompt"], rec["honest"], rec["sycophantic"], str(rec.get("style") or "standard")))
    if len(pairs) < 2:
        raise ValueError(f"{path} needs at least 2 pairs, found {len(pairs)}")
    return pairs


def stratified_split(pairs: list[Pair], holdout_fraction: float, seed: int) -> tuple[list[Pair], list[Pair]]:
    """Hold out a fraction of each domain so every domain is represented in both splits
    whenever it has at least two pairs."""
    rng = random.Random(seed)
    by_domain: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        by_domain[p.domain].append(p)
    train, holdout = [], []
    for domain in sorted(by_domain):
        group = by_domain[domain][:]
        rng.shuffle(group)
        n_hold = round(len(group) * holdout_fraction)
        if len(group) >= 2:
            n_hold = min(max(n_hold, 1), len(group) - 1)
        else:
            n_hold = 0
        holdout.extend(group[:n_hold])
        train.extend(group[n_hold:])
    if not holdout:
        raise ValueError("Holdout split is empty; add pairs or raise data.holdout_fraction")
    return train, holdout


def make_splits(cfg) -> tuple[list[Pair], list[Pair]]:
    """Training and held-out pairs.

    With `data.holdout_pairs` set, every pair in `data.pairs` trains the vector and the
    separate file is the held-out test set. Otherwise `data.pairs` is split by domain.
    """
    pairs = load_pairs(cfg.resolve_path(cfg.data.pairs))
    if not cfg.data.holdout_pairs:
        return stratified_split(pairs, cfg.data.holdout_fraction, cfg.data.seed)
    holdout = load_pairs(cfg.resolve_path(cfg.data.holdout_pairs))
    overlap_ids = {p.id for p in pairs} & {p.id for p in holdout}
    overlap_prompts = {p.prompt.strip().lower() for p in pairs} & {p.prompt.strip().lower() for p in holdout}
    if overlap_ids or overlap_prompts:
        raise ValueError(f"Held-out file overlaps the training pairs: ids={sorted(overlap_ids)[:5]} "
                         f"prompts={sorted(overlap_prompts)[:3]}")
    return pairs, holdout


def load_prompts(path: str | Path) -> list[EvalPrompt]:
    """Load your own prompts for testing.

    - `.txt`: one prompt per line (blank lines and lines starting with # ignored)
    - `.jsonl`: objects with `prompt` and optional `id`, `domain`, `premise`
    - `.csv`: a `prompt` column, optional `id`, `domain`, `premise`

    `premise` tells the judge which rubric to use: "false" (default; the user is wrong, so
    pushing back is correct) or "true" (the user is right, so disputing the claim is an error).
    """
    path = Path(path)
    rows: list[dict] = []
    if path.suffix == ".txt":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append({"prompt": line.strip()})
    elif path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    elif path.suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported prompts file type: {path.suffix} (use .txt, .jsonl or .csv)")
    out = []
    for i, r in enumerate(rows, 1):
        text = str(r.get("prompt", "")).strip()
        if not text:
            raise ValueError(f"{path}: row {i} has no prompt")
        premise = str(r.get("premise") or "false").strip().lower()
        if premise not in {"false", "true"}:
            raise ValueError(f"{path}: row {i} premise must be 'false' or 'true', got {premise!r}")
        out.append(EvalPrompt(str(r.get("id") or f"custom-{i:04d}"), text, str(r.get("domain") or "custom"), "custom",
                              premise=premise))
    return out


def prompts_from_text(text: str) -> list[EvalPrompt]:
    """One prompt per non-empty line; used by the app's batch tab."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return [EvalPrompt(f"custom-{i:04d}", l) for i, l in enumerate(lines, 1)]
