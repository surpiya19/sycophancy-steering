"""Extract one sycophancy direction per layer and measure how well each separates the classes.

Two streaming passes, so memory is O(layers x hidden) no matter how many pairs:
  1. Training pairs: accumulate sum(sycophantic_mean - honest_mean) per layer -> unit vectors.
     Also samples token norms to calibrate steering strength per layer.
  2. All pairs: project reply means and the last prompt token onto the vectors.
     Held-out projections measure separation; all of them calibrate the explainable score.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import Pair
from .hooks import LayerReducer
from .modeling import LoadedModel


# artifacts
@dataclass
class SteeringArtifacts:
    model_id: str
    num_layers: int
    hidden_size: int
    vectors: torch.Tensor            # [L, D] unit vectors, one per decoder layer output
    mean_diff_norm: torch.Tensor     # [L] length of the raw mean difference
    token_norm: torch.Tensor         # [L] median per-token residual norm (calibrates alpha)
    pair_ids: list[str]
    domains: list[str]
    is_holdout: torch.Tensor         # [N] bool
    honest_proj: torch.Tensor        # [N, L] reply-mean projection onto that layer's vector
    syc_proj: torch.Tensor           # [N, L]
    prompt_proj: torch.Tensor        # [N, L] last prompt token projection (the "pre-bias" reading)
    meta: dict = field(default_factory=dict)
    styles: list[str] = field(default_factory=list)  # per pair: standard | style_swapped

    # steering helpers
    def vector(self, layer: int) -> torch.Tensor:
        return self.vectors[layer]

    def coeff(self, layer: int, alpha: float, scale_fraction: float) -> float:
        """alpha=1 adds `scale_fraction` of a typical token's norm at that layer."""
        return float(alpha) * float(self.token_norm[layer]) * scale_fraction

    # explainable score: -1 = typical honest reply, +1 = typical sycophantic reply
    def anchors(self, layer: int) -> tuple[float, float]:
        train = ~self.is_holdout
        return float(self.honest_proj[train, layer].mean()), float(self.syc_proj[train, layer].mean())

    def score(self, layer: int, projection):
        h, s = self.anchors(layer)
        mid, half = (h + s) / 2, max((s - h) / 2, 1e-6)
        if isinstance(projection, torch.Tensor):
            return (projection.float() - mid) / half
        return (np.asarray(projection, dtype=float) - mid) / half

    def prompt_percentile(self, layer: int, projection: float) -> float:
        """Where a prompt's pre-generation reading falls among the dataset prompts (0-100)."""
        ref = self.prompt_proj[:, layer].numpy()
        return float((ref < projection).mean() * 100)

    # persistence
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: getattr(self, k) for k in self.__dataclass_fields__}, path)

    @classmethod
    def load(cls, path: str | Path) -> "SteeringArtifacts":
        if not Path(path).exists():
            raise FileNotFoundError(f"No vectors at {path}. Run `steerlab extract` first.")
        data = torch.load(path, map_location="cpu", weights_only=False)
        data.setdefault("styles", ["standard"] * len(data["pair_ids"]))  # files from before styles existed
        return cls(**data)


# batching
def _batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


_warned_boundary = False


def _encode_batch(lm: LoadedModel, prompts: list[str], replies: list[str]):
    """Tokenize chat-templated prompt + reply, returning tensors and per-example reply spans."""
    global _warned_boundary
    tok = lm.tokenizer
    prompt_texts = [lm.chat_prompt(p) for p in prompts]
    full_texts = [pt + r for pt, r in zip(prompt_texts, replies)]
    tok.padding_side = "right"
    enc = tok(full_texts, add_special_tokens=False, padding=True, return_tensors="pt")
    B, T = enc.input_ids.shape
    reply_mask = torch.zeros(B, T)
    prompt_last = torch.zeros(B, dtype=torch.long)
    for b, pt in enumerate(prompt_texts):
        p_ids = lm.encode(pt)
        full_len = int(enc.attention_mask[b].sum())
        start = len(p_ids)
        if enc.input_ids[b, :start].tolist() != p_ids and not _warned_boundary:
            print("[!] Prompt/reply tokens merged at the boundary for some examples; reply span may shift by one token.")
            _warned_boundary = True
        reply_mask[b, start:full_len] = 1.0
        prompt_last[b] = start - 1
    norm_mask = enc.attention_mask.bool().clone()
    norm_mask[:, 0] = False  # the first position is an attention sink with an outsized norm
    return enc.to(lm.device), reply_mask, prompt_last, norm_mask


@torch.no_grad()
def _run(lm: LoadedModel, pairs: list[Pair], batch_size: int, callback, norm_callback=None, desc: str = ""):
    """Forward honest and sycophantic texts for each pair; callback(kind, layer, rows, reply_mean, prompt_last)."""
    done = 0
    for chunk in _batches(pairs, batch_size):
        rows = list(range(done, done + len(chunk)))
        for kind, replies in (("honest", [p.honest for p in chunk]), ("syc", [p.sycophantic for p in chunk])):
            enc, reply_mask, prompt_last, norm_mask = _encode_batch(lm, [p.prompt for p in chunk], replies)
            reducer = LayerReducer(lm.layers,
                                   callback=lambda layer, rm, pl, k=kind, r=rows: callback(k, layer, r, rm, pl),
                                   norm_callback=norm_callback if kind == "honest" else None)
            reducer.set_batch(reply_mask, prompt_last, norm_mask)
            with reducer:
                lm.model(**enc, use_cache=False)
        done += len(chunk)
        print(f"\r[*] {desc} {done}/{len(pairs)} pairs", end="", flush=True)
    print()


def extract(lm: LoadedModel, train: list[Pair], holdout: list[Pair], batch_size: int = 8,
            max_norm_samples: int = 50_000) -> SteeringArtifacts:
    L, D = lm.num_layers, lm.hidden_size

    # ---- pass 1: mean difference on training pairs
    diff_sum = torch.zeros(L, D, dtype=torch.float64)
    norm_samples: list[list[torch.Tensor]] = [[] for _ in range(L)]
    norm_counts = [0] * L
    sign = {"honest": -1.0, "syc": 1.0}

    def accumulate(kind, layer, rows, reply_mean, prompt_last):
        diff_sum[layer] += sign[kind] * reply_mean.double().sum(0)

    def collect_norms(layer, norms):
        if norm_counts[layer] < max_norm_samples:
            norm_samples[layer].append(norms)
            norm_counts[layer] += norms.numel()

    _run(lm, train, batch_size, accumulate, collect_norms, desc="pass 1/2 (vectors)")
    mean_diff = (diff_sum / len(train)).float()
    mean_diff_norm = mean_diff.norm(dim=-1)
    vectors = mean_diff / mean_diff_norm.clamp(min=1e-8).unsqueeze(-1)
    token_norm = torch.tensor([torch.cat(s).median().item() if s else 1.0 for s in norm_samples])

    # ---- pass 2: projections for every pair
    everyone = train + holdout
    N = len(everyone)
    proj = {"honest": torch.zeros(N, L), "syc": torch.zeros(N, L)}
    prompt_proj = torch.zeros(N, L)

    def project(kind, layer, rows, reply_mean, prompt_last):
        v = vectors[layer]
        proj[kind][rows, layer] = reply_mean @ v
        if kind == "honest":
            prompt_proj[rows, layer] = prompt_last @ v

    _run(lm, everyone, batch_size, project, desc="pass 2/2 (projections)")

    return SteeringArtifacts(
        model_id=lm.model_id, num_layers=L, hidden_size=D,
        vectors=vectors, mean_diff_norm=mean_diff_norm, token_norm=token_norm,
        pair_ids=[p.id for p in everyone], domains=[p.domain for p in everyone],
        is_holdout=torch.tensor([False] * len(train) + [True] * len(holdout)),
        honest_proj=proj["honest"], syc_proj=proj["syc"], prompt_proj=prompt_proj,
        meta={"n_train": len(train), "n_holdout": len(holdout)},
        styles=[p.style for p in everyone],
    )


# diagnostics
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random positive scores above a random negative (ties count half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def layer_separation(art: SteeringArtifacts) -> pd.DataFrame:
    """Per-layer separation on held-out pairs (and train, for comparison)."""
    rows = []
    styles = np.array(art.styles or ["standard"] * len(art.pair_ids))
    hold = art.is_holdout.numpy()
    splits = [("holdout", art.is_holdout), ("train", ~art.is_holdout)]
    held_styles = sorted(set(styles[hold]))
    if len(held_styles) > 1:  # e.g. holdout[standard] vs holdout[style_swapped]: does tone drive the score?
        splits += [(f"holdout[{s}]", torch.from_numpy(hold & (styles == s))) for s in held_styles]
    for split, mask in splits:
        for layer in range(art.num_layers):
            h = art.honest_proj[mask, layer].numpy()
            s = art.syc_proj[mask, layer].numpy()
            pooled = math.sqrt((h.var(ddof=1) + s.var(ddof=1)) / 2) if len(h) > 1 else float("nan")
            rows.append({
                "split": split, "layer": layer,
                "auroc": auroc(s, h),
                "pairwise_accuracy": float((s > h).mean()),   # sycophantic reply scores above its own honest twin
                "cohens_d": float((s.mean() - h.mean()) / pooled) if pooled and pooled > 0 else float("nan"),
                "mean_diff_norm": float(art.mean_diff_norm[layer]),
            })
    return pd.DataFrame(rows)


@torch.no_grad()
def direct_token_effects(lm: LoadedModel, art: SteeringArtifacts, layer: int, k: int = 15):
    """Tokens the vector pushes up / down through the unembedding (the "direct path").

    An approximation: it ignores every layer after `layer` and treats the final
    norm as its elementwise gain only. Use `generation.next_token_shift` for the
    exact effect on a specific prompt.
    """
    head = lm.model.get_output_embeddings()
    v = art.vector(layer).to(lm.device, torch.float32)
    gain = getattr(lm.final_norm, "weight", None)
    if gain is not None:
        v = v * gain.float()
    logits = head.weight.float() @ v
    top = torch.topk(logits, k)
    bottom = torch.topk(-logits, k)
    dec = lambda i: lm.tokenizer.decode([int(i)])
    return ([(dec(i), float(s)) for s, i in zip(top.values, top.indices)],
            [(dec(i), float(-s)) for s, i in zip(bottom.values, bottom.indices)])


def write_extraction_report(art: SteeringArtifacts, sep: pd.DataFrame, out_dir: Path, chosen_layer: int,
                            direct: tuple | None = None) -> dict:
    hold = sep[sep.split == "holdout"].set_index("layer")
    best = int(hold["auroc"].idxmax())
    summary = {
        "model_id": art.model_id,
        "n_train": art.meta["n_train"], "n_holdout": art.meta["n_holdout"],
        "steering_layer": chosen_layer,
        "steering_layer_holdout_auroc": float(hold.loc[chosen_layer, "auroc"]),
        "steering_layer_holdout_pairwise_accuracy": float(hold.loc[chosen_layer, "pairwise_accuracy"]),
        "best_layer_by_holdout_auroc": best,
        "best_layer_holdout_auroc": float(hold.loc[best, "auroc"]),
    }
    for split in sorted(s for s in sep.split.unique() if s.startswith("holdout[")):
        sub = sep[sep.split == split].set_index("layer")
        style = split[len("holdout["):-1]
        summary[f"steering_layer_auroc_{style}"] = float(sub.loc[chosen_layer, "auroc"])
        summary[f"steering_layer_pairwise_accuracy_{style}"] = float(sub.loc[chosen_layer, "pairwise_accuracy"])
        summary[f"n_holdout_{style}"] = int(sum(1 for s, h in zip(art.styles, art.is_holdout.tolist()) if h and s == style))
    if direct is not None:
        summary["direct_path_promoted"] = [t for t, _ in direct[0]]
        summary["direct_path_suppressed"] = [t for t, _ in direct[1]]
    (out_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    sep.to_csv(out_dir / "layer_separation.csv", index=False)
    return summary
