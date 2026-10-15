"""Fluency metrics: is a steered reply still well-formed text?

Strong steering can degrade output into repetition or nonsense, which a judge may then
score as "neutral" or even "pushback". These metrics catch that independently:

- perplexity: how surprising the reply is to the UNSTEERED model, given the prompt.
  Normal replies sit in a stable range; broken ones shoot up.
- distinct-2: share of word bigrams that are unique. Repetitive loops push it toward 0.
"""
from __future__ import annotations

import math

import torch

from .extraction import _encode_batch
from .modeling import LoadedModel


@torch.no_grad()
def reply_nll(lm: LoadedModel, prompts: list[str], replies: list[str], batch_size: int = 8) -> list[float]:
    """Mean negative log-likelihood per reply token under the unsteered model (nan for empty replies)."""
    out: list[float] = []
    for i in range(0, len(prompts), batch_size):
        ps, rs = prompts[i:i + batch_size], replies[i:i + batch_size]
        safe = [r if r.strip() else " " for r in rs]
        enc, reply_mask, _, _ = _encode_batch(lm, ps, safe)
        logits = lm.model(**enc, use_cache=False).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        target = enc.input_ids[:, 1:]
        tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)          # log p(token t | < t)
        mask = reply_mask[:, 1:].to(tok_lp.device)
        n = mask.sum(1)
        nll = -(tok_lp * mask).sum(1) / n.clamp(min=1)
        for r, v, k in zip(rs, nll.tolist(), n.tolist()):
            out.append(float("nan") if not r.strip() or k == 0 else float(v))
    return out


def perplexity(nll: float) -> float:
    return float("nan") if nll != nll else float(math.exp(min(nll, 50.0)))


def distinct_2(text: str) -> float:
    words = text.split()
    if len(words) < 2:
        return float("nan")
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / len(bigrams)
