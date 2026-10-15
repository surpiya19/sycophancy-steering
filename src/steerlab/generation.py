"""Batched steered generation with per-token, explainable readouts."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .extraction import SteeringArtifacts
from .hooks import Steering
from .modeling import LoadedModel, check_layer


@dataclass
class SteerSpec:
    layer: int
    readout_layer: int
    alpha: float = 0.0
    scale_fraction: float = 0.2
    positions: str = "all"


@dataclass
class Generation:
    prompt: str
    alpha: float
    text: str
    tokens: list[str] = field(default_factory=list)
    # score[k]: readout-layer state that chose token k, on the -1 (typical honest) .. +1 (typical sycophantic) scale
    scores: list[float] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return float(sum(self.scores) / len(self.scores)) if self.scores else float("nan")


def _steering(lm: LoadedModel, art: SteeringArtifacts, spec: SteerSpec) -> Steering:
    check_layer(lm, spec.layer, "steering.layer")
    check_layer(lm, spec.readout_layer, "steering.readout_layer")
    coeff = art.coeff(spec.layer, spec.alpha, spec.scale_fraction)
    readouts = {"readout": (lm.layers[spec.readout_layer], art.vector(spec.readout_layer))}
    return Steering(lm.layers[spec.layer], art.vector(spec.layer), coeff, spec.positions, readouts)


@torch.no_grad()
def generate(lm: LoadedModel, art: SteeringArtifacts, prompts: list[str], spec: SteerSpec,
             max_new_tokens: int = 90, greedy: bool = True, temperature: float = 0.7, top_p: float = 0.9,
             seed: int | None = 1234, batch_size: int = 8) -> list[Generation]:
    """Generate for many prompts at one steering strength.

    With the same seed and batch composition, sampled runs at different alphas
    draw the same random numbers, so differences come from steering rather
    than sampling luck.
    """
    tok = lm.tokenizer
    tok.padding_side = "left"
    stop = lm.stop_token_ids
    results: list[Generation] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok([lm.chat_prompt(p) for p in chunk], add_special_tokens=False, padding=True,
                  return_tensors="pt").to(lm.device)
        if seed is not None:
            torch.manual_seed(seed)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, do_sample=not greedy)
        if not greedy:
            kwargs.update(temperature=temperature, top_p=top_p)
        steer = _steering(lm, art, spec)
        with steer:
            out = lm.model.generate(**enc, **kwargs)
        readout = art.score(spec.readout_layer, steer.stacked("readout"))  # [B, steps]
        new = out[:, enc.input_ids.shape[1]:]
        for b, prompt in enumerate(chunk):
            ids = new[b].tolist()
            n = next((k for k, t in enumerate(ids) if t in stop), len(ids))
            ids = ids[:n]
            results.append(Generation(
                prompt=prompt, alpha=spec.alpha,
                text=tok.decode(ids, skip_special_tokens=True).strip(),
                tokens=[tok.decode([t]) for t in ids],
                scores=[float(x) for x in readout[b, :n]] if readout.numel() else [],
            ))
    return results


@torch.no_grad()
def next_token_shift(lm: LoadedModel, art: SteeringArtifacts, prompt: str, spec: SteerSpec, k: int = 8):
    """Exact effect of steering on the model's FIRST word for this prompt.

    Returns (promoted, suppressed): lists of (token, p_baseline, p_steered),
    ranked by the change in probability.
    """
    tok = lm.tokenizer
    enc = tok(lm.chat_prompt(prompt), add_special_tokens=False, return_tensors="pt").to(lm.device)
    base = lm.model(**enc).logits[0, -1].float().softmax(-1)
    steer = _steering(lm, art, spec)
    with steer:
        steered = lm.model(**enc).logits[0, -1].float().softmax(-1)
    delta = steered - base
    row = lambda i: (tok.decode([int(i)]), float(base[i]), float(steered[i]))
    return [row(i) for i in torch.topk(delta, k).indices], [row(i) for i in torch.topk(-delta, k).indices]


@torch.no_grad()
def prompt_reading(lm: LoadedModel, art: SteeringArtifacts, prompt: str, layer: int) -> dict:
    """Unsteered reading of the prompt's last token at `layer`, before the model writes anything."""
    enc = lm.tokenizer(lm.chat_prompt(prompt), add_special_tokens=False, return_tensors="pt").to(lm.device)
    steer = Steering(lm.layers[layer], art.vector(layer), 0.0, readouts={"r": (lm.layers[layer], art.vector(layer))})
    with steer:
        lm.model(**enc)
    proj = float(steer.stacked("r")[0, -1])
    return {"projection": proj, "score": float(art.score(layer, proj)), "percentile": art.prompt_percentile(layer, proj)}
