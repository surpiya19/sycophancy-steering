"""Forward hooks for reading and steering the residual stream.

All hooks attach to a decoder block and see that block's output, so a vector
read at layer i is injected at exactly the same place it was measured.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def hidden_of(output):
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output, h):
    return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h


class LayerReducer:
    """Reduce each layer's output to (reply-span mean, last prompt token) inside the hook.

    Nothing of size [batch, seq, hidden] is kept, so memory stays O(batch x hidden)
    per layer regardless of sequence length or dataset size. Call `set_batch`
    before each forward pass; `callback(layer, reply_mean, prompt_last)` receives
    float32 tensors of shape [batch, hidden].
    """

    def __init__(self, layers: nn.ModuleList, callback: Callable[[int, torch.Tensor, torch.Tensor], None],
                 norm_callback: Callable[[int, torch.Tensor], None] | None = None):
        self.layers = layers
        self.callback = callback
        self.norm_callback = norm_callback
        self.handles: list = []
        self.reply_mask: torch.Tensor | None = None
        self.prompt_last_idx: torch.Tensor | None = None
        self.norm_mask: torch.Tensor | None = None

    def set_batch(self, reply_mask: torch.Tensor, prompt_last_idx: torch.Tensor, norm_mask: torch.Tensor | None = None):
        self.reply_mask, self.prompt_last_idx, self.norm_mask = reply_mask, prompt_last_idx, norm_mask

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            h = hidden_of(output).float()
            m = self.reply_mask.to(h.device, h.dtype).unsqueeze(-1)
            reply_mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            rows = torch.arange(h.shape[0], device=h.device)
            prompt_last = h[rows, self.prompt_last_idx.to(h.device)]
            self.callback(idx, reply_mean.cpu(), prompt_last.cpu())
            if self.norm_callback is not None and self.norm_mask is not None:
                norms = h.norm(dim=-1)[self.norm_mask.to(h.device)]
                self.norm_callback(idx, norms.cpu())
        return hook

    def __enter__(self):
        self.handles = [layer.register_forward_hook(self._make_hook(i)) for i, layer in enumerate(self.layers)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


class Steering:
    """Add `coeff * vector` to a layer's output, and optionally record per-step readouts.

    positions="all" steers every position (prompt and generated tokens).
    positions="generated" leaves the prompt untouched and steers only the last
    position of the prefill (which produces the first new token) and every
    decoding step after it.

    `readouts` maps a name to (module, unit_vector). On each forward call the
    projection of the module's last-position output onto the vector is recorded
    in `self.records[name]` as a [batch] tensor. During `generate`, call k
    corresponds to the state that chose generated token k.
    """

    def __init__(self, module: nn.Module, vector: torch.Tensor, coeff: float, positions: str = "all",
                 readouts: dict[str, tuple[nn.Module, torch.Tensor]] | None = None):
        self.module = module
        self.vector = vector
        self.coeff = float(coeff)
        self.positions = positions
        self.readouts = readouts or {}
        self.records: dict[str, list[torch.Tensor]] = {k: [] for k in self.readouts}
        self.handles: list = []

    def _steer(self, module, inputs, output):
        if self.coeff == 0.0:
            return output
        h = hidden_of(output)
        delta = (self.coeff * self.vector).to(h.device, h.dtype)
        if self.positions == "all" or h.shape[1] == 1:
            h = h + delta
        else:  # prefill under positions="generated": only the position that emits the first new token
            h = h.clone()
            h[:, -1, :] += delta
        return replace_hidden(output, h)

    def _make_readout(self, name: str, unit: torch.Tensor):
        def hook(module, inputs, output):
            h = hidden_of(output)[:, -1, :].float()
            self.records[name].append((h @ unit.to(h.device, torch.float32)).cpu())
        return hook

    def __enter__(self):
        self.handles.append(self.module.register_forward_hook(self._steer))
        # Registered after the steering hook, so a readout on the same module sees the steered output.
        for name, (mod, unit) in self.readouts.items():
            self.handles.append(mod.register_forward_hook(self._make_readout(name, unit)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def stacked(self, name: str) -> torch.Tensor:
        """[batch, steps] tensor of readouts for `name`."""
        recs = self.records[name]
        return torch.stack(recs, dim=1) if recs else torch.empty(0, 0)
