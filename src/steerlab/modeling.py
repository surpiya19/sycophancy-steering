"""Model loading and architecture plumbing that works across Hugging Face causal LMs.

Throughout the project, "layer i" means the OUTPUT of decoder block i (0-based).
Activations are always read with forward hooks on that block, never from
`output_hidden_states`. That makes reading and steering hit exactly the same
tensor by construction (hidden_states is offset by one for the embeddings, and
its last entry has the final norm applied, which hooks do not see).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

_LAYER_PATHS = (
    "model.layers",            # Llama, Mistral, Qwen, SmolLM, Gemma, Phi-3
    "model.decoder.layers",    # OPT
    "transformer.h",           # GPT-2, GPT-J, Falcon
    "gpt_neox.layers",         # Pythia, GPT-NeoX
    "transformer.blocks",      # MPT
)
_NORM_PATHS = ("model.norm", "model.decoder.final_layer_norm", "transformer.ln_f", "gpt_neox.final_layer_norm",
               "transformer.norm_f")


def _get_path(obj: object, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    for path in _LAYER_PATHS:
        layers = _get_path(model, path)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers
    raise ValueError(f"Could not locate decoder layers on {type(model).__name__}; add its path to _LAYER_PATHS")


def find_final_norm(model: nn.Module) -> nn.Module | None:
    for path in _NORM_PATHS:
        mod = _get_path(model, path)
        if isinstance(mod, nn.Module):
            return mod
    return None


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "half":  # half-size weights on any accelerator; CPU stays float32
        return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device.type, torch.float32)
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


@dataclass
class LoadedModel:
    model: nn.Module
    tokenizer: PreTrainedTokenizerBase
    layers: nn.ModuleList
    final_norm: nn.Module | None
    device: torch.device
    model_id: str

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def chat_prompt(self, user_message: str) -> str:
        """The user turn rendered exactly as the model sees it at generation time."""
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template([{"role": "user", "content": user_message}],
                                           tokenize=False, add_generation_prompt=True)
        return f"User: {user_message}\nAssistant: "

    def encode(self, text: str) -> list[int]:
        # Chat templates insert their own special tokens; never add them twice.
        return self.tokenizer(text, add_special_tokens=False).input_ids

    @property
    def stop_token_ids(self) -> set[int]:
        ids = set()
        eos = self.model.generation_config.eos_token_id if self.model.generation_config else None
        for x in (eos if isinstance(eos, list) else [eos]) + [self.tokenizer.eos_token_id]:
            if x is not None:
                ids.add(int(x))
        return ids


def check_layer(lm: LoadedModel, layer: int, name: str = "layer") -> int:
    if not 0 <= layer < lm.num_layers:
        raise ValueError(f"{name}={layer} is out of range for {lm.model_id} ({lm.num_layers} layers, 0-{lm.num_layers - 1})")
    return layer


def load_model(model_id: str, dtype: str = "auto", device: str = "auto", trust_remote_code: bool = False) -> LoadedModel:
    dev = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, dev)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, trust_remote_code=trust_remote_code)
    except TypeError:  # transformers < 4.56 uses torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
    model.to(dev).eval()
    model.requires_grad_(False)
    return LoadedModel(model, tokenizer, find_decoder_layers(model), find_final_norm(model), dev, model_id)


def load_from_config(cfg) -> LoadedModel:
    m = cfg.model
    return load_model(m.id, m.dtype, m.device, m.trust_remote_code)
