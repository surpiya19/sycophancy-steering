"""Build a tiny, randomly initialised Llama-style chat model for offline smoke tests.

Its outputs are gibberish, but it exercises every code path (chat template,
batching, hooks, extraction, generation, evaluation, app) in seconds on a CPU
with no downloads.

    python scripts/make_tiny_model.py runs/tiny-model
    steerlab extract -c configs/tiny.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CHATML = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build(out_dir: str | Path, pairs_path: str | Path | None = None, hidden: int = 64, layers: int = 6, seed: int = 0) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    pairs_path = Path(pairs_path or root / "data" / "contrastive_pairs.jsonl")
    corpus = []
    for line in pairs_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        corpus += [rec["prompt"], rec["honest"], rec["sycophantic"]]
    corpus += ["user", "assistant", "system", "A B C"]

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   bos_token="<|im_start|>")
    fast.chat_template = CHATML
    fast.save_pretrained(out)

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(fast), hidden_size=hidden, intermediate_size=hidden * 2,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=512, eos_token_id=fast.eos_token_id,
                      pad_token_id=fast.pad_token_id, bos_token_id=fast.bos_token_id, tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg)
    model.generation_config.eos_token_id = fast.eos_token_id
    model.generation_config.pad_token_id = fast.pad_token_id
    model.save_pretrained(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="runs/tiny-model")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    a = ap.parse_args()
    print(f"[+] Tiny model written to {build(a.out_dir, hidden=a.hidden, layers=a.layers)}")
