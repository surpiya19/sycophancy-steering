"""Hugging Face Space entry point. Built by scripts/prepare_space.sh; not used for local runs."""
from pathlib import Path

from steerlab.app import build_app
from steerlab.config import load_config
from steerlab.extraction import SteeringArtifacts
from steerlab.modeling import load_from_config

ROOT = Path(__file__).parent
cfg = load_config(ROOT / "configs" / "default.yaml",
                  ["model.device=cpu", "judge.backend=none", "generation.max_new_tokens=60", f"output_dir={ROOT / 'runs'}"])
lm = load_from_config(cfg)
art = SteeringArtifacts.load(cfg.artifact_dir / "vectors.pt")
demo = build_app(lm, art, cfg, cfg.artifact_dir)

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
