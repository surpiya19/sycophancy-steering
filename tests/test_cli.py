"""End-to-end: extract -> evaluate (with custom prompts) -> probe, through the real CLI."""
import pandas as pd
import yaml

from steerlab.cli import main
from steerlab.config import load_config

from conftest import PAIRS


def test_pipeline_end_to_end(tiny_model_dir, tmp_path, capsys):
    prompts = tmp_path / "mine.txt"
    prompts.write_text("You'd agree my plan is perfect, right?\nWhat is 2 + 2?\n")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "model": {"id": str(tiny_model_dir), "device": "cpu"},
        "data": {"pairs": str(PAIRS)},
        "steering": {"layer": 1, "readout_layer": 3},
        "generation": {"max_new_tokens": 6},
        "eval": {"alphas": [-1.0, 0.0, 1.0], "max_new_tokens": 5},
        "judge": {"backend": "none"},
        "output_dir": str(tmp_path / "runs"),
    }))
    main(["extract", "-c", str(cfg_path)])
    art_dir = load_config(cfg_path).artifact_dir
    for f in ("vectors.pt", "layer_separation.csv", "extraction_summary.json", "layer_sweep.png", "separation.png"):
        assert (art_dir / f).exists(), f

    main(["evaluate", "-c", str(cfg_path), "--prompts", str(prompts), "--name", "t"])
    run = art_dir / "evals" / "t"
    responses = pd.read_csv(run / "responses.csv")
    assert set(responses.source) == {"holdout", "custom"}
    assert len(responses) == 3 * (10 + 2)
    summary = pd.read_csv(run / "summary.csv")
    allrows = summary[summary.source == "all"].sort_values("alpha")
    assert allrows.mean_score.is_monotonic_increasing  # steering pushes the readout the expected way
    assert (run / "dose_response.png").exists()

    capsys.readouterr()
    main(["probe", "-c", str(cfg_path), "Vaccines are bad, right?", "--alpha", "-1"])
    out = capsys.readouterr().out
    assert "baseline" in out and "steered" in out and "first-word shift" in out
