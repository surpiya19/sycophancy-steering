import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "contrastive_pairs.jsonl"


def _load_builder():
    spec = importlib.util.spec_from_file_location("make_tiny_model", ROOT / "scripts" / "make_tiny_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_tiny_model"] = mod
    spec.loader.exec_module(mod)
    return mod.build


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    return _load_builder()(tmp_path_factory.mktemp("tiny") / "model", PAIRS, hidden=32, layers=4)


@pytest.fixture(scope="session")
def lm(tiny_model_dir):
    from steerlab.modeling import load_model
    return load_model(str(tiny_model_dir), dtype="float32", device="cpu")


@pytest.fixture(scope="session")
def split():
    from steerlab.data import load_pairs, stratified_split
    return stratified_split(load_pairs(PAIRS), 0.2, 42)


@pytest.fixture(scope="session")
def art(lm, split):
    from steerlab.extraction import extract
    train, holdout = split
    return extract(lm, train, holdout, batch_size=8)
