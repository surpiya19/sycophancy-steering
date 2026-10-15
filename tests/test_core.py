"""Model-free tests: data, config, tactics, statistics, judge cache."""
import json

import numpy as np
import pytest

from steerlab.config import load_config
from steerlab.data import load_pairs, load_prompts, make_splits, prompts_from_text, stratified_split
from steerlab.evaluation import balanced_subset, build_eval_prompts
from steerlab.evaluation import wilson
from steerlab.extraction import auroc
from steerlab.judges import CachedJudge, Judge
from steerlab.tactics import TacticAuditor

from conftest import HOLDOUT, PAIRS


def test_dataset_loads_and_split_is_stratified_and_deterministic():
    pairs = load_pairs(PAIRS)
    assert len(pairs) == 50
    tr1, ho1 = stratified_split(pairs, 0.2, 42)
    tr2, ho2 = stratified_split(pairs, 0.2, 42)
    assert [p.id for p in ho1] == [p.id for p in ho2]
    assert not {p.id for p in tr1} & {p.id for p in ho1}
    assert len(tr1) + len(ho1) == 50
    assert {p.domain for p in ho1} == {p.domain for p in pairs}  # every domain is evaluated


def test_load_pairs_rejects_bad_rows(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"prompt": "x", "honest": "y"}) + "\n" + json.dumps({"prompt": "a", "honest": "b", "sycophantic": "c"}))
    with pytest.raises(ValueError, match="missing"):
        load_pairs(bad)


@pytest.mark.parametrize("suffix,content", [
    (".txt", "# comment\nfirst prompt\n\nsecond prompt\n"),
    (".csv", "prompt,domain\nfirst prompt,x\nsecond prompt,y\n"),
    (".jsonl", '{"prompt": "first prompt"}\n{"prompt": "second prompt", "id": "b"}\n'),
])
def test_load_custom_prompts(tmp_path, suffix, content):
    f = tmp_path / f"p{suffix}"
    f.write_text(content)
    got = load_prompts(f)
    assert [p.prompt for p in got] == ["first prompt", "second prompt"]
    assert all(p.source == "custom" for p in got)
    assert [p.prompt for p in prompts_from_text("a\n\n b \n")] == ["a", "b"]


def test_config_overrides_and_unknown_keys():
    cfg = load_config(None, ["steering.layer=7", "eval.alphas=[-2, 2]", "judge.backend=none"])
    assert cfg.steering.layer == 7 and cfg.eval.alphas == [-2, 2] and cfg.judge.backend == "none"
    with pytest.raises(KeyError):
        load_config(None, ["steering.nope=1"])
    with pytest.raises(ValueError):
        load_config(None, ["steering.positions=sometimes"])


def test_tactics():
    a = TacticAuditor()
    rep = a.audit("As a doctor with 15 years of experience, vaccines are bad, right?")
    assert set(rep.hits) == {"Leading / Forced Agreement", "Authority / Epistemic Intimidation"}
    assert rep.level == "moderate"
    assert a.audit("What is the boiling point of water?").level == "low"


def test_wilson_and_auroc():
    lo, hi = wilson(0, 10)
    assert lo == 0 and 0.2 < hi < 0.35
    lo, hi = wilson(10, 10)
    assert hi == 1 and 0.65 < lo < 0.8
    assert auroc(np.array([2, 3]), np.array([0, 1])) == 1.0
    assert auroc(np.array([1.0]), np.array([1.0])) == 0.5


def test_judge_cache(tmp_path):
    class Counting(Judge):
        name = "count"
        calls = 0

        def judge(self, p, r):
            Counting.calls += 1
            return "pushback"

    j = CachedJudge(Counting(), tmp_path / "cache.jsonl")
    assert j.judge("p", "r") == "pushback" and j.judge("p", "r") == "pushback"
    assert Counting.calls == 1
    assert CachedJudge(Counting(), tmp_path / "cache.jsonl").judge("p", "r") == "pushback"  # persisted
    assert Counting.calls == 1


def test_holdout_file_is_separate_and_style_balanced():
    cfg = load_config(None, [f"data.pairs={PAIRS}", f"data.holdout_pairs={HOLDOUT}"])
    train, holdout = make_splits(cfg)
    assert len(train) == 50 and len(holdout) == 100
    assert {p.style for p in train} == {"standard"}
    styles = [p.style for p in holdout]
    assert styles.count("standard") == 60 and styles.count("style_swapped") == 40
    assert not {p.prompt for p in train} & {p.prompt for p in holdout}
    # The confound the swapped set exists to test: training tone predicts the label perfectly.
    assert all("!" in p.sycophantic and "!" not in p.honest for p in train)
    assert all("!" in p.honest and "!" not in p.sycophantic for p in holdout if p.style == "style_swapped")


def test_overlapping_holdout_is_rejected():
    cfg = load_config(None, [f"data.pairs={PAIRS}", f"data.holdout_pairs={PAIRS}"])
    with pytest.raises(ValueError, match="overlaps"):
        make_splits(cfg)


def test_holdout_limit_keeps_both_styles():
    cfg = load_config(None, [f"data.pairs={PAIRS}", f"data.holdout_pairs={HOLDOUT}", "eval.holdout_limit=10"])
    _, holdout = make_splits(cfg)
    prompts = build_eval_prompts(cfg, holdout)
    assert len(prompts) == 10 and {p.style for p in prompts} == {"standard", "style_swapped"}
    assert balanced_subset(prompts, None) == prompts
