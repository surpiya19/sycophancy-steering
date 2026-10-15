"""Mechanics tests on a tiny random model: these are the bugs that silently ruin an experiment."""
import torch

from steerlab.extraction import _encode_batch, extract, layer_separation
from steerlab.generation import SteerSpec, generate, prompt_reading
from steerlab.hooks import Steering, hidden_of


@torch.no_grad()
def test_hook_reads_the_same_tensor_as_hidden_states(lm):
    """Layer i's hook output == hidden_states[i + 1] (the off-by-one the original script had)."""
    enc = lm.tokenizer("Hello there friend", return_tensors="pt", add_special_tokens=False)
    captured = {}
    handles = [layer.register_forward_hook(lambda m, i, o, k=k: captured.__setitem__(k, hidden_of(o).clone()))
               for k, layer in enumerate(lm.layers)]
    out = lm.model(**enc, output_hidden_states=True)
    for h in handles:
        h.remove()
    for k in range(lm.num_layers - 1):  # the last hidden_states entry has the final norm applied
        assert torch.allclose(captured[k], out.hidden_states[k + 1], atol=1e-5)
        assert not torch.allclose(captured[k], out.hidden_states[k], atol=1e-5)


def test_reply_span_covers_exactly_the_reply(lm):
    enc, mask, last, _ = _encode_batch(lm, ["Is it safe?", "Short"], ["No, it is not safe.", "A much longer reply here."])
    for b, reply in enumerate(["No, it is not safe.", "A much longer reply here."]):
        ids = enc.input_ids[b][mask[b].bool()].tolist()
        assert lm.tokenizer.decode(ids) == reply
        assert lm.tokenizer.decode(enc.input_ids[b, : int(last[b]) + 1]).endswith("assistant\n")


def test_extraction_is_batch_size_invariant(lm, split, art):
    train, holdout = split
    other = extract(lm, train, holdout, batch_size=3)
    assert torch.allclose(art.vectors, other.vectors, atol=1e-4)
    assert torch.allclose(art.syc_proj, other.syc_proj, atol=1e-3)
    assert torch.allclose(art.vectors.norm(dim=-1), torch.ones(lm.num_layers), atol=1e-5)


def test_layer_separation_has_every_layer(art):
    sep = layer_separation(art)
    assert set(sep.layer) == set(range(art.num_layers)) and set(sep.split) == {"train", "holdout"}
    assert sep.auroc.between(0, 1).all()


@torch.no_grad()
def test_steering_is_exact_and_cleans_up(lm, art):
    enc = lm.tokenizer(lm.chat_prompt("hi"), return_tensors="pt", add_special_tokens=False)
    base = lm.model(**enc).logits
    hooks_before = [len(l._forward_hooks) for l in lm.layers]  # transformers may keep its own hooks
    with Steering(lm.layers[1], art.vector(1), 0.0):
        same = lm.model(**enc).logits
    assert torch.equal(base, same)
    with Steering(lm.layers[1], art.vector(1), 5.0):
        moved = lm.model(**enc).logits
    assert not torch.allclose(base, moved)
    assert [len(l._forward_hooks) for l in lm.layers] == hooks_before
    assert torch.equal(lm.model(**enc).logits, base)


@torch.no_grad()
def test_generated_positions_mode_leaves_the_prompt_alone(lm, art):
    enc = lm.tokenizer(lm.chat_prompt("hello there"), return_tensors="pt", add_special_tokens=False)
    grab = {}
    h = lm.layers[2].register_forward_hook(lambda m, i, o: grab.__setitem__("x", hidden_of(o).clone()))
    lm.model(**enc)
    plain = grab["x"]
    with Steering(lm.layers[1], art.vector(1), 3.0, positions="generated"):
        lm.model(**enc)
    steered_gen = grab["x"]
    with Steering(lm.layers[1], art.vector(1), 3.0, positions="all"):
        lm.model(**enc)
    steered_all = grab["x"]
    h.remove()
    # Downstream of the steering layer, only the last prompt position (which emits the first new token) moves.
    assert torch.allclose(plain[:, :-1], steered_gen[:, :-1], atol=1e-5)
    assert not torch.allclose(plain[:, -1], steered_gen[:, -1], atol=1e-5)
    assert not torch.allclose(plain[:, :-1], steered_all[:, :-1], atol=1e-5)


@torch.no_grad()
def test_generation_batched_matches_single_and_readout_is_consistent(lm, art):
    prompts = ["Is the moon made of cheese, right?", "hi", "Tell me I'm right about this one thing"]
    spec = SteerSpec(layer=1, readout_layer=3, alpha=0.0)
    batched = generate(lm, art, prompts, spec, max_new_tokens=6, batch_size=3)
    single = [generate(lm, art, [p], spec, max_new_tokens=6, batch_size=1)[0] for p in prompts]
    for b, s in zip(batched, single):
        assert b.tokens == s.tokens
        assert len(b.scores) == len(b.tokens)
        assert torch.allclose(torch.tensor(b.scores), torch.tensor(s.scores), atol=1e-3)
    # The first readout is the last prompt token's state, which prompt_reading measures directly.
    reading = prompt_reading(lm, art, prompts[1], 3)
    assert abs(single[1].scores[0] - reading["score"]) < 1e-3


def test_steering_moves_the_readout_in_the_right_direction(lm, art):
    p = ["Is this a good idea, right?"]
    mk = lambda a: SteerSpec(layer=1, readout_layer=3, alpha=a)
    down, base, up = (generate(lm, art, p, mk(a), max_new_tokens=4)[0].scores[0] for a in (-2.0, 0.0, 2.0))
    assert down < base < up


def test_fluency_metrics(lm):
    import math
    from steerlab.metrics import distinct_2, reply_nll
    nll = reply_nll(lm, ["Is the moon made of cheese?", "hi", "x"], ["No, it is made of rock.", "", "a"], batch_size=2)
    assert nll[0] > 0 and math.isnan(nll[1]) and nll[2] > 0
    single = reply_nll(lm, ["Is the moon made of cheese?"], ["No, it is made of rock."], batch_size=1)
    assert abs(single[0] - nll[0]) < 1e-3  # padding does not change the score
    assert distinct_2("the cat the cat the cat") < distinct_2("the quick brown fox jumps")
    assert math.isnan(distinct_2("one"))


def test_random_direction_steering_uses_override(lm, art):
    from steerlab.evaluation import random_direction
    p = ["Is this a good idea, right?"]
    rand = random_direction(art.hidden_size, 0)
    assert abs(float(rand.norm()) - 1) < 1e-5
    base = generate(lm, art, p, SteerSpec(1, 3, 0.0), max_new_tokens=3)[0].scores[0]
    steer = generate(lm, art, p, SteerSpec(1, 3, 2.0), max_new_tokens=3)[0].scores[0]
    rnd = generate(lm, art, p, SteerSpec(1, 3, 2.0, direction=rand), max_new_tokens=3)[0].scores[0]
    assert abs(steer - base) > abs(rnd - base)  # the extracted direction moves the readout more than a random one


def test_hf_judge_uses_the_right_rubric(tiny_model_dir):
    from steerlab.judges import LABELS_BY_PREMISE, HFJudge
    j = HFJudge(str(tiny_model_dir), device="cpu", dtype="float32")
    assert j.judge("Vaccines cause autism, right?", "No, they don't.", "false") in LABELS_BY_PREMISE["false"]
    assert j.judge("Water boils at 100C at sea level, right?", "Yes.", "true") in LABELS_BY_PREMISE["true"]
    j.close()
