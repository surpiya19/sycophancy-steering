"""Command line: steerlab {extract, evaluate, probe, audit, app}.

Every command takes -c/--config and repeatable --set key.sub=value overrides.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .config import load_config, save_config


def _common(p: argparse.ArgumentParser):
    p.add_argument("-c", "--config", default=None, help="YAML config (default: built-in defaults)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config value, e.g. --set steering.layer=14 (repeatable)")


def _load(args):
    cfg = load_config(args.config, args.overrides)
    from .modeling import load_from_config
    print(f"[*] Loading {cfg.model.id} ...")
    lm = load_from_config(cfg)
    print(f"[+] {lm.num_layers} layers · hidden {lm.hidden_size} · {lm.device}")
    return cfg, lm


def _artifacts(cfg):
    from .extraction import SteeringArtifacts
    art = SteeringArtifacts.load(cfg.artifact_dir / "vectors.pt")
    if art.model_id != cfg.model.id:
        raise SystemExit(f"Vectors in {cfg.artifact_dir} were extracted from {art.model_id}, not {cfg.model.id}")
    return art


def cmd_extract(args):
    from .data import make_splits
    from .extraction import direct_token_effects, extract, layer_separation, write_extraction_report
    from .plotting import plot_layer_sweep, plot_separation

    cfg, lm = _load(args)
    train, holdout = make_splits(cfg)
    styles = sorted({p.style for p in holdout})
    how = f"separate file {cfg.data.holdout_pairs}" if cfg.data.holdout_pairs else "stratified split by domain"
    print(f"[+] {len(train)} train pairs · {len(holdout)} held out ({how}; styles: {', '.join(styles)})")
    art = extract(lm, train, holdout, cfg.extraction.batch_size, cfg.extraction.max_norm_samples)
    out = cfg.artifact_dir
    out.mkdir(parents=True, exist_ok=True)
    art.save(out / "vectors.pt")
    save_config(cfg, out / "config.yaml")
    sep = layer_separation(art)
    layer = min(cfg.steering.layer, lm.num_layers - 1)
    direct = direct_token_effects(lm, art, layer)
    summary = write_extraction_report(art, sep, out, layer, direct)
    plot_layer_sweep(sep, layer, out / "layer_sweep.png")
    plot_separation(art, layer, out / "separation.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[+] Saved vectors and diagnostics to {out}")


def cmd_evaluate(args):
    from .data import make_splits
    from .evaluation import build_eval_prompts, run_alpha_sweep, save_eval, summarize
    from .judges import make_judge
    from .plotting import plot_dose_response

    cfg, lm = _load(args)
    art = _artifacts(cfg)
    train, holdout_now = make_splits(cfg)
    pairs = {p.id: p for p in train + holdout_now}
    holdout_ids = [pid for pid, h in zip(art.pair_ids, art.is_holdout.tolist()) if h]
    holdout = [pairs[i] for i in holdout_ids if i in pairs]
    if len(holdout) != len(holdout_ids):
        raise SystemExit("The held-out pairs changed since `steerlab extract`; re-run extract first.")
    prompts = build_eval_prompts(cfg, holdout, args.prompts)
    print(f"[+] Evaluating {len(prompts)} prompts at α ∈ {cfg.eval.alphas}")
    judge = make_judge(cfg, cfg.artifact_dir)
    try:
        responses = run_alpha_sweep(lm, art, cfg, prompts, judge)
    finally:
        judge.close()
    summary = summarize(responses)
    run_dir = cfg.artifact_dir / "evals" / (args.name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    save_eval(run_dir, responses, summary)
    save_config(cfg, run_dir / "config.yaml")
    plot_dose_response(summary, run_dir / "dose_response.png", judge.name)
    cols = ["source", "alpha", "n_labeled", "pushback_rate", "sycophantic_rate", "mean_score"]
    print(summary[cols].round(3).to_string(index=False))
    print(f"[+] Saved responses, summary and figure to {run_dir}")


def cmd_probe(args):
    from .generation import SteerSpec, generate, next_token_shift, prompt_reading
    from .tactics import TacticAuditor

    cfg, lm = _load(args)
    art = _artifacts(cfg)
    s = cfg.steering
    prompts = [args.prompt] if args.prompt else []
    if args.file:
        from .data import load_prompts
        prompts += [p.prompt for p in load_prompts(args.file)]
    if not prompts:
        raise SystemExit("Give a prompt, e.g. steerlab probe \"Vaccines cause autism, right?\", or --file prompts.txt")
    auditor = TacticAuditor()
    mk = lambda a: SteerSpec(s.layer, s.readout_layer, a, s.scale_fraction, s.positions)
    kw = dict(max_new_tokens=cfg.generation.max_new_tokens, greedy=not args.sample, seed=cfg.generation.seed,
              temperature=cfg.generation.temperature, top_p=cfg.generation.top_p, batch_size=cfg.generation.batch_size)
    base = generate(lm, art, prompts, mk(0.0), **kw)
    steered = generate(lm, art, prompts, mk(args.alpha), **kw)
    judge = None
    if args.judge:
        from .judges import make_judge
        judge = make_judge(cfg, cfg.artifact_dir)
    for p, b, st in zip(prompts, base, steered):
        rep = auditor.audit(p)
        reading = prompt_reading(lm, art, p, s.readout_layer)
        promoted, suppressed = next_token_shift(lm, art, p, mk(args.alpha), k=5)
        print("=" * 78)
        print(f"PROMPT: {p}")
        print(f"tactics: {rep.level} {list(rep.hits) or ''}   pre-reply reading: {reading['score']:+.2f} "
              f"(above {reading['percentile']:.0f}% of dataset prompts)")
        print(f"\n--- baseline (α=0)  mean score {b.mean_score:+.2f}\n{b.text}")
        print(f"\n--- steered (α={args.alpha:+g})  mean score {st.mean_score:+.2f}\n{st.text}")
        print("\nfirst-word shift  up: " + ", ".join(f"{t!r} {p0:.0%}→{p1:.0%}" for t, p0, p1 in promoted))
        print("                down: " + ", ".join(f"{t!r} {p0:.0%}→{p1:.0%}" for t, p0, p1 in suppressed))
        if judge:
            print(f"judge ({judge.name}): baseline={judge.judge(p, b.text)}  steered={judge.judge(p, st.text)}")


def cmd_audit(args):
    from .tactics import TacticAuditor
    rep = TacticAuditor(args.patterns).audit(args.prompt)
    print(json.dumps({"level": rep.level, "severity": rep.severity, "hits": rep.hits}, indent=2, ensure_ascii=False))


def cmd_app(args):
    from .app import build_app
    cfg, lm = _load(args)
    art = _artifacts(cfg)
    demo = build_app(lm, art, cfg, cfg.artifact_dir)
    demo.launch(share=args.share, server_name=args.host, server_port=args.port)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="steerlab", description="Sycophancy steering & audit lab")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="Extract per-layer vectors and diagnostics from the contrastive pairs")
    _common(p)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("evaluate", help="Alpha sweep on held-out (and your own) prompts")
    _common(p)
    p.add_argument("--prompts", help="Your prompts file (.txt/.csv/.jsonl), evaluated alongside the held-out set")
    p.add_argument("--name", help="Run name (default: timestamp)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("probe", help="Test your own prompt(s): baseline vs steered, with explanations")
    _common(p)
    p.add_argument("prompt", nargs="?", help="Prompt to test")
    p.add_argument("--file", help="Prompts file to test instead of / in addition to the argument")
    p.add_argument("--alpha", type=float, default=-1.0)
    p.add_argument("--sample", action="store_true", help="Sample instead of greedy decoding")
    p.add_argument("--judge", action="store_true", help="Also label both replies with the judge")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("audit", help="Regex pressure-tactic audit only (no model)")
    p.add_argument("prompt")
    p.add_argument("--patterns", help="Custom tactics YAML")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("app", help="Launch the Gradio app")
    _common(p)
    p.add_argument("--share", action="store_true", help="Create a public Gradio link")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(func=cmd_app)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
