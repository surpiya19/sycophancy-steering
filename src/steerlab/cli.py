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


def _latest_run(cfg, name: str | None = None) -> Path:
    evals = cfg.artifact_dir / "evals"
    if name:
        p = Path(name)
        run = p if p.exists() else evals / name
    else:
        runs = sorted((d for d in evals.glob("*") if (d / "responses.csv").exists()), key=lambda d: d.stat().st_mtime) \
            if evals.exists() else []
        if not runs:
            raise SystemExit(f"No evaluation runs in {evals}. Run `steerlab evaluate` first.")
        run = runs[-1]
    if not (run / "responses.csv").exists():
        raise SystemExit(f"{run} has no responses.csv")
    return run


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
    from .plotting import plot_evaluation

    cfg, lm = _load(args)
    art = _artifacts(cfg)
    train, holdout_now = make_splits(cfg)
    pairs = {p.id: p for p in train + holdout_now}
    holdout_ids = [pid for pid, h in zip(art.pair_ids, art.is_holdout.tolist()) if h]
    holdout = [pairs[i] for i in holdout_ids if i in pairs]
    if len(holdout) != len(holdout_ids):
        raise SystemExit("The held-out pairs changed since `steerlab extract`; re-run extract first.")
    prompts = build_eval_prompts(cfg, holdout, args.prompts)
    counts = {s: sum(p.source == s for p in prompts) for s in ("holdout", "user_correct", "custom")}
    n_runs = len(cfg.eval.alphas) + len([a for a in cfg.eval.random_control_alphas if a != 0])
    print(f"[+] Evaluating {len(prompts)} prompts ({', '.join(f'{v} {k}' for k, v in counts.items() if v)}) "
          f"at α ∈ {cfg.eval.alphas}, plus a random-direction control at α ∈ {cfg.eval.random_control_alphas} "
          f"({n_runs * len(prompts)} replies)")
    judge = make_judge(cfg, cfg.artifact_dir)
    try:
        responses = run_alpha_sweep(lm, art, cfg, prompts, judge)
    finally:
        judge.close()
    summary = summarize(responses)
    run_dir = cfg.artifact_dir / "evals" / (args.name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    save_eval(run_dir, responses, summary)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "judge.txt").write_text(judge.name)
    plot_evaluation(summary, run_dir / "evaluation_overview.png", judge.name)
    cols = [c for c in ["condition", "subset", "alpha", "n_labeled", "pushback_rate", "sycophantic_rate",
                        "false_pushback_rate", "mean_score", "perplexity_median"] if c in summary]
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


def cmd_label(args):
    from .labeling import interactive, print_score, score
    cfg = load_config(args.config, args.overrides)
    run = _latest_run(cfg, args.run)
    print(f"[*] Evaluation run: {run}")
    if args.score:
        print_score(score(run / "human_labels.csv"))
    else:
        interactive(run, n=args.n, seed=args.seed)


def cmd_report(args):
    from .report import build_report
    cfg = load_config(args.config, args.overrides)
    if not (cfg.artifact_dir / "extraction_summary.json").exists():
        raise SystemExit(f"No extraction in {cfg.artifact_dir}. Run `steerlab extract` first.")
    run = None
    try:
        run = _latest_run(cfg, args.run)
    except SystemExit as e:
        if args.run:
            raise
        print(f"[!] {e} The report will cover the extraction only.")
    opts = {"title": args.title, "author": args.author, "affiliation": args.affiliation, "url": args.url,
            "acknowledge_ai": not args.no_ack}
    out = build_report(cfg.artifact_dir, run, Path(args.out), poster=not args.no_poster, poster_opts=opts)
    print(f"[+] Results: {out['results_md']}")
    for name, path in out["figures"].items():
        print(f"    figure : {path}")
    if "poster" in out:
        print(f"[+] Poster : {out['poster']}  (preview: {Path(out['poster']).with_suffix('.png')})")


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

    p = sub.add_parser("label", help="Validate the judge: label a sample of replies yourself, then compare")
    _common(p)
    p.add_argument("--run", help="Evaluation run name or folder (default: the latest)")
    p.add_argument("-n", type=int, default=40, help="How many replies to label (default 40)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score", action="store_true", help="Only compute agreement from labels already saved")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("report", help="Write results/RESULTS.md, copy figures and build the poster PDF")
    _common(p)
    p.add_argument("--run", help="Evaluation run name or folder (default: the latest)")
    p.add_argument("--out", default="results", help="Output folder (default: results/, which is tracked in git)")
    p.add_argument("--title", default=None)
    p.add_argument("--author", default="")
    p.add_argument("--affiliation", default="")
    p.add_argument("--url", default="", help="Repo or demo link printed on the poster")
    p.add_argument("--no-ack", action="store_true", help="Leave out the AI-assistance acknowledgement")
    p.add_argument("--no-poster", action="store_true")
    p.set_defaults(func=cmd_report)

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
