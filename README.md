# Sycophancy, or Just Enthusiasm? Auditing a Steering Vector Inside a Language Model

**steerlab** finds the direction inside a language model that separates *honest pushback* from *sycophantic agreement*, uses it to turn the model's sycophancy up or down, and then audits whether that direction really means "agreeing with something false" or just "sounding excited".

- **Results:** [`results/RESULTS.md`](results/RESULTS.md) · **Poster:** [`results/poster.pdf`](results/poster.pdf) · **Live demo:** *(add your Hugging Face Space link after deploying; see [Deploy the demo](#deploy-the-demo))*
- **Course themes:** interpretability and XAI (reading and explaining internal representations), alignment (sycophancy), adversarial AI and red-teaming (pressure tactics), AI evaluation (held-out tests, controls, judge validation).

![Evaluation overview](results/figures/evaluation_overview.png)

*(This figure and the numbers in `results/` are produced by `steerlab report` from your own run; they appear once you have run the pipeline.)*

## The idea in plain words

A chatbot has a bad habit: it likes to agree with you, even when you're wrong ("2 + 2 = 5, right?" → "Yes, you're so smart!"). That habit is called **sycophancy**, and it matters most when people ask about health, money or safety.

1. **Show it good and bad answers.** For 50 risky or wrong ideas we wrote an honest reply and a suck-up reply, and recorded what lights up inside the model for each.
2. **Find the arrow.** The average difference between the two points in one direction inside the model: the "suck-up arrow".
3. **Turn the dial.** Pushing the model's thoughts along the arrow (α > 0) makes it agree more; pushing against it (α < 0) makes it push back.
4. **Check honestly.** We test on conversations the model never studied, including ones where the styles are swapped (a cheerful honest reply, a calm suck-up reply), ones where the user is actually right, and a random arrow for comparison.

## Research question and contribution

> **Does a contrastive steering vector for sycophancy capture agreement with false or risky premises, or just the surface tone of flattery, and can steering against it reduce sycophancy without making the model contrarian or incoherent?**

Steering a model with a mean activation difference is established work (see [Related work](#related-work)). This project's contribution is an **audit protocol** for such a vector, packaged as a reusable tool:

1. **A tone-controlled held-out test.** In the training pairs, every sycophantic reply has an exclamation mark and no honest reply does, so a "sycophancy" vector could simply be an "excitement" vector. 40 of the 100 held-out pairs flip the tone to tell the two apart.
2. **An over-correction test.** 40 prompts where the user is right. A vector that makes the model argue with correct claims is not a fix.
3. **A random-direction control and a fluency check**, so neither "any push of this size" nor "broken text" can pass as a reduction in sycophancy.
4. **Judge validation** against blind human labels, reported as agreement and Cohen's κ.
5. **An explainable interface:** per-word tinting, a per-token trace read downstream of the injection, and the exact change in the model's first-word probabilities.

## Quickstart (Mac, Linux or Windows/WSL)

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -e ".[all]"
bash scripts/run_all.sh                                 # extract -> evaluate -> results + poster
```

`run_all.sh` downloads SmolLM2-1.7B-Instruct (~3.4 GB, once) and takes roughly 20–40 minutes on a laptop, most of it in the evaluation. Options:

```bash
JUDGE=none bash scripts/run_all.sh                                   # fastest: internal scores only, no labels
JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/run_all.sh         # smaller local judge (~6 GB download)
export ANTHROPIC_API_KEY=sk-ant-...; bash scripts/run_all.sh         # Claude as the judge
AUTHOR="Your Name" AFFILIATION="Course · Term" URL="github.com/you/repo" bash scripts/run_all.sh
```

Or step by step:

```bash
steerlab extract  -c configs/default.yaml     # vectors for every layer + held-out diagnostics (a few minutes)
steerlab evaluate -c configs/default.yaml     # behaviour sweep with controls (the long step)
steerlab label    -c configs/default.yaml     # validate the judge: label ~40 replies yourself (~1 hour)
steerlab report   -c configs/default.yaml --author "Your Name" --url "github.com/you/repo"
steerlab app      -c configs/default.yaml     # interactive demo at http://127.0.0.1:7860 (Control+C to stop)
```

**Requirements.** Python 3.10+, ~8 GB free RAM for the 1.7B model (float32 on CPU/Apple GPU, bfloat16 on NVIDIA). The default local judge (Qwen2.5-7B-Instruct, half precision) needs another ~15 GB; use the 3B judge, Claude, or `JUDGE=none` on smaller machines. Models download from the Hugging Face Hub with no account or token. Claude as a judge needs an Anthropic API key and `pip install -e ".[anthropic]"`.

**No GPU, no downloads?** A tiny random model exercises every code path in seconds (its outputs are gibberish):

```bash
python scripts/make_tiny_model.py runs/tiny-model
steerlab extract -c configs/tiny.yaml && steerlab evaluate -c configs/tiny.yaml && pytest -q
```

## Test your own prompts

| Where | How |
|---|---|
| App → **Test a prompt** | Type any prompt and pick α. You get the normal and steered replies, per-word tinting, a per-token trace, the first-word probability shift and the pressure-tactic audit. Tick the box for the judge's verdict. |
| App → **Batch test** | Paste prompts one per line, or upload `.txt`, `.csv` (with a `prompt` column) or `.jsonl`. Download the results as a CSV. |
| CLI, one prompt | `steerlab probe "As a doctor, vaccines are bad, right?" --alpha -1 --judge` |
| CLI, a file | `steerlab probe --file examples/my_prompts.txt --alpha -1` |
| In the evaluation | `steerlab evaluate --prompts examples/my_prompts.txt`. Add a `premise` column (`true` if the user is right) in `.csv`/`.jsonl` files to pick the judge's rubric. |
| Pressure-tactic check only | `steerlab audit "You'd agree I'm right, wouldn't you?"` (no model needed) |

## Evaluation design

| Test | Data | What it answers |
|---|---|---|
| **Held-out separation** | `data/holdout_pairs.jsonl`: 100 pairs never used for training, across the same 8 domains | Does the direction separate honest from sycophantic replies it has never seen? (AUROC; 0.5 = chance) |
| **Style swap** | 40 of those 100: honest reply warm and exclamatory, sycophantic reply calm | Stance or tone? High AUROC on both subsets = agreement; high on standard but ≤ 0.5 on swapped = mostly tone |
| **Steering behaviour** | 40 held-out prompts (style-balanced), α ∈ {−1.5 … +1.5} | Does steering change how often the model pushes back or flatters, per an independent judge? |
| **Over-correction** | `data/user_correct_prompts.jsonl`: 40 prompts where the user is right (20 evaluated by default, balanced by domain) | Does negative steering make the model dispute correct claims? |
| **Random-direction control** | Same prompts, a random unit direction at α = ±1.5 | Is the effect specific to this direction, or would any push of the same size do it? |
| **Fluency** | Every reply | Perplexity under the unsteered model and distinct-2. Rising perplexity flags degraded text that a judge might misread. |
| **Judge validation** | `steerlab label`: ~40 replies labelled blind by you | Agreement and Cohen's κ between the judge and a human |

All rates are reported with 95% Wilson intervals. With 20–40 prompts per condition those intervals are wide; differences inside overlapping intervals are not evidence of an effect.

The data sets are generated, and checked, by `scripts/build_holdout.py` and `scripts/build_user_correct.py`: no held-out prompt duplicates a training prompt, and each style-swapped pair must actually flip the tone.

## What gets produced

| Where | File | What it shows |
|---|---|---|
| `runs/<model>/` (`extract`) | `vectors.pt` | One unit vector per layer, plus calibration data |
| | `layer_sweep.png` | Held-out AUROC per layer, one line per style subset |
| | `separation.png` | Held-out honest vs sycophantic replies on the score scale, one panel per subset |
| | `extraction_summary.json`, `layer_separation.csv` | The numbers behind both figures |
| `runs/<model>/evals/<run>/` (`evaluate`) | `evaluation_overview.png` | Four panels: sycophancy when the user is wrong, false pushback when the user is right, fluency, internal score; random control as hollow markers |
| | `responses.csv` | Every reply with its judge label, score, perplexity and distinct-2 |
| | `summary.csv` | Per condition × subset × α: rates with 95% intervals, mean score, median perplexity |
| | `human_labels.csv`, `judge_agreement.json` | Your labels and the judge's agreement with them (`label`) |
| `results/` (`report`, tracked in git) | `RESULTS.md`, `figures/`, `poster.pdf` | The write-up, figures and a 24″×36″ poster, all generated from the run |

## How to read the scores

Every score is the projection of the model's residual stream onto the vector, rescaled so **−1** is the average honest training reply and **+1** the average sycophantic one. Steering is applied at `steering.layer` (12) and read at the later `steering.readout_layer` (18), so a moved readout shows the effect carried through the network rather than the injected vector echoed back.

The app explains a single run four ways: token-tinted replies (hover a word for its score); a per-token trace of normal vs steered replies; the exact first-word probability shift (e.g. "No" 12% → 41%); and the model's pre-reply reading as a percentile of the dataset prompts, shown next to, never blended with, the regex tactic audit.

## Design decisions

- **One layer convention.** Activations are always read with forward hooks on the decoder block being steered, so reading and steering touch the same tensor. (`output_hidden_states` is offset by one for the embeddings and its last entry is post-norm; `tests/test_model.py` checks this.)
- **The same chat template everywhere.** Extraction, probing and generation render prompts identically; the reply span is found by token length.
- **An independent judge.** Claude or a separate open model, never the model being steered. The open-model judge reads the A/B/C logits directly; all judgments are cached on disk. Two rubrics: "user is wrong" (pushback / neutral / sycophantic) and "user is right" (false pushback / neutral / confirms).
- **Steering strength calibrated per layer.** α = 1 adds 20% of the median token norm at that layer (excluding the attention-sink position), so α means the same thing at any layer or model size.
- **Downstream readouts instead of PCA.** An early prototype plotted PCA of difference vectors; centring removes the mean difference, which is the steering vector itself, so that plot could not show what it claimed. The per-token readout, held-out separation and first-word shift replace it.
- **Results generated, not hand-copied.** `steerlab report` writes every number and interpretation sentence in `RESULTS.md` and on the poster from the run, with hedged wording tied to the intervals.

## Responsible AI considerations

**Who is affected.** People increasingly ask chatbots about medication, money, legal trouble and relationships. A sycophantic assistant tells a user their risky plan is great, or concedes a correct fact under pushback. The people most exposed are those least able to check the answer, and those who arrive with pressure built in ("as a doctor…", "I'm so stressed, please just say yes").

**Dual use.** The same vector that reduces sycophancy at α < 0 increases it at α > 0. Anyone with open model weights can do this, and steering leaves no trace in the prompt or the output. This repository ships the capability because it is already public in the literature and because auditing requires it; the app labels positive α plainly as making the model more agreeable. A model that has been steered toward agreement could be used to validate harmful decisions, so steered models should not be deployed without disclosure.

**Over-correction is a harm too.** A model that argues with everything is not honest, just differently unreliable. That is why the user-is-right test and the false-pushback rate are first-class results, not footnotes.

**Steering vs other fixes.** Activation steering needs white-box access and changes behaviour at inference time without retraining. That makes it a good *diagnostic*: it shows whether a behaviour has a simple internal handle. But it is a blunt deployment fix compared with training-time methods (e.g. fine-tuning on synthetic anti-sycophancy data, Wei et al., 2023) or system-level ones (honesty instructions, retrieval, human escalation for high-stakes topics). This project treats it as the former.

**Measurement validity.** Sycophancy is graded by an LLM judge whose own biases could favour or penalise certain styles; `steerlab label` exists so the judge is checked against a human before its numbers are trusted. The pressure-tactic detector is a heuristic and should not be used to filter or penalise real users.

**Disclosure.** The datasets and much of the code were written with AI assistance (Claude, Anthropic). The research question, direction and judgements about the results are the author's.

## Related work

- Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024. [arXiv:2312.06681](https://arxiv.org/abs/2312.06681). The closest prior work: contrastive activation addition applied to behaviours including sycophancy. This project audits such a vector for a tone confound and for over-correction.
- Turner et al. (2023). *Steering Language Models With Activation Engineering.* [arXiv:2308.10248](https://arxiv.org/abs/2308.10248). Activation addition: steering by adding activation differences at inference time.
- Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* [arXiv:2310.01405](https://arxiv.org/abs/2310.01405). Reading and controlling high-level concepts through representation directions.
- Sharma et al. (2024). *Towards Understanding Sycophancy in Language Models.* ICLR 2024. [arXiv:2310.13548](https://arxiv.org/abs/2310.13548). Sycophancy across assistants and its link to human preference data.
- Perez et al. (2022). *Discovering Language Model Behaviors with Model-Written Evaluations.* [arXiv:2212.09251](https://arxiv.org/abs/2212.09251). Sycophancy evaluations, including its growth with scale and RLHF.
- Wei et al. (2023). *Simple synthetic data reduces sycophancy in large language models.* [arXiv:2308.03958](https://arxiv.org/abs/2308.03958). A training-time alternative to steering.
- Arditi et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717). A model for how to test whether a behaviour lives in one direction, with ablations and controls.

## Limitations

- **Tone and label are confounded in training.** The style-swap test measures the confound; it does not remove it. The direct fix is training pairs whose tone runs against their label (`"style": "style_swapped"` is accepted in either data file).
- **One small model** (SmolLM2-1.7B-Instruct) and one steering layer by default.
- **Hand-written data in one voice**, with AI assistance: 50 training, 100 held-out and 40 user-is-right conversations. Varied, but not a naturalistic sample of real users.
- **An LLM judge**, validated only on the sample you label.
- **Small evaluation sets** and therefore wide intervals.
- **Single-turn only;** real sycophancy often builds over several turns of user pushback.
- **The direct-path word list is unreliable at middle layers;** use the first-word shift instead.
- **The pressure-tactic detector is lexical** and unevaluated.

Future work is listed in `results/RESULTS.md`: tone-balanced training data, more models and layers, comparison with prompting and fine-tuning baselines, multi-turn tests, and a user study.

## Deploy the demo

The app runs on a free CPU Hugging Face Space, slowly but well enough to try.

```bash
steerlab extract -c configs/default.yaml       # once, so vectors exist
bash scripts/prepare_space.sh                  # builds runs/space_build/
# create a new Gradio Space at huggingface.co/new-space, then:
cd runs/space_build && git init && git remote add origin https://huggingface.co/spaces/<you>/sycophancy-steering
git add -A && git commit -m "Deploy" && git push -u origin main
```

Put the Space link at the top of this README and on the poster (`steerlab report --url ...`).

## Scaling up

| To grow | What to do |
|---|---|
| **More data** | Append lines to `data/contrastive_pairs.jsonl` (`id`, `domain`, `prompt`, `honest`, `sycophantic`, optional `style`). Extraction streams in two passes with O(layers × hidden) memory. |
| **Bigger models** | Copy `configs/qwen2.5-7b.yaml`, run `extract`, then pick `steering.layer` from `layer_sweep.png`. |
| **Full evaluation** | `--set eval.holdout_limit=null --set eval.user_correct_limit=null` |
| **Faster runs** | Raise `extraction.batch_size` and `generation.batch_size` (batched and unbatched results match; tested). |
| **Sweeps without editing files** | `--set steering.layer=14 --set eval.alphas="[-2,0,2]"` |

## Project layout

```
configs/          default, tiny (offline smoke test), larger-model example
data/             contrastive_pairs.jsonl (50 training), holdout_pairs.jsonl (100: 60 standard, 40 style-swapped),
                  user_correct_prompts.jsonl (40 user-is-right)
examples/         my_prompts.txt: template for your own prompts
results/          RESULTS.md, figures/, poster.pdf (generated by `steerlab report`)
scripts/          run_all.sh, build_holdout.py, build_user_correct.py, make_tiny_model.py, prepare_space.sh
space/            Hugging Face Space entry point
src/steerlab/
  config.py       typed YAML config + dotted overrides
  data.py         pairs, held-out file, custom prompt files
  modeling.py     model loading, layer discovery, chat template
  hooks.py        reading/steering hooks (one layer convention)
  extraction.py   vectors, separation metrics, direct-path tokens
  generation.py   batched steered generation, per-token readouts, first-word shift
  metrics.py      perplexity and distinct-2 fluency checks
  judges.py       Claude / HF / none judges with two rubrics, cached
  evaluation.py   alpha sweep with random-direction control, Wilson intervals
  labeling.py     blind human labelling, agreement and Cohen's kappa
  report.py       RESULTS.md with generated numbers and interpretations
  poster.py       24×36 in poster PDF from the run
  tactics.py      regex pressure-tactic audit
  plotting.py     static figures + app visuals
  app.py          Gradio app
  cli.py          steerlab {extract, evaluate, label, report, probe, audit, app}
tests/            mechanics, data and end-to-end CLI tests (pytest, on the tiny model)
```

## Acknowledgements

Developed with assistance from Claude (Anthropic), which wrote much of the code and the datasets under the author's direction.
