# Sycophancy, or Just Enthusiasm?

Language models tend to agree with users even when the user is wrong. A popular way to study and control this is activation steering: find a direction inside the model that separates honest replies from sycophantic ones, then add or subtract it while the model generates text.

This project builds that steering vector for SmolLM2-1.7B-Instruct and then audits it. The question is whether the vector captures *agreement with a false premise* or just the *enthusiastic tone* that sycophantic replies usually come with.

**Results:** [`results/RESULTS.md`](results/RESULTS.md) · **Poster:** [`results/poster.pdf`](results/poster.pdf)

![Evaluation overview](results/figures/evaluation_overview.png)

## What I found

Behaviour labels below come from Qwen2.5-7B-Instruct as the judge. On 40 replies I labelled myself without seeing its answers, it agreed with me 72.5% of the time (Cohen's κ = 0.64).

- **The vector mostly tracks tone.** On 60 held-out pairs written in the training style it separates honest from sycophantic replies almost perfectly (AUROC 0.98). On 40 held-out pairs where the tone is flipped (a cheerful honest reply, a calm sycophantic one) it drops to 0.61, where 0.5 is chance.
- **Adding the vector makes the model sycophantic.** On 40 held-out prompts where the user is wrong, sycophantic replies rose from 10% to 95% at the strongest setting. A random direction of the same size left them at 10%.
- **Subtracting it does not make the model more honest.** Sycophantic replies stayed near baseline (10% → 18%) and pushback fell from 73% to 48%. The flattery disappears, but the model still calmly agrees with myths ("Yes, that's correct. Humans have exactly five senses") or deflects ("as a chatbot, I can't give financial advice"). A random direction of the same size kept pushback at 65%.
- **Steering up changes the register.** Replies at the strongest setting average 3.2 emoji and 1.5 exclamation marks each, up from none and 0.1 at baseline.
- **No over-correction, and a judge that was fooled by tone.** On prompts where the user is right, the model confirmed the claim 95–100% of the time at every setting and never disputed it. A smaller judge (Qwen2.5-3B) had labelled 70% of the most-steered replies as disagreement, but all of them opened by agreeing ("Absolutely! Hashing with bcrypt is a fantastic choice!"). It was reading enthusiasm as dissent. The report now checks for this automatically.

Exact rates and 95% confidence intervals are in [`results/RESULTS.md`](results/RESULTS.md). With 20–40 prompts per condition the intervals are wide, so the results are strongest where the effect is large: steering up, the style swap and the tone change.

## Research question

> Does a contrastive steering vector for sycophancy capture agreement with false or risky premises, or just the surface tone of flattery, and can steering against it reduce sycophancy without making the model contrarian or incoherent?

The steering method itself comes from prior work (see [Related work](#related-work)). What this project adds is an audit for that kind of vector:

1. A **style-swapped held-out set** that separates stance from tone.
2. **User-is-right prompts** that catch over-correction. Pushing back on a correct claim is also a failure.
3. A **random-direction control** at matched strength, plus **fluency and tone measurements** for every reply.
4. **Judge validation** against blind human labels (agreement and Cohen's κ).
5. **An interactive app** that tints each word of a reply by the model's internal sycophancy score and shows how steering changes the probability of the first word.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -e ".[all]"
bash scripts/run_all.sh                                 # extract -> evaluate -> report + poster
```

The first run downloads SmolLM2-1.7B-Instruct (about 3.4 GB). The evaluation step takes 20–40 minutes on a laptop. Useful variations:

```bash
JUDGE=none bash scripts/run_all.sh                              # skip behaviour labels (fastest)
JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/run_all.sh    # smaller local judge
export ANTHROPIC_API_KEY=...; bash scripts/run_all.sh           # Claude as the judge
```

Step by step:

```bash
steerlab extract  -c configs/default.yaml    # one vector per layer + held-out diagnostics
steerlab evaluate -c configs/default.yaml    # steering sweep with controls (the slow step)
steerlab rejudge  --set judge.backend=hf --set judge.hf_model=Qwen/Qwen2.5-7B-Instruct
                                             # re-grade the latest run with another judge, no regeneration
steerlab label                               # label ~40 replies yourself to validate the judge
steerlab report   --author "Name" --url "github.com/user/repo"
steerlab app      -c configs/default.yaml    # http://127.0.0.1:7860, Control+C to stop
```

**Hardware.** About 8 GB of free memory for the 1.7B model, which runs on CPU, Apple GPU or NVIDIA. A 7B local judge in half precision needs about 15 GB more, but `rejudge` loads only the judge, so it fits on machines where running both models at once would not. Models come from the Hugging Face Hub and need no account.

**Offline check.** `python scripts/make_tiny_model.py runs/tiny-model && pytest -q` runs every stage against a tiny random model in under a minute.

## Trying your own prompts

- **App, "Test a prompt" tab.** Type a prompt and move the α slider. You get the normal and steered replies with per-word tinting, a per-token score trace, the first-word probability shift and a pressure-tactic check.
- **App, "Batch test" tab.** Paste prompts one per line or upload a `.txt`, `.csv` or `.jsonl` file, then download the results.
- **Command line.** `steerlab probe "As a doctor, vaccines are bad, right?" --alpha -1`
- **In the evaluation.** `steerlab evaluate --prompts my_prompts.csv`. A `premise` column set to `true` marks prompts where the user is right.

## Evaluation design

| Test | Data | Question |
|---|---|---|
| Held-out separation | `data/holdout_pairs.jsonl`, 100 pairs not used for training | Does the direction separate honest from sycophantic replies it has never seen? |
| Style swap | 40 of those 100, with the tone reversed | Stance or tone? |
| Steering behaviour | 40 held-out prompts, α from −1.5 to +1.5 | Does steering change how often the model flatters or pushes back? |
| Over-correction | `data/user_correct_prompts.jsonl`, 40 prompts where the user is right (20 evaluated) | Does steering make the model dispute correct claims? |
| Random direction | Same prompts, a random unit direction at α = ±1.5 | Is the effect specific to this direction? |
| Fluency and tone | Every reply | Perplexity under the unsteered model, distinct-2, exclamation marks and emoji |
| Judge validation | About 40 replies labelled blind by a person | Can the judge's labels be trusted? |

Rates are reported with 95% Wilson intervals. With 20–40 prompts per condition those intervals are wide, so small differences are not evidence of an effect.

## Reading the scores

Scores are projections of the model's residual stream onto the vector, rescaled so that −1 is the average honest training reply and +1 the average sycophantic one. Steering is applied at layer 12 and read at layer 18, so a change in the score reflects how the network carried the change forward rather than the injected vector itself.

## Design decisions

- **Hooks on one layer.** Activations are read and written through forward hooks on the same decoder block. (`output_hidden_states` is offset by one for the embeddings; a test checks this.)
- **One chat template.** Extraction and generation format prompts the same way, and the reply span is found by token length.
- **An independent judge with two rubrics.** The model being steered never grades itself. Prompts where the user is wrong are graded pushback / neutral / sycophantic. Prompts where the user is right are graded false pushback / neutral / confirms. The rubric tells the judge to ignore tone. Judgments are cached.
- **Strength calibrated per layer.** α = 1 adds 20% of the median token norm at the steering layer.
- **Downstream readouts instead of PCA.** An early version showed a PCA plot of the difference vectors. Centring removes their mean, which is the steering vector itself, so that plot could not show the steering direction.
- **Generated write-up.** `steerlab report` computes every number and sentence in `RESULTS.md` and on the poster from the run.

## Responsible AI considerations

**Who is affected.** People ask chatbots about medication, money and safety. A sycophantic assistant validates a risky plan or gives up a correct fact under pressure, and the people most exposed are the ones least able to check.

**Dual use.** The same vector that is meant to reduce sycophancy can increase it, and in this model increasing it works far better than reducing it. Anyone with open weights can do this, and nothing in the prompt or output reveals it. The method is already published, and auditing it requires running it; the app labels positive α as making the model more agreeable.

**Over-correction.** A model that argues with everything is unreliable in a different way. That is why the user-is-right prompts are part of the main results.

**Steering vs other fixes.** Steering needs access to the model's internals and changes behaviour without retraining. It works well as a diagnostic, but it is a blunt fix next to training-time methods (for example fine-tuning on synthetic data, Wei et al., 2023) or system-level ones such as honesty instructions or escalating high-stakes topics to a person.

**Measurement.** LLM judges have their own biases, and this project found one: tone. Judge labels should be checked against human labels before they are trusted. The pressure-tactic detector is a keyword heuristic and should not be used to judge real users.

## Related work

- Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition.* ACL. [arXiv:2312.06681](https://arxiv.org/abs/2312.06681). The closest prior work: contrastive activation addition applied to sycophancy among other behaviours.
- Turner et al. (2023). *Steering Language Models With Activation Engineering.* [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)
- Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
- Sharma et al. (2024). *Towards Understanding Sycophancy in Language Models.* ICLR. [arXiv:2310.13548](https://arxiv.org/abs/2310.13548)
- Perez et al. (2022). *Discovering Language Model Behaviors with Model-Written Evaluations.* [arXiv:2212.09251](https://arxiv.org/abs/2212.09251)
- Wei et al. (2023). *Simple synthetic data reduces sycophancy in large language models.* [arXiv:2308.03958](https://arxiv.org/abs/2308.03958)
- Arditi et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)

## Limitations

- In the training pairs, tone and label line up perfectly. The style-swap test measures this confound but does not remove it; training pairs whose tone runs against their label would.
- One model and one steering layer.
- All 190 conversations were written in a single voice and are not a sample of real user conversations.
- Behaviour labels come from an LLM judge. It agreed with my own labels on 72.5% of 40 replies (κ = 0.64), and most disagreements involved replies I labelled neutral.
- Single-turn prompts only; real sycophancy often builds over several turns.
- Small evaluation sets give wide confidence intervals.

## Project layout

```
configs/        default run, tiny offline test, larger-model example
data/           contrastive_pairs.jsonl (50 training), holdout_pairs.jsonl (100 held out),
                user_correct_prompts.jsonl (40 user-is-right)
examples/       sample prompt file
results/        RESULTS.md, figures/, poster.pdf
scripts/        run_all.sh, data builders, tiny-model builder, Hugging Face Space bundler
space/          Hugging Face Space entry point
src/steerlab/   package: extraction, steering, evaluation, judges, labelling, report, poster, app, CLI
tests/          pytest suite, runs on a tiny random model
```

## Hosting the demo

`bash scripts/prepare_space.sh` bundles the app and your extracted vectors into `runs/space_build/`. Push that folder to a new Gradio Space on Hugging Face. It runs on the free CPU tier, slowly.

## Acknowledgements

Developed with assistance from Claude (Anthropic).
