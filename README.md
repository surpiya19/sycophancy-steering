# steerlab: Sycophancy Steering & Audit Lab

Find the direction in a language model's activations that separates **honest pushback** from **sycophantic agreement**. Then:

- **steer** generation with it, toward pushback or toward agreement
- **see**, token by token, how sycophantic the model's internal state is
- **measure** behaviour with an independent judge

It works with Hugging Face causal LMs whose decoder blocks it can locate: Llama, Mistral, Qwen, SmolLM, Gemma, Phi, GPT-2, GPT-NeoX and similar. Everything is driven by one YAML config.

## Install

```bash
pip install -e ".[app]"              # core + Gradio app
pip install -e ".[app,anthropic]"    # + Claude as the judge
pip install -e ".[all]"              # everything, including tests
```

## Quickstart

```bash
steerlab extract  -c configs/default.yaml        # vectors for every layer + diagnostics
steerlab evaluate -c configs/default.yaml        # alpha sweep on held-out prompts
steerlab app      -c configs/default.yaml        # interactive app at http://127.0.0.1:7860
```

To try it with no GPU and no downloads, use a tiny random model. Its outputs are gibberish, but every stage runs:

```bash
python scripts/make_tiny_model.py runs/tiny-model
steerlab extract -c configs/tiny.yaml && steerlab evaluate -c configs/tiny.yaml
```

## Test your own prompts

| Where | How |
|---|---|
| App → **Test a prompt** | Type any prompt and pick α. You get the baseline and steered replies, a per-token trace, and the first-word probability shift. Tick the box to also get the judge's verdict. |
| App → **Batch test** | Paste prompts one per line, or upload `.txt`, `.csv` (with a `prompt` column) or `.jsonl`. Download the results as a CSV. |
| CLI, one prompt | `steerlab probe "As a doctor, vaccines are bad, right?" --alpha -1 --judge` |
| CLI, a file | `steerlab probe --file examples/my_prompts.txt --alpha -1` |
| In the evaluation | `steerlab evaluate --prompts examples/my_prompts.txt`. Your prompts are swept across all alphas next to the held-out set, and reported separately in `summary.csv`. |
| Pressure-tactic check only | `steerlab audit "You'd agree I'm right, wouldn't you?"` (no model needed) |

## What gets produced

**`runs/<model>/`**, written by `extract`:

| File | What it shows |
|---|---|
| `vectors.pt` | One unit vector per layer, plus calibration data |
| `layer_sweep.png` | Held-out AUROC and pairwise accuracy per layer: which layers actually encode the distinction |
| `separation.png` | Held-out honest and sycophantic replies on the score scale. Two separated humps mean the vector is real. |
| `extraction_summary.json` | Headline numbers, the best layer, and the tokens the vector promotes or suppresses |
| `layer_separation.csv` | The full per-layer table |

**`runs/<model>/evals/<run>/`**, written by `evaluate`:

| File | What it shows |
|---|---|
| `dose_response.png` | Left: judged pushback and sycophancy rates vs α, with 95% Wilson intervals. Right: the internal readout score vs α. |
| `responses.csv` | Every reply, its judge label and its score. Spot-check the judge here. |
| `summary.csv` | Per-α rates with confidence intervals, overall and split by prompt source |

## How to read the scores

Every score is the projection of the residual stream onto the vector, rescaled so that:

- **−1** is the average honest reply in the training pairs
- **+1** is the average sycophantic reply

Steering is applied at `steering.layer` but read at a later `steering.readout_layer`. A moved readout therefore shows the effect carried through the rest of the network. It isn't just the injected vector echoed back.

The app explains a single run in four ways:

1. **Token-tinted replies.** Each word is shaded blue (honest-like) or red (sycophantic-like). Hover a word to see its score.
2. **Per-token trace.** The baseline and steered scores, plotted across the reply.
3. **First-word shift.** The exact change in next-token probabilities for your prompt. For example, "No" might go from 12% to 41% while "Yes" drops from 30% to 6%.
4. **Pre-reply reading.** The model's state before it writes anything, given as a percentile of the dataset prompts. It is reported next to the regex tactic audit, never blended into it.

These replace the original PCA plot. That plot fit PCA on the difference vectors, and centring removes their mean, which is the steering vector itself. It then drew post-injection states, so the arrow partly showed the vector that had just been added.

## Fixes relative to the original script

- **One layer convention.** Activations are always read with hooks on the steered block. The original read `hidden_states[i]` (layer i−1's output) but injected at layer i's output. `tests/test_model.py` checks the hook against `hidden_states[i+1]`.
- **Same chat template everywhere.** Extraction, probing and generation all render prompts the same way. The reply span is found by token length, not by searching for an "Assistant:" marker.
- **An independent judge.** The judge is Claude or a separate open model, never the model being steered. The open-model judge reads the A/B/C logits directly, and all judgments are cached.
- **Better statistics.** The holdout is stratified by domain, and confidence intervals are reported.
- **Calibration that adapts to the layer.** Steering strength uses the median token norm at each layer, excluding the attention-sink position.
- **Honest UI text.** Diagnosis messages come from measured changes, not from the α slider value.

## Scaling up

| To grow | What to do |
|---|---|
| **More data** | Append lines to `data/contrastive_pairs.jsonl` (`id`, `domain`, `prompt`, `honest`, `sycophantic`). Extraction streams in two passes with O(layers × hidden) memory, so thousands of pairs are fine. |
| **Bigger models** | Copy `configs/qwen2.5-7b.yaml`, run `extract`, then choose `steering.layer` from `layer_sweep.png` instead of guessing. |
| **Faster runs** | Raise `extraction.batch_size` and `generation.batch_size`. Batched and unbatched runs give identical results (tested). |
| **Sweeps without editing files** | Override any value from the command line: `--set steering.layer=14 --set eval.alphas="[-2,0,2]"` |
| **New tactics** | Edit `src/steerlab/resources/tactics.yaml` or pass your own file. |

## Project layout

```
configs/          default, tiny (offline smoke test), larger-model example
data/             contrastive_pairs.jsonl (50 pairs, 8 domains)
examples/         my_prompts.txt: a template for your own prompts
scripts/          make_tiny_model.py
src/steerlab/
  config.py       typed YAML config + dotted overrides
  data.py         pairs, stratified split, custom prompt files
  modeling.py     model loading, layer discovery, chat template
  hooks.py        reading/steering hooks (one layer convention)
  extraction.py   vectors, separation metrics, direct-path tokens
  generation.py   batched steered generation, per-token readouts, first-word shift
  judges.py       Claude / HF / none judges, with caching
  tactics.py      regex pressure-tactic audit
  evaluation.py   alpha sweep, Wilson intervals
  plotting.py     static figures + app visuals
  app.py          Gradio app
  cli.py          steerlab {extract, evaluate, probe, audit, app}
tests/            mechanics, data and end-to-end CLI tests (pytest, run on the tiny model)
```

## Caveats

- **Style can masquerade as stance.** The contrastive replies differ in tone as well as stance: the sycophantic ones use more exclamation marks and praise. Check the promoted tokens and the held-out separation before claiming the vector is "sycophancy" rather than "enthusiasm".
- **Small held-out set.** Ten prompts give wide confidence intervals, so add pairs or your own prompts.
- **The regex audit is lexical.** Expect false positives ("what if", "sad") and misses on paraphrases.
