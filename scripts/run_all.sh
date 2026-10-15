#!/usr/bin/env bash
# Full pipeline: extract -> evaluate (with controls) -> report + poster.
#
#   bash scripts/run_all.sh                       # judge picked by config (Claude if ANTHROPIC_API_KEY is set)
#   JUDGE=none bash scripts/run_all.sh            # no behaviour labels, fastest
#   JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/run_all.sh   # smaller local judge
#   AUTHOR="Your Name" URL="github.com/you/sycophancy-steering" bash scripts/run_all.sh
#
# Then validate the judge with:  steerlab label   (label ~40 replies, ~1 hour)
# and rebuild the write-up with: steerlab report --author "..." --url "..."
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/default.yaml}"
SET=()
[[ -n "${JUDGE:-}" ]] && SET+=(--set "judge.backend=${JUDGE}")
[[ -n "${JUDGE_MODEL:-}" ]] && SET+=(--set "judge.backend=hf" --set "judge.hf_model=${JUDGE_MODEL}")

echo "== 1/3 extract =="
steerlab extract -c "$CONFIG"

echo "== 2/3 evaluate (this is the long step) =="
steerlab evaluate -c "$CONFIG" "${SET[@]+"${SET[@]}"}"

echo "== 3/3 report + poster =="
steerlab report -c "$CONFIG" --author "${AUTHOR:-}" --affiliation "${AFFILIATION:-}" --url "${URL:-}"

echo
echo "Done. Open results/RESULTS.md and results/poster.pdf."
echo "Next: steerlab label   (validate the judge), then re-run: steerlab report"
