#!/usr/bin/env bash
# Bundle the app + your extracted vectors into runs/space_build/ for a Hugging Face Space.
#
#   steerlab extract -c configs/default.yaml       # once, so vectors exist
#   bash scripts/prepare_space.sh
#   cd runs/space_build && git init && git remote add origin https://huggingface.co/spaces/USERNAME/SPACE-NAME
#   git add -A && git commit -m "Deploy" && git push -u origin main
set -euo pipefail
cd "$(dirname "$0")/.."
SLUG="${SLUG:-HuggingFaceTB__SmolLM2-1.7B-Instruct}"
SRC="runs/$SLUG"
[[ -f "$SRC/vectors.pt" ]] || { echo "No $SRC/vectors.pt: run 'steerlab extract -c configs/default.yaml' first"; exit 1; }
rm -rf runs/space_build && mkdir -p runs/space_build/runs/"$SLUG"
cp space/README.md space/app.py space/requirements.txt runs/space_build/
cp -r src/steerlab runs/space_build/steerlab
cp -r configs data runs/space_build/
cp "$SRC"/vectors.pt "$SRC"/*.png "$SRC"/*.json "$SRC"/*.csv runs/space_build/runs/"$SLUG"/ 2>/dev/null || true
find runs/space_build -name __pycache__ -prune -exec rm -rf {} +
echo "[+] runs/space_build/ is ready ($(du -sh runs/space_build | cut -f1)). Push it to a new Gradio Space (see comments in this script)."
