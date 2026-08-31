#!/usr/bin/env bash
# Reproduces the full pipeline end to end with NO API KEY REQUIRED.
# Uses the ground-truth data and cached reports/tier3_adjudication_results.json
# already committed to this repo, so Tier 3's LLM output is exact and
# reproducible rather than re-called live. See README.md for what each
# step does; this script is just the copy-paste-free version of it.
set -euo pipefail

echo "== Installing dependencies =="
pip install -r requirements.txt

echo "== Regenerating synthetic data (seed 42, deterministic) =="
python data/generate_synthetic_data.py --records 400 --injection-rate 0.55 --seed 42

echo "== Running the full pipeline (Tier 1 -> 2 -> 3, cached, no API key needed) =="
python src/pipeline.py

echo "== Running the test suite (no live LLM calls) =="
pytest tests/ -v

echo
echo "Done. Launch the dashboard with: streamlit run dashboard/app.py"
