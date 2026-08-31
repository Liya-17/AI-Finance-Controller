#!/usr/bin/env bash
# Zero-friction reproduction: no API key, prints verified metrics to stdout
# in a few seconds. Uses the committed data/ground_truth.csv and
# reports/tier3_adjudication_results.json - Tier 1 and Tier 2 are recomputed
# live and re-verified against ground truth on every run; Tier 3 reuses its
# cached, already-verified decisions rather than re-calling an LLM.
#
# Fallback for Windows/no-make users - see `make demo` for the same thing
# via a Makefile, if you have make available.
set -euo pipefail

echo "Installing dependencies (matchers need pandas, rapidfuzz, python-dotenv - the"
echo "full requirements.txt, not requirements-dashboard.txt, which is deploy-only)..."
pip install -q -r requirements.txt

echo
python scripts/print_verified_metrics.py
