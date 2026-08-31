"""
Shared pytest fixtures. Every test here runs against either the
deterministic Tier 1/2 matchers (fast, no external calls) or the already-
generated reports/audit_log.csv (Tier 3's cached results) - never a live
LLM call. This keeps the suite free and fast to run in CI.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "matchers"))

from exact_matcher import load_sources, run_exact_match  # noqa: E402
from fuzzy_matcher import run_tier2  # noqa: E402

DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"


@pytest.fixture(scope="session")
def ground_truth():
    path = DATA_DIR / "ground_truth.csv"
    if not path.exists():
        pytest.skip(f"{path} not found - run `python data/generate_synthetic_data.py` first")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def tier1_outcome():
    ledger, gateway, bank = load_sources()
    return run_exact_match(ledger, gateway, bank)


@pytest.fixture(scope="session")
def tier2_outcome(tier1_outcome):
    combined, split_out, fuzzy_out = run_tier2(
        tier1_outcome.unmatched_ledger, tier1_outcome.unmatched_gateway, tier1_outcome.unmatched_bank
    )
    return combined


@pytest.fixture(scope="session")
def audit_log_df():
    path = REPORTS_DIR / "audit_log.csv"
    if not path.exists():
        pytest.skip(f"{path} not found - run `python src/pipeline.py` first")
    return pd.read_csv(path)
