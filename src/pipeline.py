"""
Full pipeline runner: Tier 1 -> Tier 2 -> Tier 3 -> audit log -> exception
queue, in one script. This is what dashboard/app.py (Phase 6) and
reports/metrics_report.md (Phase 7) both build on - a single source of
truth for the pipeline's outcome, tagged with the three-bucket model (see
audit_log.py) so nothing downstream can accidentally collapse full/partial/
exception into one "resolved" number.

Tier 3 (LLM adjudication) makes real API calls and costs real time (free-
tier rate limiting - see llm_adjudicator.py). Set --skip-tier3 to reuse the
already-saved reports/tier3_adjudication_results.json instead of
re-running it, which is the default for repeated local runs (dashboard
refreshes, report regeneration) - only pass --rerun-tier3 to actually hit
the API again.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "matchers"))

from exact_matcher import load_sources, run_exact_match
from fuzzy_matcher import run_tier2
from llm_adjudicator import AdjudicationResult, ADJUDICATOR_MODEL, require_api_key, run_tier3

from audit_log import AuditLog, print_three_bucket_summary
from exceptions import build_exception_queue, category_breakdown, save_exception_queue

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TIER3_CACHE_PATH = REPORTS_DIR / "tier3_adjudication_results.json"


def run_full_pipeline(rerun_tier3: bool = False):
    ledger, gateway, bank = load_sources()

    tier1 = run_exact_match(ledger, gateway, bank)
    combined, split_out, fuzzy_out = run_tier2(
        tier1.unmatched_ledger, tier1.unmatched_gateway, tier1.unmatched_bank
    )

    if rerun_tier3 or not TIER3_CACHE_PATH.exists():
        api_key = require_api_key()
        tier3_results = run_tier3(
            combined.unmatched_ledger, combined.unmatched_gateway, combined.unmatched_bank, api_key
        )
        TIER3_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        TIER3_CACHE_PATH.write_text(
            json.dumps([r.__dict__ for r in tier3_results], indent=2, default=str), encoding="utf-8"
        )
    else:
        tier3_raw = json.loads(TIER3_CACHE_PATH.read_text(encoding="utf-8"))
        tier3_results = [AdjudicationResult(**r) for r in tier3_raw]

    log = AuditLog()
    for m in tier1.matches:
        log.log_tier1_match(m)
    for m in combined.matches:
        log.log_tier2_match(m)
    for r in tier3_results:
        log.log_tier3_result(r, provider="google_gemini", model=ADJUDICATOR_MODEL)

    exceptions = build_exception_queue(tier3_results)

    return {
        "tier1": tier1,
        "tier2_combined": combined,
        "tier2_split": split_out,
        "tier2_fuzzy": fuzzy_out,
        "tier3_results": tier3_results,
        "audit_log": log,
        "exceptions": exceptions,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the full 3-tier reconciliation pipeline.")
    parser.add_argument("--rerun-tier3", action="store_true",
                         help="Re-run Tier 3 LLM adjudication against the live API instead of reusing "
                              "the cached reports/tier3_adjudication_results.json")
    args = parser.parse_args()

    outcome = run_full_pipeline(rerun_tier3=args.rerun_tier3)

    print(f"Tier 1: {len(outcome['tier1'].matches)} matches")
    print(f"Tier 2: {len(outcome['tier2_combined'].matches)} matches "
          f"({len(outcome['tier2_split'].matches)} split_settlement + {len(outcome['tier2_fuzzy'].matches)} pairwise fuzzy)")
    print(f"Tier 3: {len(outcome['tier3_results'])} adjudicated")
    print()

    print_three_bucket_summary(outcome["audit_log"])

    print()
    breakdown = category_breakdown(outcome["exceptions"])
    print(f"Exception category breakdown: {breakdown}")

    csv_path, json_path = outcome["audit_log"].save()
    print(f"\nAudit log saved to: {csv_path}, {json_path}")

    exc_csv, exc_json = save_exception_queue(outcome["exceptions"])
    print(f"Exception queue saved to: {exc_csv}, {exc_json}")

    return outcome


if __name__ == "__main__":
    main()
