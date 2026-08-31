"""
The single-process, zero-API-key demo entry point. Runs the same
verification each matcher's own main() does (Tier 1 exact-match precision,
Tier 2 precision/recall/leak-checks, the three-bucket outcome), but in one
Python process instead of three separate script invocations - this is
purely a startup-time optimization (avoids importing pandas three times)
so `make demo` / `run_demo.sh` finish in single-digit seconds; it computes
nothing that src/matchers/exact_matcher.py, fuzzy_matcher.py, and
src/pipeline.py don't already compute and print individually.

Reuses reports/tier3_adjudication_results.json (committed to this repo) for
Tier 3 - no live LLM call, no API key required. See README.md's "Reproduce
in 30 seconds" section.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "src" / "matchers"))

from exact_matcher import load_sources, run_exact_match, verify_against_ground_truth as verify_t1
from fuzzy_matcher import run_tier2, verify_against_ground_truth as verify_t2
from pipeline import run_full_pipeline


def main():
    ledger, gateway, bank = load_sources()

    print("=" * 70)
    print("TIER 1 - exact match")
    print("=" * 70)
    tier1 = run_exact_match(ledger, gateway, bank)
    t1_report = verify_t1(tier1)
    print(f"Matched: {t1_report['total_tier1_matches']}/140 events "
          f"({t1_report['pct_of_all_events_matched']:.1%})")
    print(f"Precision (verified against ground truth): "
          f"{t1_report['correct_matches']}/{t1_report['total_tier1_matches']} "
          f"= {t1_report['precision']:.4%}")
    print(f"Matched scenario breakdown: {t1_report['matched_scenario_breakdown']}")

    print()
    print("=" * 70)
    print("TIER 2 - fuzzy/algorithmic")
    print("=" * 70)
    combined, split_out, fuzzy_out = run_tier2(
        tier1.unmatched_ledger, tier1.unmatched_gateway, tier1.unmatched_bank
    )
    t2_report = verify_t2(combined)
    print(f"Matched: {t2_report['total_tier2_matches']} of {t2_report['n_in_scope_events']} in-scope events")
    print(f"Precision (verified against ground truth): "
          f"{t2_report['correct_matches']}/{t2_report['total_tier2_matches']} = {t2_report['precision']:.4%}")
    print(f"Recall: {t2_report['recall']:.4%}")
    ss = t2_report["split_settlement"]
    print(f"split_settlement (separate): {ss['correct']}/{ss['correct']+ss['incorrect']} "
          f"precision={ss['precision']:.4%} recall={ss['recall']:.4%}")
    print(f"Leak checks - duplicate events matched: {len(t2_report['duplicate_leaks'])} (must be 0), "
          f"orphan events matched: {len(t2_report['orphan_leaks'])} (must be 0)")

    print()
    print("=" * 70)
    print("FULL PIPELINE (Tier 1 -> 2 -> 3, Tier 3 from committed cache, no API key)")
    print("=" * 70)
    outcome = run_full_pipeline(rerun_tier3=False)
    s = outcome["audit_log"].summary()
    print(f"{'Bucket':<30}{'Count':>8}{'% of 140':>12}")
    print(f"{'Full match (3-way)':<30}{s['full_match']:>8}{s['full_match_pct']:>12.1%}")
    print(f"{'Partial match (2-of-3)':<30}{s['partial_match']:>8}{s['partial_match_pct']:>12.1%}")
    print(f"{'Flagged exception':<30}{s['flagged_exception']:>8}{s['flagged_exception_pct']:>12.1%}")
    print(f"{'TOTAL':<30}{s['total_events']:>8}{'100.0%':>12}")

    print()
    print("All numbers above are computed live against data/ground_truth.csv, not hardcoded.")
    print("Full report: reports/metrics_report.md | Live dashboard: see README.md")


if __name__ == "__main__":
    main()
