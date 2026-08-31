"""
Phase 5 - exception handling.

Consumes the pipeline's three-way outcome (Tier 1 + Tier 2 matches, Tier 3
adjudication results) and categorizes everything that landed in the
`flagged_exception` bucket (see audit_log.py) into a human-readable reason,
so a reviewer opening the exception queue sees WHY each row is unresolved,
not just THAT it is.

This module does NOT try to resolve anything - by the time a row reaches
here, Tier 1, Tier 2, and Tier 3 have all already had a shot at it. Forcing
a categorization heuristic to also attempt a match would reintroduce the
exact risk the tiered design exists to avoid (see llm_naive_experiment.py's
measured 82.7% precision when a single pass tries to do too much at once).

CATEGORIES
----------
  duplicate_ambiguity   - Tier 3 returned 'uncertain' because two or more
                           candidate rows on one side are indistinguishable
                           (the `duplicate` scenario's shape). Needs a human
                           to decide: true double-write, or two legitimate
                           separate payments.
  timing_mismatch        - a candidate exists but was rejected/left
                           unresolved specifically over a date/time
                           disagreement larger than Tier 2's window and not
                           resolved with confidence by Tier 3.
  data_quality_issue      - malformed/inconsistent underlying data (e.g. an
                           amount or id that doesn't parse, a row with no
                           amount overlap in a wide window at all) rather
                           than a matching-algorithm limitation.
  no_candidate_found      - the row genuinely appears to have no plausible
                           counterpart in the remaining pool at all (Tier 3's
                           'no_match' verdict, or a Tier 1/2 leftover that
                           never even reached Tier 3 - defensive case).
  unclassified            - fallback for anything not captured by the above
                           (kept small deliberately; growth here signals the
                           categorization rules need another category, not
                           that this bucket is fine to ignore).

Reminder: `partial_match` rows (confirmed one-sided matches - see
audit_log.py) are NOT exceptions. They are correct, resolved outcomes and
must never be run through this categorizer or counted in the exception
queue - see the three-bucket model in audit_log.py's docstring.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

VALID_CATEGORIES = {
    "duplicate_ambiguity",
    "timing_mismatch",
    "data_quality_issue",
    "no_candidate_found",
    "unclassified",
}


@dataclass
class ExceptionRecord:
    ledger_id: str
    category: str
    verdict: str            # the Tier 3 verdict that produced this exception (uncertain | no_match)
    confidence: float
    rationale: str
    candidates_considered: dict

    def __post_init__(self):
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"invalid category: {self.category!r}")


def categorize_tier3_uncertain(result) -> str:
    """
    Categorize a single Tier 3 AdjudicationResult that did NOT resolve to a
    match, using the rationale text and candidate shape - not a second
    attempt at matching. This is intentionally simple pattern-matching over
    the model's own stated reasoning (which already explains WHY it
    couldn't decide), not independent re-analysis of the raw records.
    """
    rationale_lower = result.rationale.lower()
    gw_candidates = result.candidates_considered.get("gateway_ids", [])
    bk_candidates = result.candidates_considered.get("bank_ids", [])

    if result.verdict == "no_match":
        if not gw_candidates and not bk_candidates:
            return "no_candidate_found"
        return "data_quality_issue"

    # verdict == "uncertain": tied/indistinguishable candidates is the
    # dominant real pattern observed in this dataset (see
    # reports/tier3_adjudication_results.json) - two rows sharing an
    # identical id+amount on one side, which is exactly the `duplicate`
    # scenario's shape.
    tie_keywords = ["identical", "duplicate", "two candidate", "indistinguishable", "competing"]
    if any(kw in rationale_lower for kw in tie_keywords):
        return "duplicate_ambiguity"

    time_keywords = ["date", "timestamp", "chronology", "time of day", "value_datetime", "txn_datetime"]
    if any(kw in rationale_lower for kw in time_keywords):
        return "timing_mismatch"

    return "unclassified"


def build_exception_queue(tier3_results: list) -> list:
    """
    tier3_results: list of llm_adjudicator.AdjudicationResult

    Only rows with verdict in {'no_match', 'uncertain'} become exceptions -
    'match' verdicts (full or partial) are resolved outcomes, not
    exceptions, and must not appear here (see module docstring).
    """
    exceptions = []
    for r in tier3_results:
        if r.verdict not in ("no_match", "uncertain"):
            continue
        category = categorize_tier3_uncertain(r)
        exceptions.append(
            ExceptionRecord(
                ledger_id=r.ledger_id,
                category=category,
                verdict=r.verdict,
                confidence=r.confidence,
                rationale=r.rationale,
                candidates_considered=r.candidates_considered,
            )
        )
    return exceptions


def category_breakdown(exceptions: list) -> dict:
    counts = {c: 0 for c in VALID_CATEGORIES}
    for e in exceptions:
        counts[e.category] += 1
    return counts


def save_exception_queue(exceptions: list, out_dir: Path = REPORTS_DIR, basename: str = "exception_queue"):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(e) for e in exceptions])
    csv_path = out_dir / f"{basename}.csv"
    json_path = out_dir / f"{basename}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps([asdict(e) for e in exceptions], indent=2, default=str), encoding="utf-8")
    return csv_path, json_path


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "matchers"))
    from exact_matcher import load_sources, run_exact_match
    from fuzzy_matcher import run_tier2
    from llm_adjudicator import AdjudicationResult

    results_path = REPORTS_DIR / "tier3_adjudication_results.json"
    if not results_path.exists():
        print(f"FATAL: {results_path} not found. Run src/matchers/llm_adjudicator.py first "
              f"(Phase 4, Step B) to produce Tier 3 results before building the exception queue.")
        sys.exit(1)

    tier3_raw = json.loads(results_path.read_text(encoding="utf-8"))
    tier3_results = [AdjudicationResult(**r) for r in tier3_raw]

    exceptions = build_exception_queue(tier3_results)
    breakdown = category_breakdown(exceptions)

    print(f"Exception queue: {len(exceptions)} flagged exceptions out of {len(tier3_results)} "
          f"Tier 3-adjudicated rows.")
    print(f"Category breakdown: {breakdown}")

    for e in exceptions:
        print(f"\n  {e.ledger_id} [{e.category}] (verdict={e.verdict}, confidence={e.confidence:.2f})")
        print(f"    {e.rationale[:200]}")

    csv_path, json_path = save_exception_queue(exceptions)
    print(f"\nSaved to: {csv_path}, {json_path}")

    return exceptions


if __name__ == "__main__":
    main()
