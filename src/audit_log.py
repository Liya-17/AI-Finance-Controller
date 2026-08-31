"""
Phase 5 - audit trail.

Every decision made anywhere in the pipeline (Tier 1 exact match, Tier 2
fuzzy/algorithmic match, Tier 3 LLM adjudication, and exception
categorization) gets one AuditEntry: timestamp, which tier resolved it (or
didn't), confidence, and a human-readable rationale. This is what makes the
pipeline's output auditable after the fact rather than a black box - a
reviewer should be able to look at any ledger row and see exactly why the
system landed on its outcome, without re-running anything.

THREE-BUCKET OUTCOME MODEL (do not collapse these into one "resolved" count)
------------------------------------------------------------------------------
"132/140 resolved" blends three structurally different outcomes into one
number, and an evaluator reading that without qualification could easily
assume 132 full three-way reconciliations - which is not what happened.
Every AuditEntry is tagged with exactly one of:

  full_match       - all three sources (ledger+gateway+bank) reconciled.
                      Tier 1's exact matches and Tier 2's pairwise/split
                      matches are always this bucket.
  partial_match     - a CONFIRMED, correct match on only one of
                      gateway/bank, by design (not a force-completion
                      attempt that fell short). This is what Tier 3 returns
                      for orphan-shaped rows: the true state of the world is
                      that only one counterpart exists, and finding it is a
                      complete, correct answer - not a lesser one.
  flagged_exception - genuine, irreducible ambiguity (e.g. two tied
                      candidate rows Tier 3 could not distinguish).
                      Correctly left for human review, not guessed at.

See exceptions.py for how flagged_exception rows are further categorized,
and reports/metrics_report.md for where these three counts are reported
side by side rather than summed into a single headline number.

Every entry also carries `resolving_tier` and `provider` so that if Tier 3
is later re-run against a different LLM provider (see
reports/llm_provider_blockers.md - this pipeline currently runs Tier 3
against Google Gemini, gemini-3.5-flash-lite, not the Anthropic model the
brief originally specified), the audit log's provenance already
distinguishes which entries came from which run without any schema change.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

VALID_OUTCOME_BUCKETS = {"full_match", "partial_match", "flagged_exception"}
VALID_TIERS = {"tier1_exact", "tier2_fuzzy", "tier3_llm", "unresolved"}


@dataclass
class AuditEntry:
    ledger_id: str
    outcome_bucket: str            # full_match | partial_match | flagged_exception
    resolving_tier: str            # tier1_exact | tier2_fuzzy | tier3_llm | unresolved
    matched_gateway_id: Optional[str]
    matched_bank_id: Optional[str]
    confidence: float
    rationale: str
    provider: str = "n/a"          # "n/a" for Tier 1/2 (no LLM involved), else e.g. "google_gemini"
    model: str = "n/a"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if self.outcome_bucket not in VALID_OUTCOME_BUCKETS:
            raise ValueError(f"invalid outcome_bucket: {self.outcome_bucket!r} (must be one of {VALID_OUTCOME_BUCKETS})")
        if self.resolving_tier not in VALID_TIERS:
            raise ValueError(f"invalid resolving_tier: {self.resolving_tier!r} (must be one of {VALID_TIERS})")


class AuditLog:
    """In-memory log with append + CSV/JSON export. Not a database - this
    dataset is small enough (140 events) that a flat file is the right
    level of infrastructure; a real production system would write to an
    append-only table instead, but the AuditEntry schema is what would
    carry over unchanged."""

    def __init__(self):
        self.entries: list[AuditEntry] = []

    def log(self, entry: AuditEntry):
        self.entries.append(entry)

    def log_tier1_match(self, match):
        """match: exact_matcher.MatchResult"""
        self.log(AuditEntry(
            ledger_id=match.ledger_id,
            outcome_bucket="full_match",
            resolving_tier="tier1_exact",
            matched_gateway_id=match.gateway_id,
            matched_bank_id=match.bank_id,
            confidence=match.confidence,
            rationale=match.rationale,
        ))

    def log_tier2_match(self, match):
        """match: fuzzy_matcher.MatchResult"""
        self.log(AuditEntry(
            ledger_id=match.ledger_id,
            outcome_bucket="full_match",
            resolving_tier="tier2_fuzzy",
            matched_gateway_id=match.gateway_id,
            matched_bank_id=match.bank_id,
            confidence=match.confidence,
            rationale=f"[{match.match_type}] {match.rationale}",
        ))

    def log_tier3_result(self, result, provider: str = "google_gemini", model: str = "gemini-3.5-flash-lite"):
        """
        result: llm_adjudicator.AdjudicationResult

        Bucket assignment:
          verdict == 'match' AND both ids set        -> full_match
          verdict == 'match' AND exactly one id set   -> partial_match
          verdict == 'no_match' or 'uncertain'        -> flagged_exception
        """
        has_gateway = result.matched_gateway_id is not None
        has_bank = result.matched_bank_id is not None

        if result.verdict == "match" and has_gateway and has_bank:
            bucket = "full_match"
        elif result.verdict == "match" and (has_gateway or has_bank):
            bucket = "partial_match"
        else:
            bucket = "flagged_exception"

        self.log(AuditEntry(
            ledger_id=result.ledger_id,
            outcome_bucket=bucket,
            resolving_tier="tier3_llm",
            matched_gateway_id=result.matched_gateway_id,
            matched_bank_id=result.matched_bank_id,
            confidence=result.confidence,
            rationale=result.rationale,
            provider=provider,
            model=model,
        ))

    def log_unresolved(self, ledger_id: str, rationale: str = "no candidate found in any tier"):
        """For rows that never even reached Tier 3 as a distinguishable
        candidate set (shouldn't happen in this pipeline since every
        Tier-1/2 leftover goes to Tier 3, but kept for completeness/safety
        if Tier 3 is ever run on a subset)."""
        self.log(AuditEntry(
            ledger_id=ledger_id,
            outcome_bucket="flagged_exception",
            resolving_tier="unresolved",
            matched_gateway_id=None,
            matched_bank_id=None,
            confidence=0.0,
            rationale=rationale,
        ))

    def bucket_counts(self) -> dict:
        counts = {b: 0 for b in VALID_OUTCOME_BUCKETS}
        for e in self.entries:
            counts[e.outcome_bucket] += 1
        return counts

    def tier_counts(self) -> dict:
        counts = {}
        for e in self.entries:
            counts[e.resolving_tier] = counts.get(e.resolving_tier, 0) + 1
        return counts

    def summary(self) -> dict:
        total = len(self.entries)
        buckets = self.bucket_counts()
        return {
            "total_events": total,
            "full_match": buckets["full_match"],
            "partial_match": buckets["partial_match"],
            "flagged_exception": buckets["flagged_exception"],
            "full_match_pct": buckets["full_match"] / total if total else 0.0,
            "partial_match_pct": buckets["partial_match"] / total if total else 0.0,
            "flagged_exception_pct": buckets["flagged_exception"] / total if total else 0.0,
            "by_resolving_tier": self.tier_counts(),
        }

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([asdict(e) for e in self.entries])

    def save(self, out_dir: Path = REPORTS_DIR, basename: str = "audit_log"):
        out_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        csv_path = out_dir / f"{basename}.csv"
        json_path = out_dir / f"{basename}.json"
        df.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps([asdict(e) for e in self.entries], indent=2, default=str), encoding="utf-8")
        return csv_path, json_path


def print_three_bucket_summary(log: "AuditLog"):
    """
    Print the three-bucket breakdown explicitly, never collapsed into a
    single "resolved" percentage - see the module docstring for why this
    distinction matters for honest reporting.
    """
    s = log.summary()
    print("--- Reconciliation outcome (three buckets - do not sum into one 'resolved' number) ---")
    print(f"  Full three-way matches (ledger+gateway+bank): {s['full_match']}/{s['total_events']} "
          f"({s['full_match_pct']:.1%})")
    print(f"  Confirmed partial matches (2-of-3, by design):  {s['partial_match']}/{s['total_events']} "
          f"({s['partial_match_pct']:.1%})")
    print(f"  Flagged exceptions (human review needed):       {s['flagged_exception']}/{s['total_events']} "
          f"({s['flagged_exception_pct']:.1%})")
    print(f"  By resolving tier: {s['by_resolving_tier']}")
