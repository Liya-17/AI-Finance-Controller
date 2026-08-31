"""
Tier 1 - deterministic exact matcher.

Reconciles core_ledger, gateway_settlement, and bank_statement on RAW,
byte-for-byte identical transaction ID + amount + calendar date. No
normalization, no case-folding, no prefix stripping, no date *windows*
(same-day only, checked as an equality not a tolerance), no fuzzy anything -
that logic belongs to Tier 2. This tier exists to cheaply and safely clear
the "nothing went wrong" slice of the data (the `clean` scenario in the
synthetic generator) before the more expensive/uncertain tiers run on the
remainder.

Date is included in the exact-match key (not just id+amount) specifically so
that `timezone_shift` events - where id and amount are untouched but the
calendar date differs across sources - correctly fall through to Tier 2's
date-window logic instead of accidentally passing here.

Because Tier 1's output is treated as settled truth by every downstream tier
and by the audit trail, it must have zero false positives - a miss just
falls through to Tier 2, but a wrong match here corrupts everything built on
top of it. See verify_against_ground_truth() below, and
reports/metrics_report.md for the measured precision on the full pipeline.

Match unit: a THREE-WAY match (ledger + gateway + bank), not pairwise. A
ledger row only counts as Tier-1-resolved if the exact same txn_id + amount
is found in *both* the gateway and bank tables. This keeps "events resolved"
reporting consistent with how ground_truth.csv is structured (one row per
event, with ledger_ids/gateway_ids/bank_ids as sibling columns).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class MatchResult:
    """One resolved three-way match."""
    ledger_id: str
    gateway_id: str
    bank_id: str
    txn_id: str
    amount: float
    tier: str = "tier1_exact"
    confidence: float = 1.0
    rationale: str = "raw txn_id + amount + date identical across all three sources"


@dataclass
class TierOutcome:
    matches: list = field(default_factory=list)          # list[MatchResult]
    unmatched_ledger: pd.DataFrame = None
    unmatched_gateway: pd.DataFrame = None
    unmatched_bank: pd.DataFrame = None


def load_sources(data_dir: Path = DATA_DIR):
    ledger = pd.read_csv(data_dir / "core_ledger.csv", dtype={"txn_id": str})
    gateway = pd.read_csv(data_dir / "gateway_settlement.csv", dtype={"gateway_txn_id": str})
    bank = pd.read_csv(data_dir / "bank_statement.csv", dtype={"reference_no": str})
    return ledger, gateway, bank


def run_exact_match(ledger: pd.DataFrame, gateway: pd.DataFrame, bank: pd.DataFrame) -> TierOutcome:
    """
    Three-way exact match on RAW (txn_id, amount, date).

    A ledger row matches iff there exists exactly-one gateway row and
    exactly-one bank row sharing its raw txn_id, raw amount, AND raw
    calendar date. Ambiguous keys (more than one row on any side sharing the
    same raw key - e.g. a duplicate-scenario row) are deliberately NOT
    auto-resolved here: picking one of several equally-exact candidates
    would be a guess, not a deterministic fact, so those rows are left for
    Tier 2/3 to adjudicate.
    """
    # Build (txn_id, amount, date) -> [row indices] lookup for gateway and bank.
    def build_index(df: pd.DataFrame, id_col: str, date_col: str):
        index = {}
        for idx, row in df.iterrows():
            key = (row[id_col], round(float(row["amount"]), 2), row[date_col])
            index.setdefault(key, []).append(idx)
        return index

    gateway_index = build_index(gateway, "gateway_txn_id", "settlement_date")
    bank_index = build_index(bank, "reference_no", "value_date")

    matches = []
    matched_ledger_idx = set()
    matched_gateway_idx = set()
    matched_bank_idx = set()

    for idx, row in ledger.iterrows():
        key = (row["txn_id"], round(float(row["amount"]), 2), row["txn_date"])
        g_candidates = gateway_index.get(key, [])
        b_candidates = bank_index.get(key, [])

        # Require an unambiguous single candidate on each side - this is
        # also what naturally excludes split_settlement (multiple gateway
        # rows carry suffixed ids like "TXN...-S1", so raw key never matches
        # in the first place) and duplicate (multiple gateway/bank rows
        # would tie on the same raw key, so we bail out rather than guess).
        if len(g_candidates) == 1 and len(b_candidates) == 1:
            g_idx, b_idx = g_candidates[0], b_candidates[0]
            # Also guard against a gateway/bank row already claimed by
            # another ledger row (shouldn't happen with unique raw ids, but
            # keeps the invariant airtight if txn_id generation ever repeats).
            if g_idx in matched_gateway_idx or b_idx in matched_bank_idx:
                continue

            matches.append(
                MatchResult(
                    ledger_id=row["ledger_id"],
                    gateway_id=gateway.loc[g_idx, "gateway_id"],
                    bank_id=bank.loc[b_idx, "bank_id"],
                    txn_id=row["txn_id"],
                    amount=round(float(row["amount"]), 2),
                )
            )
            matched_ledger_idx.add(idx)
            matched_gateway_idx.add(g_idx)
            matched_bank_idx.add(b_idx)

    unmatched_ledger = ledger.drop(index=list(matched_ledger_idx)).reset_index(drop=True)
    unmatched_gateway = gateway.drop(index=list(matched_gateway_idx)).reset_index(drop=True)
    unmatched_bank = bank.drop(index=list(matched_bank_idx)).reset_index(drop=True)

    return TierOutcome(
        matches=matches,
        unmatched_ledger=unmatched_ledger,
        unmatched_gateway=unmatched_gateway,
        unmatched_bank=unmatched_bank,
    )


def verify_against_ground_truth(outcome: TierOutcome, ground_truth_path: Path = DATA_DIR / "ground_truth.csv"):
    """
    Score Tier 1's matches against the hidden ground truth. Returns a dict
    with precision (of the matches Tier 1 made, how many are actually
    correct three-way matches for the SAME event) and the scenario
    breakdown of what got matched - this is what confirms:
      (a) Tier 1 is only resolving ~45% (the clean slice), not creeping into
          id_truncated/timezone_shift territory, and
      (b) zero false positives - every match Tier 1 made is a true match.
    """
    gt = pd.read_csv(ground_truth_path)

    # event_id -> scenario, and reverse lookups from each source id -> event_id
    ledger_to_event = {}
    gateway_to_event = {}
    bank_to_event = {}
    event_scenario = {}
    for _, row in gt.iterrows():
        event_scenario[row["event_id"]] = row["scenario"]
        if pd.notna(row["ledger_ids"]) and row["ledger_ids"]:
            for lid in str(row["ledger_ids"]).split(";"):
                ledger_to_event[lid] = row["event_id"]
        if pd.notna(row["gateway_ids"]) and row["gateway_ids"]:
            for gid in str(row["gateway_ids"]).split(";"):
                gateway_to_event[gid] = row["event_id"]
        if pd.notna(row["bank_ids"]) and row["bank_ids"]:
            for bid in str(row["bank_ids"]).split(";"):
                bank_to_event[bid] = row["event_id"]

    correct = 0
    incorrect = 0
    incorrect_examples = []
    matched_scenarios = []

    for m in outcome.matches:
        l_evt = ledger_to_event.get(m.ledger_id)
        g_evt = gateway_to_event.get(m.gateway_id)
        b_evt = bank_to_event.get(m.bank_id)
        is_correct = (l_evt is not None) and (l_evt == g_evt == b_evt)
        if is_correct:
            correct += 1
            matched_scenarios.append(event_scenario[l_evt])
        else:
            incorrect += 1
            incorrect_examples.append(
                {
                    "ledger_id": m.ledger_id,
                    "gateway_id": m.gateway_id,
                    "bank_id": m.bank_id,
                    "ledger_event": l_evt,
                    "gateway_event": g_evt,
                    "bank_event": b_evt,
                }
            )

    total = len(outcome.matches)
    precision = correct / total if total else float("nan")

    scenario_counts = pd.Series(matched_scenarios).value_counts().to_dict() if matched_scenarios else {}
    n_clean_events = (gt["scenario"] == "clean").sum()
    n_total_events = len(gt)

    return {
        "total_tier1_matches": total,
        "correct_matches": correct,
        "incorrect_matches": incorrect,
        "precision": precision,
        "incorrect_examples": incorrect_examples,
        "matched_scenario_breakdown": scenario_counts,
        "pct_of_all_events_matched": total / n_total_events,
        "n_clean_events": int(n_clean_events),
        "pct_clean_events": n_clean_events / n_total_events,
    }


def main():
    ledger, gateway, bank = load_sources()
    outcome = run_exact_match(ledger, gateway, bank)

    n_events_total = len(pd.read_csv(DATA_DIR / "ground_truth.csv"))
    print(f"Tier 1 (exact match): {len(outcome.matches)} three-way matches "
          f"out of {n_events_total} total events ({len(outcome.matches)/n_events_total:.1%})")
    print(f"Unmatched: ledger={len(outcome.unmatched_ledger)}, "
          f"gateway={len(outcome.unmatched_gateway)}, bank={len(outcome.unmatched_bank)}")

    report = verify_against_ground_truth(outcome)
    print()
    print("--- Ground truth verification ---")
    print(f"Precision: {report['correct_matches']}/{report['total_tier1_matches']} "
          f"= {report['precision']:.4%}")
    print(f"Matched scenario breakdown: {report['matched_scenario_breakdown']}")
    print(f"Clean events in dataset: {report['n_clean_events']} ({report['pct_clean_events']:.1%})")
    print(f"Tier 1 matched {report['pct_of_all_events_matched']:.1%} of all events")

    if report["incorrect_matches"] > 0:
        print()
        print(f"!!! {report['incorrect_matches']} FALSE POSITIVE(S) - see incorrect_examples:")
        for ex in report["incorrect_examples"]:
            print(f"  {ex}")

    return outcome, report


if __name__ == "__main__":
    main()
