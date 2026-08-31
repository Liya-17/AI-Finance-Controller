"""
Tier 2 (fuzzy_matcher.py) must never match a `duplicate`-scenario event
(two raw-identical candidate rows - genuinely ambiguous, reserved for Tier
3's judgment) or an `orphan`-scenario event (no true counterpart on one
side - force-matching one would fabricate a reconciliation that never
happened). This is exactly the leak that made it into an early build of
this pipeline, where the tool's own printed precision looked like 100%
right up until it was checked against ground truth - see
reports/metrics_report.md, "What broke" item 2. This test exists so that
specific regression can never happen silently again.
"""

import pandas as pd


def test_no_scenario_leakage(tier2_outcome, ground_truth):
    ledger_to_scenario = {}
    for _, row in ground_truth.iterrows():
        if pd.notna(row["ledger_ids"]):
            ledger_to_scenario[row["ledger_ids"]] = row["scenario"]

    matched_scenarios = [ledger_to_scenario.get(m.ledger_id) for m in tier2_outcome.matches]

    duplicate_leaks = [s for s in matched_scenarios if s == "duplicate"]
    orphan_leaks = [s for s in matched_scenarios if s == "orphan"]

    assert duplicate_leaks == [], f"Tier 2 matched {len(duplicate_leaks)} duplicate-scenario event(s) - must be 0"
    assert orphan_leaks == [], f"Tier 2 matched {len(orphan_leaks)} orphan-scenario event(s) - must be 0"


def test_tier2_zero_false_positives(tier2_outcome, ground_truth):
    ledger_to_event, gateway_to_event, bank_to_event = {}, {}, {}
    for _, row in ground_truth.iterrows():
        if pd.notna(row["ledger_ids"]):
            for lid in str(row["ledger_ids"]).split(";"):
                ledger_to_event[lid] = row["event_id"]
        if pd.notna(row["gateway_ids"]):
            for gid in str(row["gateway_ids"]).split(";"):
                gateway_to_event[gid] = row["event_id"]
        if pd.notna(row["bank_ids"]):
            for bid in str(row["bank_ids"]).split(";"):
                bank_to_event[bid] = row["event_id"]

    assert len(tier2_outcome.matches) > 0, "Tier 2 produced no matches at all - something upstream is broken"

    incorrect = []
    for m in tier2_outcome.matches:
        l_evt = ledger_to_event.get(m.ledger_id)
        g_evts = {gateway_to_event.get(g) for g in m.gateway_id.split(";")}
        b_evt = bank_to_event.get(m.bank_id)
        if l_evt is None or g_evts != {l_evt} or b_evt != l_evt:
            incorrect.append((m.ledger_id, m.gateway_id, m.bank_id, l_evt))

    assert incorrect == [], f"Tier 2 produced {len(incorrect)} false positive(s): {incorrect}"
