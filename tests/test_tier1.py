"""
Tier 1 (exact_matcher.py) is settled truth for everything downstream - a
false positive here is worse than a miss, since Tier 2/3 never re-check a
Tier 1 match. This test locks in the zero-false-positive guarantee against
ground truth, so a future change to exact_matcher.py that quietly loosens
the matching key gets caught immediately, not discovered after the fact
the way the original duplicate-leak bug was (see reports/metrics_report.md).
"""

import pandas as pd


def test_tier1_zero_false_positives(tier1_outcome, ground_truth):
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

    assert len(tier1_outcome.matches) > 0, "Tier 1 produced no matches at all - something upstream is broken"

    incorrect = []
    for m in tier1_outcome.matches:
        l_evt = ledger_to_event.get(m.ledger_id)
        g_evt = gateway_to_event.get(m.gateway_id)
        b_evt = bank_to_event.get(m.bank_id)
        if l_evt is None or not (l_evt == g_evt == b_evt):
            incorrect.append((m.ledger_id, m.gateway_id, m.bank_id, l_evt, g_evt, b_evt))

    assert incorrect == [], f"Tier 1 produced {len(incorrect)} false positive(s): {incorrect}"


def test_tier1_matches_clean_scenario_only(tier1_outcome, ground_truth):
    """Tier 1 should resolve ~exactly the clean-scenario proportion, not
    creep into scenarios that need normalization/tolerance (Tier 2's job)."""
    ledger_to_scenario = {}
    for _, row in ground_truth.iterrows():
        if pd.notna(row["ledger_ids"]):
            ledger_to_scenario[row["ledger_ids"]] = row["scenario"]

    matched_scenarios = {ledger_to_scenario.get(m.ledger_id) for m in tier1_outcome.matches}
    assert matched_scenarios == {"clean"}, (
        f"Tier 1 matched scenario(s) other than 'clean': {matched_scenarios - {'clean'}}"
    )
