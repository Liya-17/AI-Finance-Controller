"""
The three-bucket outcome model (full_match / partial_match /
flagged_exception - see src/audit_log.py's module docstring) exists
specifically so results are never collapsed into one ambiguous "resolved %"
number. This test locks in the one invariant that model depends on: every
event lands in exactly one bucket, and the three buckets sum to the total
event count with nothing lost or double-counted.

Uses the cached reports/audit_log.csv (produced by src/pipeline.py) rather
than re-running Tier 3 live, so this test is free and fast - see the module
docstring on conftest.py.
"""

VALID_BUCKETS = {"full_match", "partial_match", "flagged_exception"}


def test_three_bucket_sum(audit_log_df, ground_truth):
    n_total_events = len(ground_truth)

    assert set(audit_log_df["outcome_bucket"].unique()) <= VALID_BUCKETS, (
        f"audit log contains an outcome_bucket outside {VALID_BUCKETS}: "
        f"{set(audit_log_df['outcome_bucket'].unique()) - VALID_BUCKETS}"
    )

    bucket_counts = audit_log_df["outcome_bucket"].value_counts()
    total_logged = bucket_counts.sum()

    assert total_logged == n_total_events, (
        f"audit log has {total_logged} entries but ground truth has {n_total_events} events - "
        f"some event is missing or double-logged"
    )

    full_match = bucket_counts.get("full_match", 0)
    partial_match = bucket_counts.get("partial_match", 0)
    flagged_exception = bucket_counts.get("flagged_exception", 0)

    assert full_match + partial_match + flagged_exception == n_total_events, (
        f"bucket counts ({full_match} + {partial_match} + {flagged_exception}) "
        f"don't sum to {n_total_events}"
    )

    # every ledger_id appears exactly once - no event silently logged twice
    assert audit_log_df["ledger_id"].duplicated().sum() == 0, "audit log has duplicate ledger_id entries"
