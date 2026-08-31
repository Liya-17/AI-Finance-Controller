"""
Phase 6 dashboard, Phase 2 (dashboard fixes + additions) revision.

Reads the artifacts pipeline.py already produces:
  reports/audit_log.csv           - every tier's decision, three-bucket tagged
  reports/exception_queue.csv     - categorized flagged exceptions
  reports/tier3_calibration.json  - per-row (confidence, verdict, correct) for Tier 3
  reports/naive_llm_experiment_summary.json - the naive-baseline run's real numbers
  data/ground_truth.csv           - for the per-scenario recall table (joined at render
                                      time, never hardcoded - see build_scenario_table())

Does NOT re-run the pipeline or call any LLM - read-only view over already-
computed, already-verified results. If reports/audit_log.csv doesn't exist,
run `python src/pipeline.py` first.

Design discipline carried over from the rest of this project: the top-line
metrics are the three buckets (full match / partial match / flagged
exception), never collapsed into one ambiguous "resolved %" - see
src/audit_log.py's module docstring for why that distinction matters here.
The fixed status/categorical palette below is deliberate (green/amber/orange
= the three buckets; blue/aqua/violet = the three tiers) and is preserved
across every chart, not restyled per-section.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
DATA_DIR = ROOT_DIR / "data"

# Status palette (fixed, never themed) - reserved meaning, not decorative
COLOR_GOOD = "#0ca30c"       # full_match - fully reconciled
COLOR_WARNING = "#fab219"    # partial_match - correct but incomplete by design
COLOR_SERIOUS = "#ec835a"    # flagged_exception - needs human review
COLOR_MUTED = "#6b7280"

# Categorical palette, fixed order (slot 1/2/3 - never cycled/reassigned)
TIER_COLORS = {
    "tier1_exact": "#2a78d6",   # blue
    "tier2_fuzzy": "#1baf7a",   # aqua
    "tier3_llm": "#4a3aa7",     # violet
}
TIER_LABELS = {
    "tier1_exact": "Tier 1 (exact match)",
    "tier2_fuzzy": "Tier 2 (fuzzy/algorithmic)",
    "tier3_llm": "Tier 3 (LLM adjudication)",
}
BUCKET_COLORS = {
    "full_match": COLOR_GOOD,
    "partial_match": COLOR_WARNING,
    "flagged_exception": COLOR_SERIOUS,
}
BUCKET_LABELS = {
    "full_match": "Full match (3-way)",
    "partial_match": "Partial match (2-of-3, by design)",
    "flagged_exception": "Flagged exception",
}
CATEGORY_LABELS = {
    "duplicate_ambiguity": "Duplicate ambiguity",
    "timing_mismatch": "Timing mismatch",
    "data_quality_issue": "Data quality issue",
    "no_candidate_found": "No candidate found",
    "unclassified": "Unclassified",
}


st.set_page_config(page_title="Reconciliation Pipeline Dashboard", layout="wide")


@st.cache_data
def load_data():
    audit_path = REPORTS_DIR / "audit_log.csv"
    exc_path = REPORTS_DIR / "exception_queue.csv"
    gt_path = DATA_DIR / "ground_truth.csv"
    calib_path = REPORTS_DIR / "tier3_calibration.json"
    naive_path = REPORTS_DIR / "naive_llm_experiment_summary.json"
    tier3_summary_path = REPORTS_DIR / "tier3_adjudication_summary.json"

    if not audit_path.exists():
        return None, None, None, None, None, None

    audit_df = pd.read_csv(audit_path)
    exc_df = pd.read_csv(exc_path) if exc_path.exists() else pd.DataFrame()
    gt_df = pd.read_csv(gt_path) if gt_path.exists() else None
    calibration = json.loads(calib_path.read_text(encoding="utf-8")) if calib_path.exists() else []
    naive_summary = json.loads(naive_path.read_text(encoding="utf-8")) if naive_path.exists() else None
    tier3_summary = json.loads(tier3_summary_path.read_text(encoding="utf-8")) if tier3_summary_path.exists() else None

    return audit_df, exc_df, gt_df, calibration, naive_summary, tier3_summary


def build_scenario_table(audit_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per injected failure-mode scenario: count, resolved, resolving
    tier(s). Computed live by joining audit_log (ledger_id -> outcome) with
    ground_truth (ledger_id -> event -> scenario) - never hardcoded, so this
    stays correct if the dataset or its scenario mix changes.
    """
    ledger_to_event = {}
    for _, row in gt_df.iterrows():
        if pd.notna(row["ledger_ids"]):
            ledger_to_event[row["ledger_ids"]] = row["event_id"]
    event_to_scenario = dict(zip(gt_df["event_id"], gt_df["scenario"]))

    df = audit_df.copy()
    df["scenario"] = df["ledger_id"].map(ledger_to_event).map(event_to_scenario)
    df = df.dropna(subset=["scenario"])

    rows = []
    for scenario, group in df.groupby("scenario"):
        total = len(group)
        resolved = (group["outcome_bucket"] != "flagged_exception").sum()
        tiers = sorted(group["resolving_tier"].unique(), key=lambda t: list(TIER_LABELS).index(t) if t in TIER_LABELS else 99)
        tier_label = ", ".join(TIER_LABELS.get(t, t) for t in tiers)
        rows.append({
            "Scenario": scenario,
            "Count": total,
            "Resolved": int(resolved),
            "Recall": resolved / total if total else 0.0,
            "Resolving tier(s)": tier_label,
        })
    result = pd.DataFrame(rows).sort_values("Scenario").reset_index(drop=True)
    return result


audit_df, exc_df, gt_df, calibration, naive_summary, tier3_summary = load_data()

st.title("Multi-Source Reconciliation Dashboard")
st.caption(
    "Ledger + gateway settlement + bank statement, reconciled across three tiers "
    "(exact match -> fuzzy/algorithmic -> LLM adjudication). Read-only view over "
    "`reports/audit_log.csv` - run `python src/pipeline.py` to regenerate."
)

if audit_df is None:
    st.error(
        "No audit log found at `reports/audit_log.csv`. Run the pipeline first:\n\n"
        "```\npython src/pipeline.py\n```"
    )
    st.stop()

total_events = len(audit_df)
bucket_counts = audit_df["outcome_bucket"].value_counts().reindex(
    ["full_match", "partial_match", "flagged_exception"], fill_value=0
)
tier_counts = audit_df["resolving_tier"].value_counts().reindex(
    ["tier1_exact", "tier2_fuzzy", "tier3_llm"], fill_value=0
)

# ---------------------------------------------------------------------------
# Top: three-bucket summary metrics - never a single "resolved %" tile
# ---------------------------------------------------------------------------
st.subheader("Reconciliation outcome")
st.caption(
    "These three numbers are reported separately, not summed into one "
    "“resolved” percentage — a full match and a confirmed partial "
    "match are both correct outcomes, but they are not the same claim."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total events", f"{total_events}")
col2.metric(
    "Full match (3-way)",
    f"{bucket_counts['full_match']} ({bucket_counts['full_match']/total_events:.1%})",
)
col3.metric(
    "Partial match (2-of-3)",
    f"{bucket_counts['partial_match']} ({bucket_counts['partial_match']/total_events:.1%})",
)
col4.metric(
    "Flagged exceptions",
    f"{bucket_counts['flagged_exception']} ({bucket_counts['flagged_exception']/total_events:.1%})",
)

st.divider()

# ---------------------------------------------------------------------------
# Breakdown charts: by outcome bucket, and by resolving tier (FIXED: Tier 3
# now shows resolved vs. flagged as separate stacked segments, not one bar
# implying all 22 of its events resolved when only 14 did)
# ---------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Outcome breakdown**")
    labels = [BUCKET_LABELS[b] for b in bucket_counts.index]
    colors = [BUCKET_COLORS[b] for b in bucket_counts.index]
    fig = go.Figure(
        go.Bar(
            x=bucket_counts.values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=bucket_counts.values,
            textposition="outside",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Events",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.markdown("**Events handled by tier**")
    st.caption(
        "Tier 1 and Tier 2 always fully resolve what they touch. Tier 3 is split: "
        "resolved (a confirmed match) vs. flagged (correctly left for human review, "
        "not a Tier 3 failure)."
    )
    tier_resolved = {}
    tier_flagged = {}
    for t in ["tier1_exact", "tier2_fuzzy", "tier3_llm"]:
        sub = audit_df[audit_df["resolving_tier"] == t]
        tier_resolved[t] = (sub["outcome_bucket"] != "flagged_exception").sum()
        tier_flagged[t] = (sub["outcome_bucket"] == "flagged_exception").sum()

    tiers_present = [t for t in ["tier1_exact", "tier2_fuzzy", "tier3_llm"] if tier_counts[t] > 0]
    labels = [TIER_LABELS[t] for t in tiers_present]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Resolved",
        x=[tier_resolved[t] for t in tiers_present],
        y=labels,
        orientation="h",
        marker_color=[TIER_COLORS[t] for t in tiers_present],
        text=[tier_resolved[t] if tier_resolved[t] else "" for t in tiers_present],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="Flagged (not resolved)",
        x=[tier_flagged[t] for t in tiers_present],
        y=labels,
        orientation="h",
        marker_color=COLOR_SERIOUS,
        marker_pattern_shape="/",
        text=[tier_flagged[t] if tier_flagged[t] else "" for t in tiers_present],
        textposition="inside",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Events",
        yaxis=dict(autorange="reversed"),
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Naive vs. tiered comparison - the project's core argument, on screen
# ---------------------------------------------------------------------------
st.subheader("Naive single-LLM-call vs. this tiered pipeline")
st.caption(
    "Not hypothetical - llm_naive_experiment.py actually ran a single unstructured "
    "LLM call against the same event pool this pipeline handles. Numbers below are "
    "read live from reports/naive_llm_experiment_summary.json."
)

if naive_summary is None:
    st.warning(
        "reports/naive_llm_experiment_summary.json not found. Run "
        "`python src/matchers/llm_naive_experiment.py` to generate it (costs a real API call)."
    )
else:
    naive_precision = naive_summary.get("precision", float("nan"))
    naive_incorrect = naive_summary.get("incorrect", 0)
    naive_total = naive_summary.get("total_claimed", 0)
    naive_examples = naive_summary.get("incorrect_examples", [])
    orphan_force_matches = sum(
        1 for ex in naive_examples if ex.get("ledger_true_scenario") == "orphan"
    )

    tiered_precision = 1.0  # measured 0 false positives across Tier 1+2+3, see reports/metrics_report.md
    tiered_wrong = 0

    ncol1, ncol2, ncol3, ncol4 = st.columns(4)
    ncol1.metric("Naive precision", f"{naive_precision:.1%}", f"{naive_incorrect}/{naive_total} wrong", delta_color="off")
    ncol2.metric("Tiered precision", f"{tiered_precision:.1%}", f"{tiered_wrong} wrong", delta_color="off")
    ncol3.metric(
        "Orphan events force-matched",
        f"{orphan_force_matches}",
        "by the naive call, with confident invented rationale",
        delta_color="off",
    )
    ncol4.metric(
        "LLM calls made",
        f"{naive_summary.get('scope_events', '?')} (naive) vs. {(tier3_summary or {}).get('total_adjudicated', '?')} (tiered)",
    )

    if naive_incorrect > 0 and orphan_force_matches == naive_incorrect:
        st.caption(
            f"Every one of the naive run's {naive_incorrect} wrong matches was an orphan event "
            f"(genuinely missing from one source) that the naive prompt confidently paired with an "
            f"unrelated row anyway — this is the audit-trail-corruption risk the tiered design exists "
            f"to prevent."
        )

st.divider()

# ---------------------------------------------------------------------------
# Per-scenario recall table - one row per injected failure mode, computed
# from ground truth at render time (never hardcoded)
# ---------------------------------------------------------------------------
st.subheader("Per-scenario recall")
st.caption(
    "Every injected failure mode, joined live from data/ground_truth.csv against "
    "reports/audit_log.csv - not a hardcoded table."
)

if gt_df is None:
    st.warning("data/ground_truth.csv not found - cannot compute the per-scenario table.")
else:
    scenario_table = build_scenario_table(audit_df, gt_df)
    st.dataframe(
        scenario_table.style.format({"Recall": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Cost panel - Phase 5 (cost economics) will add full per-call token/cost
# instrumentation to llm_adjudicator.py and llm_naive_experiment.py; until
# then this panel shows what's actually measured today (the naive run's real
# token counts, and the call-count saving, which is measurable now - 76 naive
# calls vs. 22 tiered calls, not an estimate) rather than a placeholder.
# ---------------------------------------------------------------------------
st.subheader("Cost (partial - full instrumentation lands in Phase 5)")

if naive_summary is None:
    st.caption("No cost data available yet.")
else:
    ccol1, ccol2, ccol3 = st.columns(3)
    ccol1.metric(
        "Naive run tokens",
        f"{naive_summary.get('input_tokens', 0):,} in / {naive_summary.get('output_tokens', 0):,} out",
    )
    ccol2.metric("Naive run cost", f"${naive_summary.get('estimated_cost_usd', 0):.4f}", "Gemini free tier", delta_color="off")
    n_naive_calls = naive_summary.get("scope_events", 0)
    n_tiered_calls = (tier3_summary or {}).get("total_adjudicated", 0)
    call_reduction = 1 - (n_tiered_calls / n_naive_calls) if n_naive_calls else 0
    ccol3.metric(
        "LLM call reduction",
        f"{call_reduction:.0%}",
        f"{n_naive_calls} -> {n_tiered_calls} calls, same event pool",
        delta_color="off",
    )
    st.caption(
        "Tier 3's own per-call token/latency/cost instrumentation (and the ₹/$ monthly "
        "projections at 100k/1M transactions) is Phase 5 scope - not yet built. What's shown "
        "here is real, measured data from the naive experiment plus the call-count reduction, "
        "which is directly measurable from both runs' actual scope, not an estimate."
    )

st.divider()

# ---------------------------------------------------------------------------
# Confidence calibration - bucket Tier 3 verdicts by stated confidence,
# compare to actual accuracy. Rendered honestly even when n is too small to
# conclude anything, rather than hidden.
# ---------------------------------------------------------------------------
st.subheader("Tier 3 confidence calibration")

if not calibration:
    st.caption(
        "reports/tier3_calibration.json not found - run `python src/pipeline.py` to generate it."
    )
else:
    calib_df = pd.DataFrame(calibration)
    n = len(calib_df)
    overall_accuracy = calib_df["correct"].mean()

    if n < 30:
        st.warning(
            f"**n too small to conclude.** Only {n} Tier 3 verdicts exist in this dataset - "
            f"not enough to plot a meaningful calibration curve (bucketing {n} points by "
            f"confidence would produce buckets of 1-3 points each, which is noise, not signal). "
            f"What's true and worth stating plainly: overall accuracy on this run was "
            f"{overall_accuracy:.0%} ({int(calib_df['correct'].sum())}/{n}), and every verdict's "
            f"raw (confidence, correct) pair is below so nothing is hidden - just not "
            f"over-interpreted."
        )
        display_df = calib_df[["ledger_id", "verdict", "confidence", "correct"]].sort_values(
            "confidence", ascending=False
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        # Only reached once n is large enough for real buckets - not exercised
        # by this dataset's n=22, kept so this panel is correct if the
        # dataset (or a multi-seed pooled run - see Phase 4) grows n past 30.
        calib_df["confidence_bucket"] = pd.cut(
            calib_df["confidence"], bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True
        )
        bucketed = calib_df.groupby("confidence_bucket", observed=True).agg(
            n=("correct", "size"), accuracy=("correct", "mean")
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=COLOR_MUTED),
            name="Perfect calibration", showlegend=True,
        ))
        fig.add_trace(go.Bar(
            x=[str(b) for b in bucketed["confidence_bucket"]],
            y=bucketed["accuracy"],
            marker_color=TIER_COLORS["tier3_llm"],
            text=bucketed["n"],
            textposition="outside",
            name="Actual accuracy (n per bucket)",
        ))
        fig.update_layout(
            height=320, xaxis_title="Stated confidence bucket", yaxis_title="Actual accuracy",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Exception queue: category breakdown + drill-down table
# ---------------------------------------------------------------------------
st.subheader("Exception queue")

if exc_df.empty:
    st.success("No flagged exceptions in this run.")
else:
    cat_counts = exc_df["category"].value_counts()
    cat_labels = [CATEGORY_LABELS.get(c, c) for c in cat_counts.index]

    cat_col, table_col = st.columns([1, 2])

    with cat_col:
        st.markdown("**By category**")
        fig = go.Figure(
            go.Bar(
                x=cat_counts.values,
                y=cat_labels,
                orientation="h",
                marker_color=COLOR_SERIOUS,
                text=cat_counts.values,
                textposition="outside",
            )
        )
        fig.update_layout(
            height=max(200, 60 * len(cat_counts)),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Events",
            yaxis=dict(autorange="reversed"),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with table_col:
        st.markdown("**Filter**")
        category_options = ["All"] + sorted(exc_df["category"].unique().tolist())
        selected_category = st.selectbox(
            "Category", category_options, label_visibility="collapsed"
        )
        filtered = exc_df if selected_category == "All" else exc_df[exc_df["category"] == selected_category]
        st.caption(f"{len(filtered)} of {len(exc_df)} exceptions shown")
        st.caption(
            "Confidence here is the model's confidence that this row genuinely requires human "
            "review, not confidence in any specific match — an `uncertain` verdict with high "
            "confidence means the model is sure the ambiguity is real, not a contradiction."
        )

        st.dataframe(
            filtered[["ledger_id", "category", "verdict", "confidence"]].rename(
                columns={
                    "ledger_id": "Ledger ID",
                    "category": "Category",
                    "verdict": "Tier 3 verdict",
                    "confidence": "Abstention confidence",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Rationale: expand a row below to see why it wasn't resolved**")
    for _, row in filtered.iterrows():
        with st.expander(f"{row['ledger_id']}  —  {CATEGORY_LABELS.get(row['category'], row['category'])}"):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.metric("Abstention confidence", f"{row['confidence']:.2f}")
                st.text(f"Verdict: {row['verdict']}")
            with c2:
                st.markdown("**Rationale (from Tier 3 adjudication):**")
                st.write(row["rationale"])
                if "candidates_considered" in row and pd.notna(row["candidates_considered"]):
                    st.markdown("**Candidates considered:**")
                    st.code(row["candidates_considered"], language="json")

st.divider()

# ---------------------------------------------------------------------------
# Full audit trail (all tiers, filterable)
# ---------------------------------------------------------------------------
st.subheader("Full audit trail")
st.caption("Every decision across all three tiers - timestamp, resolving tier, confidence, rationale.")

filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    bucket_filter = st.multiselect(
        "Outcome bucket",
        options=["full_match", "partial_match", "flagged_exception"],
        default=["full_match", "partial_match", "flagged_exception"],
        format_func=lambda b: BUCKET_LABELS[b],
    )
with filter_col2:
    tier_filter = st.multiselect(
        "Resolving tier",
        options=["tier1_exact", "tier2_fuzzy", "tier3_llm"],
        default=["tier1_exact", "tier2_fuzzy", "tier3_llm"],
        format_func=lambda t: TIER_LABELS[t],
    )

audit_filtered = audit_df[
    audit_df["outcome_bucket"].isin(bucket_filter) & audit_df["resolving_tier"].isin(tier_filter)
]
st.caption(f"{len(audit_filtered)} of {len(audit_df)} entries shown")
st.dataframe(
    audit_filtered[
        ["ledger_id", "outcome_bucket", "resolving_tier", "matched_gateway_id",
         "matched_bank_id", "confidence", "rationale", "provider", "model"]
    ],
    use_container_width=True,
    hide_index=True,
)
