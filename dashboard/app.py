"""
Phase 6 - Streamlit dashboard for the reconciliation pipeline.

Reads the artifacts pipeline.py (Phase 5) already produces:
  reports/audit_log.csv       - every tier's decision, three-bucket tagged
  reports/exception_queue.csv - categorized flagged exceptions

Does NOT re-run the pipeline or call any LLM - this is a read-only view over
already-computed, already-verified results. If those files don't exist yet,
run `python src/pipeline.py` first.

Design discipline carried over from the rest of this project: the top-line
metrics are the three buckets (full match / partial match / flagged
exception), never collapsed into one ambiguous "resolved %" - see
src/audit_log.py's module docstring for why that distinction matters here.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"

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
    if not audit_path.exists():
        return None, None
    audit_df = pd.read_csv(audit_path)
    exc_df = pd.read_csv(exc_path) if exc_path.exists() else pd.DataFrame()
    return audit_df, exc_df


audit_df, exc_df = load_data()

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
# Breakdown charts: by outcome bucket, and by resolving tier
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
    st.markdown("**Resolved by tier**")
    labels = [TIER_LABELS[t] for t in tier_counts.index]
    colors = [TIER_COLORS[t] for t in tier_counts.index]
    fig = go.Figure(
        go.Bar(
            x=tier_counts.values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=tier_counts.values,
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

        st.dataframe(
            filtered[["ledger_id", "category", "verdict", "confidence"]].rename(
                columns={
                    "ledger_id": "Ledger ID",
                    "category": "Category",
                    "verdict": "Tier 3 verdict",
                    "confidence": "Confidence",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Drill down: click a row to see why it wasn't resolved**")
    for _, row in filtered.iterrows():
        with st.expander(f"{row['ledger_id']}  —  {CATEGORY_LABELS.get(row['category'], row['category'])}"):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.metric("Confidence", f"{row['confidence']:.2f}")
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
