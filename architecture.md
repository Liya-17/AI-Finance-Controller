# Architecture

## Pipeline overview

```
                 ┌─────────────────┐
                 │  data/generate_  │
                 │  synthetic_data  │   (Phase 1 — one-time, produces the
                 │       .py        │    3 source CSVs + hidden ground_truth.csv)
                 └────────┬─────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
 core_ledger.csv  gateway_settlement  bank_statement.csv
   (140 rows)      .csv (154 rows)      (137 rows)
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
        ┌──────────────────────────────────────┐
        │  TIER 1 — exact_matcher.py             │
        │  raw (txn_id, amount, date), 3-way,    │
        │  no normalization, no tolerance        │
        │                                         │
        │  64/140 matched · 100% precision        │
        │  · 0.018s                               │
        └──────────────────┬──────────────────────┘
                            │ 76 ledger rows unresolved
                            ▼
        ┌──────────────────────────────────────┐
        │  TIER 2 — fuzzy_matcher.py              │
        │  ┌────────────────┐ ┌─────────────────┐│
        │  │ subset-sum       │ │ pairwise scorer ││
        │  │ split_settlement │ │ id_truncated,   ││
        │  │ (one-to-many)    │ │ timezone_shift, ││
        │  │                  │ │ rounding_drift, ││
        │  │                  │ │ near_match_name ││
        │  └────────────────┘ └─────────────────┘│
        │  each candidate independently gated;    │
        │  ambiguous ties excluded, not guessed    │
        │                                          │
        │  54/76 matched · 100% precision/recall   │
        │  · 0.466s · 0 duplicate/orphan leaks     │
        └──────────────────┬───────────────────────┘
                            │ 22 ledger rows unresolved
                            │ (9 duplicate + 13 orphan, by construction)
                            ▼
        ┌──────────────────────────────────────┐
        │  TIER 3 — llm_adjudicator.py            │
        │  one call per row, structured JSON      │
        │  output, Gemini gemini-3.5-flash-lite   │
        │  Verdict: match / no_match / uncertain  │
        │  Partial (2-of-3) match is a VALID,     │
        │  complete verdict — never force-        │
        │  completed to 3-way                     │
        │                                          │
        │  14 match (100% on this run, n=14)      │
        │  8 uncertain → exception queue          │
        └──────────────────┬───────────────────────┘
                            ▼
        ┌──────────────────────────────────────┐
        │  audit_log.py                           │
        │  every decision → 3-bucket tag:          │
        │   full_match / partial_match /           │
        │   flagged_exception                      │
        │  + resolving_tier, confidence, rationale,│
        │    provider, model, timestamp            │
        └────────┬─────────────────────┬───────────┘
                 │                     │
                 ▼                     ▼
        ┌────────────────┐   ┌──────────────────┐
        │ exceptions.py    │   │ dashboard/app.py  │
        │ categorizes       │   │ Streamlit,        │
        │ flagged_exception │   │ 3-bucket metrics, │
        │ rows (duplicate_  │   │ drill-down,       │
        │ ambiguity, timing_│   │ filterable audit  │
        │ mismatch, etc.)   │   │ trail             │
        └────────────────┘   └──────────────────┘

  pipeline.py ties Tier 1 → 2 → 3 → audit_log → exceptions together end to end.
```

## Why a tiered pipeline, not one LLM call

This was measured, not assumed — `src/matchers/llm_naive_experiment.py`
(Phase 4, Step A) actually ran a single naive LLM call against the exact
same 76-event pool Tier 2+3 handle, and the result is the argument for the
tiered design:

| | Single LLM call (naive) | Tiered pipeline |
|---|---|---|
| Precision | 82.7% (62/75 claimed matches correct) | 100% at every tier |
| Failure mode | 13 wrong matches, **9 of which force-matched orphan events** that genuinely have no counterpart, with confident invented rationale | 0 measured false positives; orphans resolve to a correct one-sided match or are flagged, never guessed |
| Cost driver | Every row pays LLM latency + tokens, including the ~84% that a raw equality check would have caught for free | Only ~16% of events (22/140) ever reach an LLM call |
| Auditability | One large, hard-to-verify response covering everything at once | Every decision individually logged with its resolving tier and rationale |

The core reasoning: **an LLM is a probabilistic tool being asked to do a
mix of deterministic and genuinely ambiguous work in one pass.** For the
deterministic ~84% (identical IDs, IDs that only need normalizing, dates
that only need a tolerance window, amounts within a known rounding band),
a rule is strictly better — it's free, instant, and cannot hallucinate.
Handing that work to an LLM anyway doesn't just waste cost; it gives the
model more surface area to be wrong on things a rule would get right for
free, and dilutes its attention on the small slice of rows that actually
need judgment. The tiered design routes each row to the cheapest tool that
can honestly resolve it, and only spends LLM calls — and their nonzero
error rate — on rows that survived two rule-based passes and are still
ambiguous by construction (structurally-tied duplicates, structurally
one-sided orphans).

## Design rationale: precision over recall, at every tier

Every tier in this pipeline is built so that **a wrong match is treated as
worse than a missed one** — a miss falls through honestly to the next tier
or the exception queue; a wrong match becomes settled truth for everything
downstream and corrupts the audit trail. Concretely:

- **Tier 1** requires an *unambiguous* single candidate on each side. If two
  rows tie on the same raw key (e.g. a duplicate write), Tier 1 declines
  rather than picking one arbitrarily.
- **Tier 2** carries the same ambiguous-tie discipline forward explicitly
  (this had to be added after an early version let duplicate rows leak
  through via greedy highest-confidence assignment — see
  `reports/metrics_report.md` for the full story), plus a
  `CONFIDENCE_THRESHOLD` below which nothing is matched.
- **Tier 3**'s prompt explicitly instructs the model that abstaining
  (`no_match` / `uncertain`) is a correct and expected outcome, not a
  failure to find something — directly countering the force-matching
  behavior observed in the naive Step A run.

## Three-bucket outcome model

`full_match` (all three sources reconciled), `partial_match` (a confirmed,
correct match on only one of gateway/bank — the true and complete answer
for orphan-shaped rows, not an incomplete attempt), and `flagged_exception`
(genuine ambiguity, left for a human). These are never summed into one
"resolved %" — see `src/audit_log.py`'s module docstring and
`reports/metrics_report.md` for why that distinction is load-bearing for
honest reporting, not a formality.

## Provider note

The brief specified the Anthropic API (`claude-sonnet-4-6`). Three
providers were evaluated in sequence — Anthropic, OpenAI, and Google
Gemini — with the first two blocked on account credits and a first Gemini
key blocked at the project level; a second Gemini key worked on the free
tier. Google Gemini (`gemini-3.5-flash-lite`) is the deliberate model this
pipeline runs Tier 3 against for this submission. Full trail of raw errors:
`reports/llm_provider_blockers.md`. The adjudication design (structured
JSON schema, one-call-per-row, the abstention-friendly prompt) is
provider-agnostic — swapping back to Claude is a model-name and SDK-call
change in `src/matchers/llm_adjudicator.py`, not a redesign.
