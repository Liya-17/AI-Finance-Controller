# For reviewers — read this first

One page. Everything below is verifiable against the repo, not asserted.

## The result, in three numbers (not one)

| Bucket | Count | % of 140 | What it means |
|---|---:|---:|---|
| Full match (3-way) | 118 | 84.3% | Ledger + gateway + bank all reconciled |
| Confirmed partial match | 14 | 10.0% | Correctly resolved to 2-of-3, by design (not a shortfall) |
| Flagged exception | 8 | 5.7% | Genuine ambiguity, correctly left for a human |

100% precision measured at every tier. These three numbers are **never**
summed into one "resolved %" anywhere in this repo — see why in
[`src/audit_log.py`](src/audit_log.py)'s module docstring.

## Live dashboard

**[PASTE_STREAMLIT_CLOUD_URL_HERE](PASTE_STREAMLIT_CLOUD_URL_HERE)** — read-only,
no setup, no login. See [`reports/deploy_instructions.md`](reports/deploy_instructions.md)
if this link isn't live yet.

## Reproduce in 30 seconds — no API key required

```bash
pip install -r requirements.txt
python scripts/print_verified_metrics.py
```

Or `make demo` (Linux/Mac) / `bash run_demo.sh` (Windows / no-make). Prints
Tier 1 and Tier 2 recomputed live and re-verified against
`data/ground_truth.csv`, plus the three-bucket outcome, in under 10 seconds.
Tier 3 reuses its committed, already-verified decisions
(`reports/tier3_adjudication_results.json`) rather than calling an LLM.

## What to look at, and why (5 files, ~15 minutes)

1. **[`src/matchers/llm_naive_experiment.py`](src/matchers/llm_naive_experiment.py)**
   — the naive-vs-tiered experiment. Actually run (not simulated) against
   the same 76-event pool the tiered pipeline handles: 82.7% precision,
   9 of 13 wrong matches force-matched genuinely-orphan events with
   confident, invented rationale. This is the measured argument for the
   tiered design — see the module docstring and
   `reports/naive_llm_experiment_summary.json` for the raw numbers.

2. **[`src/matchers/fuzzy_matcher.py:238-240`](src/matchers/fuzzy_matcher.py#L238-L240)**
   — the tie-detection guard. An early version of Tier 2 let all 9
   `duplicate`-scenario events leak through as confident matches (its own
   printed precision looked like 100% until checked against ground truth).
   This is the fix: if more than one candidate on any side ties for a
   match, none of them are matched — ambiguity is excluded, never guessed.
   `tests/test_tier2.py::test_no_scenario_leakage` regression-tests this
   exact bug class.

3. **[`src/matchers/llm_adjudicator.py:128`](src/matchers/llm_adjudicator.py#L128)**
   (`SYSTEM_INSTRUCTION`) — the abstention-friendly Tier 3 prompt. Explicitly
   tells the model that a one-sided (2-of-3) match is a *complete, correct*
   answer for orphan-shaped rows, and that abstaining is better than
   guessing. The first version of this prompt didn't say that — it produced
   5 wrongly-unresolved orphans out of 13 (the model found the true
   counterpart and then declined to call it a match). One prompt change
   fixed all 5; see `reports/metrics_report.md`'s "What broke" section for
   the before/after.

4. **[`src/audit_log.py`](src/audit_log.py)** — the three-bucket audit trail.
   Every decision from every tier is tagged `full_match` / `partial_match`
   / `flagged_exception`, with resolving tier, confidence, rationale,
   provider, and model. `tests/test_audit_log.py::test_three_bucket_sum`
   asserts the three buckets sum to exactly 140 with no event lost or
   double-counted.

5. **[`src/providers.py`](src/providers.py)** — the provider adapter layer.
   One interface, three backends (Gemini/Anthropic/OpenAI), selected by
   `LLM_PROVIDER`. `GeminiProvider` is fully verified (it produced every
   real result in this repo); `AnthropicProvider`/`OpenAIProvider` are
   written to each SDK's real structured-output shape but unverified — both
   accounts were blocked on billing for the whole build (see
   `reports/llm_provider_blockers.md`). This file states that precisely,
   not inflated.

## What this deliberately does not claim

- **Not "100% accurate."** Tier 3's 100% precision is measured on **n=14**
  match verdicts — a small sample, reported as such
  (`reports/metrics_report.md` says "100% on this run (n=14)", not an
  unqualified 100%). The 8 flagged exceptions are correctly *not* resolved,
  not swept into a success number.
- **Not the originally specified model.** The brief asked for
  `claude-sonnet-4-6`. This build runs on Google Gemini
  (`gemini-3.5-flash-lite`, free tier) after Anthropic and OpenAI accounts
  both ran out of credits mid-build. The provider-agnostic adapter
  (`src/providers.py`) means trying Claude for real is one env var and one
  command away, not a rewrite — but it hasn't actually been run, and this
  file says so rather than implying otherwise.
- **Not tested at scale.** All numbers above are on a 140-event synthetic
  dataset. Nothing in this repo claims performance or precision at
  production transaction volumes.
- **Not a finished human-review product.** The exception queue is currently
  read-only in the dashboard — flagged rows are visible with full rationale,
  but there's no reviewer-resolution workflow yet.
- **Not measuring real Razorpay data.** The dataset is synthetic
  (`data/generate_synthetic_data.py`, seeded, reproducible), with injected
  failure modes designed to resemble real reconciliation breaks, but it is
  not real transaction data.

## Where the numbers come from

Every figure above traces to a file, not a claim: `reports/metrics_report.md`
(full report), `reports/audit_log.csv` / `.json` (row-level provenance),
`reports/tier3_adjudication_results.json` (every LLM decision + rationale),
`reports/naive_llm_experiment_summary.json` (the naive-baseline run),
`reports/llm_provider_blockers.md` (the provider trail). `architecture.md`
has the pipeline diagram and design rationale.
