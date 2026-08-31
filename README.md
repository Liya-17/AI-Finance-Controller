# Multi-Source Reconciliation Agent

A tiered reconciliation pipeline that matches transactions across three
independent banking-ops sources — an internal core ledger, a payment
gateway settlement file, and a bank statement — with injected realistic
failure modes (truncated IDs, timezone-shifted dates, split settlements,
rounding drift, duplicates, near-match merchant names, and genuine orphans),
scored against a hidden ground truth.

Built for the Razorpay AI Buildathon, Track 04: AI Finance Controller.

**Result:** 118/140 events fully reconciled (84.3%), 14/140 confirmed
partial matches (10.0%), 8/140 correctly flagged as exceptions (5.7%) —
100% precision measured at every tier. See
[`reports/metrics_report.md`](reports/metrics_report.md) for the full
numbers, the naive-vs-disciplined LLM comparison, and an honest account of
what broke during the build. See [`architecture.md`](architecture.md) for
the pipeline design and rationale.

## Reproduce in 30 seconds — no API key needed

Every Tier 3 (LLM) decision from a real run is committed to this repo at
[`reports/tier3_adjudication_results.json`](reports/tier3_adjudication_results.json).
`src/pipeline.py` reuses that cache by default, so the entire pipeline —
data generation, all three matching tiers, the audit log, and the test
suite — reproduces the exact numbers above **with zero API calls and no
`.env` file**. Verified by literally removing `.env` and re-running this:

```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py --records 400 --injection-rate 0.55 --seed 42
python src/pipeline.py
pytest tests/ -v
```

Or, one command: `bash run_all.sh` (POSIX shell; on Windows, run it from
Git Bash or activate the venv first — see Setup below).

You only need `GEMINI_API_KEY` if you want to *re-run* Tier 3 against a
live model (`python src/pipeline.py --rerun-tier3`) or replay the naive
baseline experiment (`llm_naive_experiment.py`) — see Phase 4 below.

## Highlights: what broke

The full account is in
[`reports/metrics_report.md`](reports/metrics_report.md#what-broke-and-how-it-was-fixed-the-honest-account),
but the highlights:

- **A self-reported "100% precision" that was actually a false positive
  leak.** Tier 2's first version let all 9 `duplicate`-scenario events
  (two raw-identical candidate rows) leak through as confident matches,
  because the ambiguous-tie guard that already protected Tier 1 hadn't
  been carried into Tier 2's greedy scorer — the tool's own printed
  precision number looked perfect right up until it was checked against
  ground truth.
- **A "correct" prompt that quietly taught the model to under-resolve.**
  Tier 3's adjudicator initially never told the model that a one-sided
  (2-of-3) match is a *complete* answer for orphan-shaped rows — so it
  kept finding the true counterpart and then declining to call it a match,
  reasoning that "complete reconciliation requires both gateway and bank
  confirmation." Result: 5 of 13 orphans wrongly left unresolved. One
  prompt change (stating explicitly that a partial match is valid, not
  incomplete) fixed all 5.
- The naive single-LLM-call baseline (`llm_naive_experiment.py`), run for
  real against the same 76-event pool the tiered pipeline handles, scored
  82.7% precision with 9 of 13 wrong matches force-matching orphan events
  that genuinely have no counterpart — confident, plausible-sounding, and
  false. This is the measured argument for the tiered design, not an
  assumption.
- Three LLM providers were evaluated before landing on one with usable
  credits — Anthropic and OpenAI both blocked on billing, a first Gemini
  key blocked at the project level. That trail turned into a real design
  decision: Tier 3 now runs behind a provider-agnostic adapter
  ([`src/providers.py`](src/providers.py)), not a hardcoded SDK call. See
  the Provider note below.

## Provider note: provider-agnostic by design

Tier 3 goes through one interface — [`src/providers.py`](src/providers.py)
— with three backends (Gemini, Anthropic, OpenAI) behind it, selected by
`LLM_PROVIDER`:

```bash
LLM_PROVIDER=gemini python src/pipeline.py --rerun-tier3      # default, verified
LLM_PROVIDER=anthropic python src/pipeline.py --rerun-tier3   # claude-sonnet-4-6
LLM_PROVIDER=openai python src/pipeline.py --rerun-tier3      # gpt-4o-mini
```

This is demonstrable, not just asserted: `llm_adjudicator.py` never imports
a provider SDK directly, and the response schema
(`llm_adjudicator.RESPONSE_SCHEMA`) is written in standard JSON Schema, with
Gemini-dialect translation (`nullable: true` vs. `"type": ["string",
"null"]`) isolated inside `GeminiProvider` — a real incompatibility found
and handled, not glossed over.

**Verification status, stated precisely:** `GeminiProvider` is fully
verified — it's what every real run in this repo, including the committed
`reports/tier3_adjudication_results.json`, actually used.
`AnthropicProvider` and `OpenAIProvider` are written to each SDK's
documented structured-output shape but have not been run against a live
account — both were blocked on billing for the entire build (the brief
specified `claude-sonnet-4-6`; Anthropic and OpenAI credits ran out before
either could be tested). Full trail of the raw errors:
[`reports/llm_provider_blockers.md`](reports/llm_provider_blockers.md). If
you have Anthropic or OpenAI credits, running Tier 3 against the originally
specified model is the one command above — no other code change needed —
and confirming it works would remove this caveat entirely.

## Setup

```bash
python -m venv myenv
# Windows:
myenv\Scripts\Activate.ps1
# macOS/Linux:
source myenv/bin/activate

pip install -r requirements.txt
```

**Optional — only needed to re-run Tier 3 live instead of using the
committed cache** (see "Reproduce in 30 seconds" above for the no-key
path). Create a `.env` file in the project root (never commit this — it's
already in `.gitignore`):

```
GEMINI_API_KEY=your-key-here
```

Get a free-tier key at [aistudio.google.com](https://aistudio.google.com/apikey).
`llm_adjudicator.py` and `llm_naive_experiment.py` fail loudly with a clear
error message if this is missing and you try to call the live API — they
never silently skip the LLM step or fall back to a mocked response.

## Running the pipeline

Each phase can be run individually, or the whole thing end-to-end via
`src/pipeline.py`. Run from the project root.

### Phase 1 — generate synthetic data

```bash
python data/generate_synthetic_data.py --records 400 --injection-rate 0.55 --seed 42
```

Produces `data/core_ledger.csv`, `data/gateway_settlement.csv`,
`data/bank_statement.csv`, and `data/ground_truth.csv` (hidden mapping,
used only for scoring — the matchers never read it).

### Phase 2 — Tier 1: exact matcher

```bash
python src/matchers/exact_matcher.py
```

Raw `(txn_id, amount, date)` three-way match, no normalization. Prints
match count, ground-truth-verified precision, and the matched-scenario
breakdown.

### Phase 3 — Tier 2: fuzzy/algorithmic matcher

```bash
python src/matchers/fuzzy_matcher.py
```

Runs on whatever Tier 1 left unresolved. Handles ID normalization,
date-window tolerance, amount tolerance, name fuzzing, and split-settlement
subset-sum grouping (reported separately from the pairwise cases).

### Phase 4 — Tier 3: LLM adjudication

**Step A (naive baseline — the "what broke" experiment):**

```bash
python src/matchers/llm_naive_experiment.py
```

Dumps the *entire* Tier-1 remainder into one LLM call and measures what
goes wrong. Costs real API calls — don't run this repeatedly.

**Step B (disciplined adjudicator — production Tier 3):**

```bash
python src/matchers/llm_adjudicator.py
```

Only adjudicates the true remainder after Tier 1+2 (a fraction of Step A's
scope). One call per row, structured JSON output, rate-limited to stay
under Gemini's free-tier 15 requests/minute. Results are cached to
`reports/tier3_adjudication_results.json` so later runs of `pipeline.py`
don't need to re-call the API.

### Phase 5 — exceptions + audit trail

```bash
python src/exceptions.py   # categorizes flagged exceptions from the cached Tier 3 results
python src/pipeline.py     # runs Tier 1 -> 2 -> 3 end-to-end, writes the audit log + exception queue
```

`pipeline.py` reuses the cached Tier 3 results by default. Pass
`--rerun-tier3` to hit the live API again instead:

```bash
python src/pipeline.py --rerun-tier3
```

### Phase 6 — dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. Read-only view over
`reports/audit_log.csv` and `reports/exception_queue.csv` — run
`src/pipeline.py` first if those don't exist yet. No LLM calls happen from
the dashboard itself.

### Phase 7 — reports

Already generated in `reports/metrics_report.md`. Regenerate the audit log
and exception queue that feed it (and the dashboard) with:

```bash
python src/pipeline.py
```

## Project structure

```
data/
  generate_synthetic_data.py   Phase 1 — synthetic data + ground truth generator
  core_ledger.csv               generated
  gateway_settlement.csv        generated
  bank_statement.csv            generated
  ground_truth.csv              generated (hidden mapping, scoring only)
src/
  matchers/
    exact_matcher.py            Phase 2 — Tier 1
    fuzzy_matcher.py             Phase 3 — Tier 2
    llm_naive_experiment.py     Phase 4 Step A — naive baseline (the "what broke" experiment)
    llm_adjudicator.py           Phase 4 Step B — Tier 3, disciplined
  audit_log.py                  Phase 5 — three-bucket audit trail
  exceptions.py                  Phase 5 — exception categorization
  pipeline.py                    end-to-end runner (Tier 1 -> 2 -> 3 -> audit -> exceptions)
dashboard/
  app.py                         Phase 6 — Streamlit dashboard
reports/
  metrics_report.md              Phase 7 — precision/recall, throughput, honest exception summary
  llm_provider_blockers.md       the 3-provider billing-wall trail
  audit_log.csv / .json          generated
  exception_queue.csv / .json    generated
  naive_llm_*                     generated (Step A raw output + summary)
  tier3_adjudication_*            generated (Step B results + summary)
architecture.md                  pipeline diagram + design rationale
requirements.txt
.gitignore
```

## Tech stack

Python, pandas, rapidfuzz, Faker, Streamlit, Google Gemini API
(`google-genai`).
