# Multi-Source Reconciliation Agent

A tiered exact → fuzzy → LLM pipeline that reconciles transactions across
ledger, payment-gateway, and bank sources — with measured, ground-truth-
verified precision at every tier, not asserted accuracy.

**🔴 Live dashboard: [PASTE_STREAMLIT_CLOUD_URL_HERE](PASTE_STREAMLIT_CLOUD_URL_HERE)**
*(deploy via [`reports/deploy_instructions.md`](reports/deploy_instructions.md), then replace this line)*

[![Tests](https://github.com/Liya-17/AI-Finance-Controller/actions/workflows/test.yml/badge.svg)](https://github.com/Liya-17/AI-Finance-Controller/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**👉 [Reviewing this? Read `JUDGE.md`](JUDGE.md)** — one page: the headline
result, what to look at and why, and what this deliberately does not claim.

Built for the Razorpay AI Buildathon, Track 04: AI Finance Controller.

## The result, in three numbers

| Bucket | Count | % of 140 | What it means |
|---|---:|---:|---|
| Full match (3-way) | 118 | 84.3% | Ledger + gateway + bank all reconciled |
| Confirmed partial match | 14 | 10.0% | Correctly resolved to 2-of-3, by design — not a shortfall |
| Flagged exception | 8 | 5.7% | Genuine ambiguity, correctly left for a human |

100% precision measured at every tier, verified against a hidden ground
truth the matchers never see. These three numbers are never summed into
one ambiguous "resolved %" anywhere in this repo — see
[`src/audit_log.py`](src/audit_log.py)'s module docstring for why.

*(Dashboard screenshot goes here once captured — see `reports/deploy_instructions.md`.)*

## The core argument: naive single-LLM-call vs. this tiered pipeline

Not a hypothetical — [`llm_naive_experiment.py`](src/matchers/llm_naive_experiment.py)
actually ran a single unstructured LLM call against the same 76-event pool
the tiered pipeline (Tier 2 + Tier 3) handles, so this is apples-to-apples:

| | Naive (one LLM call) | This pipeline (tiered) |
|---|---|---|
| Scope | All 76 Tier-1 leftovers, one call | Tier 2 resolves 54/76 algorithmically; Tier 3 only adjudicates the true ~22-event remainder |
| Precision | **82.7%** (62/75 claimed matches correct) | **100%** at every tier |
| Wrong matches | **13**, of which **9 force-matched orphan events** with confident, invented rationale | **0** measured false positives |
| Time / cost | 24.1s, $0 (free tier) | ~0.5s (Tier 1+2, no LLM) + ~90s for 22 rate-limited Tier 3 calls, $0 |

Full breakdown, raw output, and the "what broke" story behind these numbers:
[`reports/metrics_report.md`](reports/metrics_report.md). Pipeline design
rationale: [`architecture.md`](architecture.md).

## Reproduce in 30 seconds — no API key needed

Every Tier 3 (LLM) decision from a real run is committed to this repo at
[`reports/tier3_adjudication_results.json`](reports/tier3_adjudication_results.json).
The pipeline reuses that cache by default, so the entire thing — Tier 1 and
Tier 2 recomputed live and re-verified against `data/ground_truth.csv`,
plus the three-bucket outcome — reproduces the numbers above **with zero
API calls and no `.env` file**, in a few seconds:

```bash
pip install -r requirements.txt
python scripts/print_verified_metrics.py
```

Or `make demo` (Linux/Mac) / `bash run_demo.sh` (Windows / no-make).
Verified by literally removing `.env` and re-running the whole sequence —
identical results.

You only need an API key (`GEMINI_API_KEY` by default — see "Provider-
agnostic by design" below for other providers) if you want to *re-run*
Tier 3 against a live model instead of the committed cache.

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
- Three LLM providers were evaluated before landing on one with usable
  credits — Anthropic and OpenAI both blocked on billing, a first Gemini
  key blocked at the project level. That trail turned into a real design
  decision: Tier 3 now runs behind a provider-agnostic adapter
  ([`src/providers.py`](src/providers.py)), not a hardcoded SDK call — see
  below.

See the naive-vs-tiered comparison table above for the third, and biggest,
piece of "what broke" evidence.

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
pip install -r requirements-dashboard.txt   # or `make dashboard`
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. Read-only view over
`reports/audit_log.csv` and `reports/exception_queue.csv` (both committed
to this repo) — run `src/pipeline.py` first if you've regenerated data and
want the dashboard to reflect it. No LLM calls happen from the dashboard
itself, and it never imports `google-genai`/`anthropic`/`openai` — that's
why it has its own lighter `requirements-dashboard.txt`.

### Phase 7 — reports

Already generated in `reports/metrics_report.md`. Regenerate the audit log
and exception queue that feed it (and the dashboard) with:

```bash
python src/pipeline.py
```

## Project structure

```
JUDGE.md                        one-page reviewer guide — read this first
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
  providers.py                  LLM provider adapter (Gemini/Anthropic/OpenAI, one interface)
  audit_log.py                  Phase 5 — three-bucket audit trail
  exceptions.py                  Phase 5 — exception categorization
  pipeline.py                    end-to-end runner (Tier 1 -> 2 -> 3 -> audit -> exceptions)
dashboard/
  app.py                         Phase 6 — Streamlit dashboard (needs only requirements-dashboard.txt)
scripts/
  print_verified_metrics.py     the `make demo` / `run_demo.sh` entry point
tests/                          pytest suite, no live LLM calls (see .github/workflows/test.yml)
reports/
  metrics_report.md              Phase 7 — precision/recall, throughput, honest exception summary
  llm_provider_blockers.md       the 3-provider billing-wall trail
  deploy_instructions.md         Streamlit Community Cloud deploy steps
  audit_log.csv / .json          generated
  exception_queue.csv / .json    generated
  naive_llm_*                     generated (Step A raw output + summary)
  tier3_adjudication_*            generated (Step B results + summary)
architecture.md                  pipeline diagram + design rationale
requirements.txt                 full project (matchers, tests, all 3 LLM SDKs)
requirements-dashboard.txt      dashboard-only subset (pandas/streamlit/plotly) — Streamlit Cloud deploy uses this
Makefile / run_demo.sh          `make demo` and its Windows/no-make fallback
run_all.sh                       longer reproduce path (regenerates data too, runs pytest)
.github/workflows/test.yml      CI — runs the test suite on every push
LICENSE                          MIT
.gitignore
```

## Tech stack

Python, pandas, rapidfuzz, Faker, Streamlit, Plotly. LLM: Google Gemini
(`google-genai`, verified) behind a provider-agnostic adapter that also
supports Anthropic and OpenAI SDKs (written, unverified — see "Provider-
agnostic by design" above).
