# Phase 4 provider blockers - the real trail

The project brief specified the Anthropic API (`claude-sonnet-4-6`). Getting
a working, real API call for Phase 4 took three provider attempts, each
blocked at the account/billing level rather than by a code defect. This is
part of the honest "what broke" account for this build (see
`reports/metrics_report.md` for the fuller narrative).

## 1. Anthropic (claude-sonnet-4-6) - blocked

```
anthropic.BadRequestError: Error code: 400 - {
  'type': 'error',
  'error': {
    'type': 'invalid_request_error',
    'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'
  },
  'request_id': 'req_011CeaeEj7VnNvxc3F41g8tA'
}
```

`.env` / `python-dotenv` wiring, the `require_api_key()` fail-loudly check,
and the request construction were all confirmed correct - this was purely a
billing/credits issue on the account.

## 2. OpenAI (gpt-4o-mini) - blocked

```
openai.RateLimitError: Error code: 429 - {
  'error': {
    'message': 'You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.',
    'type': 'insufficient_quota',
    'code': 'credit_balance_exhausted'
  }
}
```

Same story: correct wiring, no usable balance on the account.

## 3. Google Gemini - first key blocked, second key worked

First `GEMINI_API_KEY`: authenticated successfully and could list ~50
available models, but every `generate_content` call against a current model
returned:

```
google.genai.errors.ClientError: 403 PERMISSION_DENIED. {
  'error': {'code': 403, 'message': 'Your project has been denied access. Please contact support.', 'status': 'PERMISSION_DENIED'}
}
```

(older/retired models returned 404 "no longer available to new users"
instead). This looked like the underlying Google Cloud project was blocked
outright, not a per-model quota issue.

A second `GEMINI_API_KEY` (different project) was generated and retried.
`gemini-flash-latest` hit a transient `503 UNAVAILABLE` (high demand) on
several attempts, but `gemini-3.5-flash-lite` succeeded immediately:

```
'OK'
prompt_token_count=8  candidates_token_count=1  total_token_count=9
```

**Final choice for Phase 4: `gemini-3.5-flash-lite` via `GEMINI_API_KEY`,
free tier, $0 cost.** Both `llm_naive_experiment.py` (Step A) and
`llm_adjudicator.py` (Step B) run against this model. The tiered-pipeline
design and the naive-vs-disciplined comparison methodology are
provider-agnostic - only the SDK call and model name changed from the
brief's original Anthropic specification.

**This is the accepted production choice for this submission, not a
placeholder awaiting an Anthropic re-run.** The three-provider billing-wall
trail above is kept as honest "what broke" material, but Gemini is the
model the pipeline is built and evaluated against going forward. One
accepted tradeoff that follows directly from this choice: Gemini's free
tier caps `gemini-3.5-flash-lite` at 15 requests/minute
(`FREE_TIER_REQUESTS_PER_MINUTE` in `llm_adjudicator.py`), which bounds
Tier 3's throughput on a large exception queue - see
`reports/metrics_report.md` for the measured throughput and how it factors
into the track's "Throughput plus measured accuracy" bar. Because Tier 3
only ever adjudicates the small remainder Tier 1+2 leave behind (22 of 140
events in this dataset, 15.7%), this rate limit is a real but bounded
constraint, not a bottleneck on the pipeline's overall throughput.
